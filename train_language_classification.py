import os
os.environ["HF_DATASETS_ENABLE_TORCHCODEC"] = "0"

import random
import hydra
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch import Trainer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import wandb

from model import SoundPhi
from torchmetrics.classification import MulticlassAccuracy

torch.set_float32_matmul_precision('medium')


def init_worker_fn(worker_id):
    os.environ["HF_DATASETS_ENABLE_TORCHCODEC"] = "0"


class SibFleursMultilingualDataset(Dataset):
    LANGUAGES = {
        "italian": "ita_Latn",
        "english": "eng_Latn",
        "turkish": "tur_Latn",
        "french":  "fra_Latn",
    }

    def __init__(self, split: str = "train"):
        super().__init__()
        self.split = split
        print(f"Initializing SibFleursDataset for split: '{self.split}'...")
        self.dataset = self._load_and_combine_data()
        print(f"Successfully loaded {len(self.dataset)} total samples.")

    def _load_and_combine_data(self):
        loaded_subsets = []
        for lang_name, lang_code in self.LANGUAGES.items():
            print(f" -> Loading {lang_name} ({lang_code})...")
            try:
                ds = load_dataset("WueNLP/sib-fleurs", name=lang_code, split=self.split)
                # FIX: capture lang_name in default arg to avoid closure bug
                ds = ds.map(lambda x, ln=lang_name: {"custom_language_label": ln})
                loaded_subsets.append(ds)
            except Exception as e:
                raise RuntimeError(f"Failed to load language {lang_name}. Error: {e}")
        return concatenate_datasets(loaded_subsets)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[int(idx)]
        audio_data = sample.get("audio")
        if isinstance(audio_data, list) and len(audio_data) > 0:
            audio_data = audio_data[0]
        return {
            "audio": audio_data,  # {"array": np.ndarray, "sampling_rate": int}
            "language": sample.get("custom_language_label"),
        }


class VoiceDatasetWrapper(Dataset):
    LABEL_MAP = {"italian": 0, "english": 1, "turkish": 2, "french": 3}

    def __init__(self, hf_dataset, sample_rate: int, segment_length: int):
        self._dataset = hf_dataset
        self._sample_rate = sample_rate
        self._segment_length = segment_length

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        audio_data = sample["audio"]
        lang_str = sample["language"]

        label_id = torch.tensor(self.LABEL_MAP[lang_str], dtype=torch.long)

        x = torch.from_numpy(audio_data["array"]).float()
        orig_sr = audio_data["sampling_rate"]

        # Mono conversion
        if x.dim() > 1:
            x = torch.mean(x, dim=-1)

        x = x.unsqueeze(0)  # [1, Time]

        if orig_sr != self._sample_rate:
            x = torchaudio.functional.resample(x, orig_sr, self._sample_rate)

        x = x.squeeze(0)  # [Time]

        # Amplitude normalisation
        max_val = torch.max(torch.abs(x))
        if max_val > 0:
            x = x * (0.95 / max_val)

        # Pad if shorter than segment
        if x.shape[0] < self._segment_length:
            x = F.pad(x, [0, self._segment_length - x.shape[0]])

        # Random crop
        pos = random.randint(0, x.shape[0] - self._segment_length)
        x = x[pos: pos + self._segment_length]

        return x, label_id


class ExperimentPhi(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_args = args.dataset_args
        self.num_classes = 4
        self.automatic_optimization = False

        self.model = SoundPhi(
            latent_space_dim=args.model.latent_space_dim,
            n_q=16,
            codebook_size=args.model.codebook_size,
        )
        
        # 1. FREEZE THE ENCODER BACKBONE
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Add your higher-capacity MLP classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(args.model.latent_space_dim), # Normalize incoming encoder ranges first
            nn.Linear(args.model.latent_space_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),                           # Prevents a small head from jumping to early overfit states
            nn.Linear(256, self.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=self.num_classes)
        
        self.validation_step_outputs = []
        self._train_hf = SibFleursMultilingualDataset(split="train")
        self._val_hf   = SibFleursMultilingualDataset(split="validation")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.args.optimizers.lr, # e.g., 5e-4 or 1e-3
            weight_decay=1e-4  # AdamW adds regularization to smooth linear layers
        )
        
        # Linear warmup then cosine decay over max_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.args.trainer.max_steps, 
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]

    def forward(self, audio_input):
        # Ensure the input tensor has 3 dimensions [B, 1, T] required by PhiEncoder
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(1)
            
        features = self.model(audio_input, mode="encode")  # Layout: [16, 256, 50]
        
        # =============================================================
        # FIX: Pool over the LAST dimension (dim=-1) to collapse Time (50)
        # This keeps your 256 feature channels intact!
        # =============================================================
        features = torch.mean(features, dim=-1)            # Layout becomes: [16, 256]
        
        return self.classifier(features)                   # Layout becomes: [16, 4]

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        self.toggle_optimizer(optimizer)

        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Update the metric state quietly without returning the immediate batch value
        self.train_acc.update(logits, y)

        # Log the progressive accumulated accuracy instead of the single batch jitter
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.untoggle_optimizer(optimizer)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc  = self.val_acc(logits, y)

        self.log("val/loss", loss, prog_bar=False, sync_dist=True)
        self.validation_step_outputs.append({"val_loss": loss, "val_acc": acc})

    def on_validation_epoch_end(self):
        self.log("val/avg_accuracy", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()
        self.validation_step_outputs.clear()

    def train_dataloader(self):
        return self._make_dataloader(self._train_hf, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self._val_hf, shuffle=False)

    def _make_dataloader(self, hf_dataset, shuffle: bool):
        def collate(batch_items):
            waveforms, labels = zip(*batch_items)
            return torch.stack(waveforms).unsqueeze(1), torch.stack(labels)

        ds = VoiceDatasetWrapper(
            hf_dataset,
            sample_rate=self.args.sample_rate,
            segment_length=self.dataset_args.segment_length,
        )
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=20,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=init_worker_fn,
        )


@hydra.main(version_base=None, config_path='config', config_name='base')
def train(args):
    artifact_url = args.wandb.get("artifact_url", None)

    logger = (
        WandbLogger(log_model="all", project='soundphi', name="train_01")
        if args.logger == "wandb"
        else CSVLogger("logs", name="exp_1")
    )

    if artifact_url is not None:
        wandb.init(project=args.wandb.get("project", "soundphi"))
        artifact     = wandb.use_artifact(artifact_url, type='model')
        artifact_dir = artifact.download()
        model = ExperimentPhi.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
    else:
        model = ExperimentPhi(args=args)

        checkpoint_path = "checkpoints/encoder_quant.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            # =========================================================
            # DETECT & FIX PREFIX MISMATCH:
            # Check if keys are missing the "encoder." scope prefix.
            # If so, create an explicitly remapped dictionary copy.
            # =========================================================
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith("encoder."):
                print("-> Structural scope naming mismatch detected. Rewriting state_dict keys...")
                state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}

            # Now perform the structural audit with the remapped prefixes
            missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
            
            print("\n" + "="*50)
            print("CHECKPOINT LOADING REPORT")
            print("="*50)
            print(f"Successfully matched and loaded: {len(state_dict) - len(unexpected_keys)} layers.")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys in checkpoint (skipped): {unexpected_keys}")
            if missing_keys:
                print(f"ℹ️ Missing keys expected by SoundPhi (skipped): {missing_keys}")
            print("="*50 + "\n")
        else:
            print("Warning: checkpoint not found, training from scratch.")

    trainer = Trainer(
        logger=logger,
        devices=args.trainer.devices,
        accelerator=args.trainer.accelerator,
        max_steps=args.trainer.max_steps,
    )
    trainer.fit(model)


if __name__ == "__main__":
    train()