import torch 
from torch import nn
from vector_quantize_pytorch import ResidualVQ
from phi import Encoder as PhiEncoder 
from soundstream.decoder import Decoder as SoundStreamDecoder
from typing import Literal 
# from torchsummary import summary 

# class SoundPhi(nn.Module):

#     def __init__(self, 
#                  latent_space_dim,
#                  n_q,
#                  codebook_size):
        
#         super().__init__()
#         self.encoder = PhiEncoder(C=32,D=latent_space_dim)
#         self.decoder = SoundStreamDecoder(C=40, D=latent_space_dim)

#         self.quantizer = ResidualVQ(
#             num_quantizers=n_q,
#             codebook_size=codebook_size,
#             dim=latent_space_dim,
#             kmeans_init=True,
#             kmeans_iters=100,
#             threshold_ema_dead_code=2,
#             quantize_dropout=0.1
#         )

#     def forward(
#             self,
#             x,
#             mode: Literal['end-to-end', 'encode', 'decode'] = 'end-to-end',
#         ):
#         # x: batch_size x 1 x (T / 1)
#         # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
#         # o: batch_size x 1 x (T / 1)

#         if mode == 'end-to-end':
#             e = self.encoder(x)
#             quantized, indices, _ = self.quantizer(e.permute((0,2,1)))
#             # print(indices[:,:,:1].shape)
#             # print(indices)
#             quantized = self.quantizer.get_output_from_indices(indices)
#             o = self.decoder(quantized.permute((0,2,1)))
#             return o
        
#         if mode == 'encode':
#             e = self.encoder(x)
#             quantized, _, _ = self.quantizer(e.permute((0,2,1)))
#             return quantized
        
#         if mode == 'decode':
#             o = self.decoder(x.permute((0,2,1)))
#             return o
class SoundPhi(nn.Module):
    def __init__(self, latent_space_dim: int, **kwargs):
        # **kwargs absorbs n_q and codebook_size so call sites don't need changing
        super().__init__()
        self.encoder = PhiEncoder(C=32, D=latent_space_dim)

    def forward(self, x, mode=None):
        # x: [B, 1, T]
        # returns: [B, T', D]
        return self.encoder(x)
    
if __name__=="__main__":
    net = SoundPhi(latent_space_dim=256,
                    n_q=16,
                    codebook_size=1024).cuda()
    
    # summary(net, (1,16000))
    # print(net(torch.randn(1,1,16000).cuda()).shape)
    encoder = PhiEncoder(C=32, D=256, strides=(5, 5, 8, 16))
    size_model = 0
    for param in encoder.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"phiencoder size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

    size_model = 0
    for param in net.decoder.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"soundstream decoder size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

    