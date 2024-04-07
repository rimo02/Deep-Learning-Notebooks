
from vit import ViT
from decoder import MaskDecoder
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import torch

class Segmenter(nn.Module):
    def __init__(self,
                 in_channels,
                 scale,
                 patch_size,
                 image_size,
                 enc_depth,
                 dec_depth,
                 enc_embdd,
                 dec_embdd,
                 n_cls):
        super().__init__()
        self.encoder = ViT(in_channels,
                           patch_size,
                           enc_embdd,
                           image_size,
                           enc_depth)
        self.decoder = MaskDecoder(scale,
                                   dec_depth,
                                   patch_size,
                                   n_cls,
                                   dec_embdd)

    def forward(self, img):
        H, W = img.size(2), img.size(3)
        x = self.encoder(img)
        print(x.shape)
        x = x[:, 1:]  # remove Cls token
        masks = self.decoder(x, (H, W))
        print(masks.shape)
        out = F.interpolate(masks, size=(H, W), mode="bilinear")
        return out

model=Segmenter(3,0.05,16,256,12,6,768,768,1)
print(model(torch.randn([16,3,256,256])).shape)
