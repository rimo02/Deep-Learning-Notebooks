import torch
from einops import rearrange
import torch.nn as nn
from vit import TransformerEncoder


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, embedd_dim):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.embedd_dim = embedd_dim
        self.head = nn.Linear(embedd_dim, n_cls)

    def forward(self, x, img_size: int = 256):
        H = W = img_size
        num_patch = H//self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=num_patch)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, scale, depth, patch_size, n_cls, dec_embdd):
        super().__init__()
        self.n_cls = n_cls
        self.dec_embdd = dec_embdd
        self.scale = scale
        self.patch_size = patch_size

        self.cls_emb = nn.Parameter(
            torch.randn([1, n_cls, dec_embdd*2]))
        self.dec_proj = nn.Linear(dec_embdd, dec_embdd*2)

        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(dec_embdd*2, dec_embdd*2))
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(dec_embdd*2, dec_embdd*2))

        self.decoder_norm = nn.LayerNorm(dec_embdd*2)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.blocks = TransformerEncoder(depth=depth, emb_size=dec_embdd*2)

    def forward(self, x, img_size):
        H, W = img_size
        GS = H//self.patch_size
        x = self.dec_proj(x)
        # Adding a cls token for each segmenting class
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        out = (self.blocks(x))
        x = self.decoder_norm(out)

        patches, cls_seg_feat = x[:, self.n_cls], x[:, :self.n_cls:]
        patches = patches @ self.proj_patch  # 1 x 61 x 768
        cls_seg_feat = cls_seg_feat @ self.proj_classes  # 1 x 4 x 768
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        return masks
