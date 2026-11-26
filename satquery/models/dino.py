"""DINOv2 ViT-S/14 with registers (Oquab et al., 2023; Apache-2.0), frozen, as the image encoder of the dense
high-resolution models. Parameter names follow the official release, so its weights load unchanged:
https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
A 392x392 image gives two 14x14x384 maps (block 9 and block 12, average-pooled from 28x28), stacked to 768 channels:
the same shape as the ResNet-18 maps it replaces.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM, DEPTH, HEADS, REGISTERS, PATCH, BASE = 384, 12, 6, 4, 14, 37
INPUT = 392
TAPS = (8, 11)      # blocks whose output becomes a feature map


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(DIM, 1e-6), nn.LayerNorm(DIM, 1e-6)
        self.attn = nn.ModuleDict({"qkv": nn.Linear(DIM, 3 * DIM), "proj": nn.Linear(DIM, DIM)})
        self.mlp = nn.ModuleDict({"fc1": nn.Linear(DIM, 4 * DIM), "fc2": nn.Linear(4 * DIM, DIM)})
        self.ls1 = nn.ParameterDict({"gamma": nn.Parameter(torch.ones(DIM))})
        self.ls2 = nn.ParameterDict({"gamma": nn.Parameter(torch.ones(DIM))})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.attn["qkv"](self.norm1(x)).reshape(b, n, 3, HEADS, d // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.ls1["gamma"] * self.attn["proj"](a)
        return x + self.ls2["gamma"] * self.mlp["fc2"](F.gelu(self.mlp["fc1"](self.norm2(x))))


class DinoV2S(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.register_tokens = nn.Parameter(torch.zeros(1, REGISTERS, DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, DIM))       # in the release; unused here
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + BASE * BASE, DIM))
        self.patch_embed = nn.ModuleDict({"proj": nn.Conv2d(3, DIM, PATCH, PATCH)})
        self.blocks = nn.ModuleList([_Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, INPUT, INPUT), ImageNet-normalised -> (B, 768, 14, 14)."""
        x = self.patch_embed["proj"](x)
        b, d, h, w = x.shape
        pos = F.interpolate(self.pos_embed[:, 1:].reshape(1, BASE, BASE, d).permute(0, 3, 1, 2), size=(h, w),
                            mode="bicubic", antialias=True)
        x = torch.cat([(self.cls_token + self.pos_embed[:, :1]).expand(b, -1, -1),
                       self.register_tokens.expand(b, -1, -1), (x + pos).flatten(2).transpose(1, 2)], 1)
        maps = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in TAPS:
                m = self.norm(x)[:, 1 + REGISTERS:].transpose(1, 2).reshape(b, d, h, w)
                maps.append(F.avg_pool2d(m, 2))
        return torch.cat(maps, 1)


def load_dino(path: str | Path) -> DinoV2S:
    """The frozen encoder; half-precision weight files (ours) load back as float32."""
    m = DinoV2S()
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()})
    return m.eval().requires_grad_(False)
