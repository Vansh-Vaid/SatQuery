"""Lightweight Siamese U-Net for bi-temporal change detection (optional learned backend).

Architecture: shared-weight encoder applied to both dates, absolute feature
differences fused at every scale, U-Net decoder -> 1-channel change logit.
~1.9 M parameters, trainable on a free Kaggle/Colab T4 on LEVIR-CD in < 2 h
(see training/train_change.py). Input: two RGB images in [0,1], (B,3,H,W).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SiameseUNet(nn.Module):
    def __init__(self, in_ch: int = 3, widths: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.enc.append(_block(c, w))
            c = w
        self.dec = nn.ModuleList()
        rev = list(widths[::-1])
        for i in range(len(rev) - 1):
            self.dec.append(_block(rev[i] + rev[i + 1], rev[i + 1]))
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, blk in enumerate(self.enc):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = blk(x)
            feats.append(x)
        return feats

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        fa, fb = self._encode(a), self._encode(b)
        diffs = [torch.abs(x - y) for x, y in zip(fa, fb)]
        x = diffs[-1]
        for i, blk in enumerate(self.dec):
            skip = diffs[-(i + 2)]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skip], dim=1))
        return self.head(x)


def load_change_net(path: str, device: str = "cpu") -> SiameseUNet:
    model = SiameseUNet()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
    return model.eval().to(device)
