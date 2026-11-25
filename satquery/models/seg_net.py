"""Small U-Net water segmenter shared by training (training/train_flood.py) and the flood_mapper tool.

One architecture, three input variants, chosen from what the image offers:
  * "s1"  - Sentinel-1 radar, VV and VH in dB                         (sees water through cloud)
  * "s2"  - multispectral: B02 B03 B04 B08 B11 B12 reflectance        (Sentinel-2, or Landsat via band mapping)
  * "rgb" - true colour as the app renders it (render.to_rgb)          (photos, screenshots, RGB GeoTIFFs)
Inputs are built here for both training and inference, so the two cannot drift apart.
"""
from __future__ import annotations

import numpy as np

from satquery.imagery.raster import Modality, Raster
from satquery.imagery.render import to_rgb

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
IN_CH = {"s1": 2, "s2": len(S2_BANDS), "rgb": 3}
# Dynamic World (Brown et al., 2022) land-cover classes; label value k (1..9) -> index k-1. 0 = no data, 10 = cloud.
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops", "scrub", "built", "bare", "snow_ice"]
# Dynamic World class -> SatQuery's four measured classes (landcover.py), for comparison and for the existing tools
DW_TO_SATQUERY = {"water": "water", "flooded_vegetation": "water", "trees": "vegetation", "grass": "vegetation",
                  "crops": "vegetation", "scrub": "vegetation", "built": "built_up", "bare": "bare_soil",
                  "snow_ice": "bare_soil"}


def variant_for(r: Raster) -> str:
    if r.modality is Modality.SAR:
        return "s1"
    if r.has(*S2_BANDS):
        return "s2"
    return "rgb"


def model_input(r: Raster, variant: str) -> np.ndarray:
    """(C, H, W) float32 in [0, 1] for the given variant."""
    if variant == "s1":
        vv = r.band("VV")
        vv = r.data[0] if vv is None else vv
        vh = r.band("VH")
        vh = vv if vh is None else vh
        x = np.stack([vv, vh])
        return np.clip((np.nan_to_num(x, nan=-30.0) + 30.0) / 35.0, 0, 1).astype(np.float32)
    if variant == "s2":
        x = np.stack([r.band(b) for b in S2_BANDS])
        return np.clip(np.nan_to_num(x, nan=0.0), 0, 1).astype(np.float32)
    return np.transpose(to_rgb(r), (2, 0, 1)).astype(np.float32) / 255.0


def build_unet(in_ch: int, base: int = 16, n_out: int = 1):
    """U-Net with n_out output maps: 1 = binary (water), len(DW_CLASSES) = land-cover classes."""
    net = _unet(in_ch, base, n_out)
    return net


def predict_classes(model, x: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Class probabilities (K, H, W) from a multi-class U-Net, for one input (C, H, W)."""
    import torch
    import torch.nn.functional as F

    c, h, w = x.shape
    t = torch.from_numpy(x)[None].to(next(model.parameters()).device)
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        t = F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False)
    hh, ww = t.shape[-2:]
    t = F.pad(t, (0, (-ww) % 8, 0, (-hh) % 8), mode="reflect" if min(hh, ww) > 8 else "constant")
    with torch.inference_mode():
        p = torch.softmax(model(t), 1)[:, :, :hh, :ww]
    if s < 1:
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
    return p[0].cpu().numpy()


def _unet(in_ch: int, base: int, n_out: int):
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.d1, self.d2, self.d3, self.d4 = block(in_ch, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
            self.pool = nn.MaxPool2d(2)
            self.u3, self.c3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), block(c[3], c[2])
            self.u2, self.c2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(c[2], c[1])
            self.u1, self.c1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(c[1], c[0])
            self.out = nn.Conv2d(c[0], n_out, 1)

        def forward(self, x):
            x1 = self.d1(x)
            x2 = self.d2(self.pool(x1))
            x3 = self.d3(self.pool(x2))
            x4 = self.d4(self.pool(x3))
            y = self.c3(torch.cat([self.u3(x4), x3], 1))
            y = self.c2(torch.cat([self.u2(y), x2], 1))
            y = self.c1(torch.cat([self.u1(y), x1], 1))
            y = self.out(y)
            return y[:, 0] if n_out == 1 else y

    return UNet()


def predict_water(model, x: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Water probability (H, W) for one input (C, H, W); padded to a multiple of 8, downsampled if huge."""
    import torch
    import torch.nn.functional as F

    c, h, w = x.shape
    t = torch.from_numpy(x)[None]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        t = F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False)
    hh, ww = t.shape[-2:]
    ph, pw = (-hh) % 8, (-ww) % 8
    t = F.pad(t, (0, pw, 0, ph), mode="reflect" if min(hh, ww) > 8 else "constant")
    with torch.inference_mode():
        p = torch.sigmoid(model(t))[:, :hh, :ww]
    if s < 1:
        p = F.interpolate(p[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    return p[0].numpy()
