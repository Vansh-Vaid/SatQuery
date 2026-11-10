"""Shared encoding for the CDVQA-trained change-question specialist (training/train_cdvqa.py, tools/cdvqa.py).

Pair features (3116-d), from two co-registered optical images (before, after):
  * EuroSAT-adapted ResNet-18 features on a 4x4 tile grid of each date: mean- and max-pooled (2 x 1024)
  * per-tile feature differences |f_after - f_before|: mean- and max-pooled (1024)
  * mean EuroSAT class probabilities of each date (2 x 10)
  * pixel change statistics on radiometrically normalised RGB (10)
  * SatQuery land-cover fractions of each date and their difference (12)
Question encoding: bag of unigrams + bigrams (shared with the RSVQA specialist).
"""
from __future__ import annotations

import numpy as np

from satquery.imagery.raster import Raster
from satquery.imagery.render import to_rgb
from satquery.tools.landcover import classify_optical
from satquery.tools.vocab import CLASSES

GRID, TILE = 4, 64
IMG_DIM = 1024 * 3 + 20 + 10 + 12

# CDVQA answer labels -> readable text (and back, for scoring)
CLASS_TEXT = {"NVG_surface": "non-vegetated ground surface", "low_vegetation": "low vegetation",
              "buildings": "buildings", "water": "water", "trees": "trees", "playgrounds": "playgrounds"}


def label_text(label: str) -> str:
    if label in CLASS_TEXT:
        return CLASS_TEXT[label]
    if "_to_" in label:
        a, b = label.split("_to_")
        return f"{a}-{b}%"
    if label.isdigit():
        return f"{label}%"
    return label


def _tiles(rgb: np.ndarray, mean, std):
    import torch
    from PIL import Image

    arr = np.asarray(Image.fromarray(rgb).resize((GRID * TILE, GRID * TILE), Image.BILINEAR)).copy()
    t = [arr[y:y + TILE, x:x + TILE] for y in range(0, GRID * TILE, TILE) for x in range(0, GRID * TILE, TILE)]
    x = torch.from_numpy(np.stack(t)).permute(0, 3, 1, 2).float() / 255.0
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1), arr.astype(np.float32)


def pair_features(before: Raster, after: Raster, backbone, mean, std) -> np.ndarray:
    import torch

    xa, a = _tiles(to_rgb(before), mean, std)
    xb, b = _tiles(to_rgb(after), mean, std)
    with torch.inference_mode():
        fa, fb = backbone["body"](xa), backbone["body"](xb)
        pa, pb = torch.softmax(backbone["head"](fa), 1), torch.softmax(backbone["head"](fb), 1)
    fa, fb = fa.numpy(), fb.numpy()
    d = np.abs(fb - fa)
    # pixel change on per-image standardised RGB (removes global illumination differences)
    za = (a - a.reshape(-1, 3).mean(0)) / (a.reshape(-1, 3).std(0) + 1e-6)
    zb = (b - b.reshape(-1, 3).mean(0)) / (b.reshape(-1, 3).std(0) + 1e-6)
    mag = np.linalg.norm(zb - za, axis=-1)
    stats = [mag.mean(), mag.std(), *np.percentile(mag, [50, 75, 90, 95]),
             *[(mag > t).mean() for t in (1.0, 1.5, 2.0, 3.0)]]
    la, lb = classify_optical(before).fractions(), classify_optical(after).fractions()
    lc = [la[c] for c in CLASSES] + [lb[c] for c in CLASSES] + [lb[c] - la[c] for c in CLASSES]
    return np.concatenate([fa.mean(0), fa.max(0), fb.mean(0), fb.max(0), d.mean(0), d.max(0),
                           pa.numpy().mean(0), pb.numpy().mean(0), stats, lc]).astype(np.float32)
