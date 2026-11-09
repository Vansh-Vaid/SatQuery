"""Shared encoding for the RSVQA-trained VQA specialist (training/train_rsvqa_lr.py and tools/rsvqa.py).

Image features (1038-d):
  * EuroSAT-adapted ResNet-18 penultimate features over a 4x4 grid of tiles: mean- and max-pooled (1024)
  * the classifier's mean EuroSAT class probabilities (10)
  * SatQuery's land-cover fractions: water, vegetation, built-up, bare soil (4)
Question: binary bag of unigrams + bigrams over the training vocabulary.
Answers: yes/no, rural/urban, and RSVQA-LR count bins.
"""
from __future__ import annotations

import re

import numpy as np

from satquery.imagery.raster import Raster
from satquery.imagery.render import to_rgb
from satquery.tools.landcover import classify_optical
from satquery.tools.vocab import CLASSES

ANSWERS = ["yes", "no", "rural", "urban", "0", "1-10", "11-100", "101-1000", ">1000"]
COUNT_BINS = ANSWERS[4:]
BIN_TEXT = {"0": "0", "1-10": "between 1 and 10", "11-100": "between 11 and 100",
            "101-1000": "between 101 and 1000", ">1000": "more than 1000"}
IMG_DIM = 1024 + 10 + 4
GRID = 4
TILE = 64


def quantize(n: float) -> str:
    return "0" if n <= 0 else "1-10" if n <= 10 else "11-100" if n <= 100 else "101-1000" if n <= 1000 else ">1000"


def question_type(q: str) -> str:
    q = q.lower()
    if re.search(r"\brural\b.*\burban\b|\burban\b.*\brural\b", q):
        return "rural_urban"
    if re.search(r"\b(how many|number of|amount of)\b", q) and not re.search(r"\b(equal|more|less|than)\b", q):
        return "count"
    return "yesno"


def answer_mask(qtype: str) -> np.ndarray:
    m = np.zeros(len(ANSWERS), bool)
    if qtype == "rural_urban":
        m[2:4] = True
    elif qtype == "count":
        m[4:] = True
    else:
        m[:2] = True
    return m


def tokens(q: str) -> list[str]:
    w = re.findall(r"[a-z]+", q.lower())
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


def encode_question(q: str, vocab: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab), np.float32)
    for t in tokens(q):
        i = vocab.get(t)
        if i is not None:
            v[i] = 1.0
    return v


def image_features(r: Raster, backbone, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """1038-d image vector: pooled tile features, mean class probabilities, land-cover fractions."""
    import torch
    from PIL import Image

    rgb = to_rgb(r)
    im = Image.fromarray(rgb).resize((GRID * TILE, GRID * TILE), Image.BILINEAR)
    arr = np.asarray(im).copy()
    tiles = [arr[y:y + TILE, x:x + TILE] for y in range(0, GRID * TILE, TILE) for x in range(0, GRID * TILE, TILE)]
    x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    with torch.inference_mode():
        feats = backbone["body"](x)                       # (16, 512)
        probs = torch.softmax(backbone["head"](feats), 1)  # (16, 10)
    f = feats.numpy()
    lc = classify_optical(r).fractions()
    return np.concatenate([f.mean(0), f.max(0), probs.numpy().mean(0), [lc[c] for c in CLASSES]]).astype(np.float32)


def load_backbone(scene_ckpt: str):
    """EuroSAT ResNet-18 split into feature body and classification head."""
    import torch
    from torchvision import models

    state = torch.load(scene_ckpt, map_location="cpu", weights_only=False)
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(state["classes"]))
    m.load_state_dict(state["model"])
    m.eval()
    head = m.fc
    m.fc = torch.nn.Identity()
    return {"body": m, "head": head}, np.array(state["mean"], np.float32), np.array(state["std"], np.float32)


def build_model(vocab_size: int, hidden: int = 512, dropout: float = 0.3):
    """Image and question projected to a shared space, fused by elementwise product (Lobry et al. 2020)."""
    import torch.nn as nn

    class RSVQANet(nn.Module):
        def __init__(self):
            super().__init__()
            self.img = nn.Sequential(nn.Linear(IMG_DIM, hidden), nn.Tanh(), nn.Dropout(dropout))
            self.q = nn.Sequential(nn.Linear(vocab_size, hidden), nn.Tanh(), nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, len(ANSWERS)))

        def forward(self, img, q):
            return self.head(self.img(img) * self.q(q))

    return RSVQANet()
