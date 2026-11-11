"""Dense referring-expression grounder for high-resolution aerial images (shared by training and the tool).

Image: frozen ImageNet ResNet-18 at 448x448. layer3 (28x28x256, average-pooled to 14x14) and layer4
(14x14x512) are stacked into a 14x14x768 map and projected to PROJ_DIM channels with a fixed PCA fitted on
training images, so the cached maps fit in memory on a laptop.
Expression: bag of unigrams/bigrams (`rsvqa_features.tokens`) through an EmbeddingBag.
Fusion: the expression modulates every cell of the map (FiLM) in three 3x3 conv blocks. Each cell then scores
"the object's centre is here" and predicts a box relative to itself. The answer is the box of the best
cell, and that cell's softmax probability is the confidence.

Used by training/train_vrsbench_grounding.py and satquery/tools/hrground.py.
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from satquery.models.hr_features import MEAN, STD
from satquery.models.rsvqa_features import tokens

GRID = 14
INPUT = 448
RAW_DIM = 768
PROJ_DIM = 256
MAX_TOKENS = 48
SHIFT = 1.5      # a cell may place the box centre up to 1.5 cells away (neighbours share the work)

_H_SWAP = {"left": "right", "right": "left", "west": "east", "east": "west",
           "western": "eastern", "eastern": "western", "leftmost": "rightmost", "rightmost": "leftmost"}
_V_SWAP = {"top": "bottom", "bottom": "top", "upper": "lower", "lower": "upper", "north": "south",
           "south": "north", "northern": "southern", "southern": "northern", "above": "below", "below": "above",
           "topmost": "bottommost", "bottommost": "topmost"}


# ---- image side --------------------------------------------------------------------------------------------
def load_dense_encoder() -> tuple[nn.Module, nn.Module]:
    from torchvision import models

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    to_l3 = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3).eval()
    return to_l3, m.layer4.eval()


def encode_dense(enc: tuple[nn.Module, nn.Module], rgbs: list[np.ndarray]) -> np.ndarray:
    """HxWx3 uint8 arrays -> (N, RAW_DIM, GRID, GRID) float32 raw feature maps."""
    from PIL import Image

    x = np.stack([np.asarray(Image.fromarray(a).convert("RGB").resize((INPUT, INPUT), Image.BILINEAR), np.float32)
                  / 255.0 for a in rgbs])
    x = torch.from_numpy((x - MEAN) / STD).permute(0, 3, 1, 2)
    with torch.inference_mode():
        f3 = enc[0](x)
        f4 = enc[1](f3)
        return torch.cat([F.avg_pool2d(f3, 2), f4], 1).numpy()


def project(raw: np.ndarray, pca_mean: np.ndarray, pca_comp: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(N, RAW_DIM, G, G) -> (N, PROJ_DIM, G, G): fixed PCA, then unit variance per component."""
    n, c, g, _ = raw.shape
    flat = raw.transpose(0, 2, 3, 1).reshape(-1, c) - pca_mean
    out = (flat @ pca_comp) / scale
    return out.reshape(n, g, g, -1).transpose(0, 3, 1, 2).astype(np.float32)


# ---- expression side ---------------------------------------------------------------------------------------
def flip_words(q: str, h: bool, v: bool) -> str:
    """Swap direction words so the expression still matches a horizontally / vertically flipped image."""
    table = {**(_H_SWAP if h else {}), **(_V_SWAP if v else {})}
    if not table:
        return q
    return re.sub(r"[a-z]+", lambda m: table.get(m.group(0), m.group(0)), q.lower())


def token_ids(q: str, vocab: dict[str, int]) -> np.ndarray:
    """Unique known token ids, padded with -1 to MAX_TOKENS (same information as the binary bag of words)."""
    ids = sorted({vocab[t] for t in tokens(q) if t in vocab})[:MAX_TOKENS]
    return np.array(ids + [-1] * (MAX_TOKENS - len(ids)), np.int64)


# ---- model -------------------------------------------------------------------------------------------------
class _FiLMBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Conv2d(d, d, 3, padding=1)
        self.norm = nn.GroupNorm(8, d, affine=False)
        self.film = nn.Linear(d, 2 * d)

    def forward(self, x, q):
        g, b = self.film(q).chunk(2, 1)
        return x + F.relu(self.norm(self.conv(x)) * (1 + g[:, :, None, None]) + b[:, :, None, None])


class DenseGrounder(nn.Module):
    def __init__(self, n_vocab: int, c_in: int = PROJ_DIM, d: int = 192, grid: int = GRID):
        super().__init__()
        self.grid = grid
        self.inp = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(c_in, d, 1), nn.GroupNorm(8, d), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, d, grid, grid) * 0.02)
        self.emb = nn.EmbeddingBag(n_vocab + 1, d, mode="sum", padding_idx=n_vocab)
        self.q = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, d), nn.ReLU())
        self.blocks = nn.ModuleList([_FiLMBlock(d) for _ in range(3)])
        self.heat = nn.Conv2d(d, 1, 1)
        self.reg = nn.Conv2d(d, 4, 1)
        self.n_vocab = n_vocab

    def forward(self, f: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """f (B, C, G, G), ids (B, T) with -1 padding -> cell logits (B, G*G), boxes (B, G*G, 4) in 0-1."""
        ids = torch.where(ids < 0, torch.full_like(ids, self.n_vocab), ids)
        q = self.q(self.emb(ids))
        x = self.inp(f) + self.pos
        for blk in self.blocks:
            x = blk(x, q)
        logits = self.heat(x).flatten(1)
        r = self.reg(x)                                           # (B, 4, G, G)
        g = self.grid
        ys, xs = torch.meshgrid(torch.arange(g, device=f.device), torch.arange(g, device=f.device), indexing="ij")
        cx = (xs + 0.5) / g + torch.tanh(r[:, 0]) * (SHIFT / g)   # centre stays near its cell
        cy = (ys + 0.5) / g + torch.tanh(r[:, 1]) * (SHIFT / g)
        w, h = torch.sigmoid(r[:, 2]), torch.sigmoid(r[:, 3])
        box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)
        return logits, box.flatten(1, 2)

    @torch.inference_mode()
    def predict(self, f: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Best box per sample (B, 4) and its cell probability (B,)."""
        logits, boxes = self(f, ids)
        p = logits.softmax(1)
        k = p.argmax(1)
        return boxes[torch.arange(len(k)), k], p[torch.arange(len(k)), k]
