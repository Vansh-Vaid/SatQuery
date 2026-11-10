"""Image features for high-resolution (sub-metre) aerial scenes, used by the VRSBench-trained VQA specialist.

ImageNet ResNet-18 (frozen) on the whole 224x224 image. The last feature map (7x7x512) is pooled to a global
vector and a 3x3 grid, so position questions ("top-left", "bottom-right") keep their spatial signal.
Output: 512 + 9*512 = 5120 floats. Shared by training/train_vrsbench_vqa.py and satquery/tools/hrvqa.py.
"""
from __future__ import annotations

import numpy as np

IMG_DIM = 512 * 10
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_encoder():
    import torch
    from torchvision import models

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return torch.nn.Sequential(*list(m.children())[:-2]).eval()   # up to layer4 -> (B, 512, 7, 7)


def encode_batch(body, rgbs: list[np.ndarray]) -> np.ndarray:
    """rgbs: HxWx3 uint8 arrays -> (N, 5120) float32."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    x = np.stack([np.asarray(Image.fromarray(a).convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255.0
                  for a in rgbs])
    x = torch.from_numpy((x - MEAN) / STD).permute(0, 3, 1, 2)
    with torch.inference_mode():
        f = body(x)
        g = f.mean((2, 3))
        grid = F.adaptive_avg_pool2d(f, 3).flatten(1)
    return torch.cat([g, grid], 1).numpy().astype(np.float32)
