"""rs_scene_classifier: EuroSAT-adapted ResNet-18 (a remote-sensing-adapted visual component).

Fine-tuned from ImageNet on EuroSAT Sentinel-2 patches (training/train_scene_classifier.py).
The scene is tiled into ~640 m tiles (EuroSAT's patch footprint at 10 m), each classified into one of
10 land-use classes. Outputs a scene-level land-use distribution, a coarse land-use map, and answers to
land-use questions ("what kind of area is this?", "is there a highway?").
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.raster import Modality
from satquery.imagery.render import to_rgb
from satquery.imagery.validate import InputConfig
from satquery.tools.base import Layer, ParamSpec, Tool, ToolContext, ToolResult, ToolSpec, TaskType

LABELS = {"AnnualCrop": "annual cropland", "Forest": "forest", "HerbaceousVegetation": "herbaceous vegetation",
          "Highway": "highway / major road", "Industrial": "industrial area", "Pasture": "pasture",
          "PermanentCrop": "permanent crops (orchards, vineyards)", "Residential": "residential area",
          "River": "river", "SeaLake": "sea or lake"}
COLORS = {"AnnualCrop": (233, 196, 106), "Forest": (79, 91, 34), "HerbaceousVegetation": (169, 179, 107),
          "Highway": (120, 120, 120), "Industrial": (177, 79, 18), "Pasture": (138, 154, 60),
          "PermanentCrop": (168, 107, 18), "Residential": (210, 101, 31), "River": (79, 134, 166),
          "SeaLake": (46, 106, 142)}
# query phrase -> class. "specific" classes are ones the land-cover reasoner cannot answer on its own,
# so a question about them makes this tool the primary source.
PHRASES = [
    (r"permanent crops?|orchards?|vineyards?|olive groves?", "PermanentCrop", True),
    (r"annual crops?|cropland|crop fields?|farmland|agricultur\w*|arable", "AnnualCrop", True),
    (r"herbaceous|grassland|meadows?|shrubland", "HerbaceousVegetation", True),
    (r"highways?|motorways?|major roads?|freeways?|roads?", "Highway", True),
    (r"industrial|factor(y|ies)|warehouses?", "Industrial", True),
    (r"pastures?|grazing", "Pasture", True),
    (r"residential|houses?|housing|neighbou?rhoods?", "Residential", True),
    (r"forests?|woodlands?|trees", "Forest", False),
    (r"rivers?|streams?", "River", False),
    (r"seas?|lakes?|oceans?|reservoirs?", "SeaLake", False),
]
_LAND_USE = re.compile(r"\b(land[- ]?use|what (kind|type|sort) of (scene|area|place|land|landscape|terrain)|"
                       r"what is (this|shown|in this)|classif\w*|what does this (image|scene) show|scene type)\b")
_PRESENCE = re.compile(r"^\s*(is|are|does|do|can)\b|\b(any|present|visible|contain\w*)\b")
_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


def match_classes(q: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    for pat, cls, specific in PHRASES:
        if re.search(rf"\b({pat})\b", q, re.I) and cls not in [c for c, _ in out]:
            out.append((cls, specific))
    return out


def checkpoint_path() -> Path:
    p = Path(settings.scene_checkpoint)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def load_model():
    path = checkpoint_path()
    key = str(path)
    with _lock:
        if key not in _model_cache:
            import torch
            from torchvision import models
            state = torch.load(path, map_location="cpu", weights_only=False)
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, len(state["classes"]))
            m.load_state_dict(state["model"])
            m.eval()
            mp = Path(str(path) + ".metrics.json")
            metrics = json.loads(mp.read_text()) if mp.exists() else {}
            _model_cache[key] = (m, state, metrics)
        return _model_cache[key]


class SceneClassifier(Tool):
    spec = ToolSpec(
        name="rs_scene_classifier",
        title="EuroSAT-adapted scene classifier (ResNet-18)",
        description="Remote-sensing-adapted CNN fine-tuned on EuroSAT Sentinel-2 imagery. Classifies ~640 m tiles "
                    "into 10 land-use classes (annual crop, forest, herbaceous vegetation, highway, industrial, "
                    "pasture, permanent crop, residential, river, sea/lake), giving a scene-level land-use "
                    "distribution, a coarse land-use map, and answers to land-use questions. Optical input only.",
        tasks=(TaskType.VQA, TaskType.CAPTION, TaskType.CROSS_MODAL),
        input_configs=(InputConfig.SINGLE, InputConfig.CROSS_MODAL),
        kind="learned (RS-adapted)",
        params=(ParamSpec("mode", "enum", "auto", "qa | describe | auto", choices=("auto", "qa", "describe")),
                ParamSpec("presence_threshold", "float", 0.25, "Share of tiles (or probability for a single "
                                                                "tile) above which a class is reported present", 0.05, 0.9)),
    )

    def available(self) -> tuple[bool, str]:
        p = checkpoint_path()
        if not p.exists():
            return False, f"checkpoint not found ({p.name}); run training/train_scene_classifier.py"
        return True, f"ResNet-18 fine-tuned on EuroSAT ({p.name})"

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        import torch
        from PIL import Image

        r = next((x for x in ctx.rasters if x.modality.is_optical), None)
        if r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["Trained on optical imagery; no optical image supplied."])
        model, state, metrics = load_model()
        classes: list[str] = state["classes"]
        size = int(state.get("size", 64))
        rgb = to_rgb(r)
        h, w = rgb.shape[:2]
        # tile at EuroSAT's ground footprint (64 px x 10 m = 640 m) when the pixel size is known
        tile = int(round(640.0 / r.gsd_m)) if r.gsd_m else 64
        tile = int(np.clip(tile, 32, max(32, min(h, w))))
        if min(h, w) < tile * 1.5:
            boxes = [(0, 0, w, h)]
        else:
            ys = list(range(0, h - tile + 1, tile)) or [0]
            xs = list(range(0, w - tile + 1, tile)) or [0]
            boxes = [(x, y, x + tile, y + tile) for y in ys for x in xs]
        mean = torch.tensor(state["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(state["std"]).view(1, 3, 1, 1)
        batch, keep = [], []
        valid = r.valid_mask
        for (x0, y0, x1, y1) in boxes:
            if valid is not None and valid[y0:y1, x0:x1].mean() < 0.5:
                continue
            im = Image.fromarray(rgb[y0:y1, x0:x1]).resize((size, size), Image.BILINEAR)
            batch.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0)
            keep.append((x0, y0, x1, y1))
        if not batch:
            return ToolResult(self.spec.name, "unsupported", notes=["No tile has enough valid pixels."])
        with torch.inference_mode():
            probs = torch.softmax(model((torch.stack(batch) - mean) / std), dim=1).numpy()
        scene = probs.mean(axis=0)
        tile_cls = probs.argmax(axis=1)
        share = np.bincount(tile_cls, minlength=len(classes)) / len(keep)
        order = np.argsort(-scene)
        top = [(classes[i], float(scene[i])) for i in order[:3] if scene[i] >= 0.05] or [(classes[order[0]], float(scene[order[0]]))]
        conf = float(np.mean(probs.max(axis=1)))
        cmap = np.full((h, w), -1, np.int16)
        for (x0, y0, x1, y1), c in zip(keep, tile_cls):
            cmap[y0:y1, x0:x1] = c
        layers = [Layer(f"land use: {LABELS[classes[i]]}", "mask", cmap == i, color=COLORS[classes[i]],
                        legend=f"EuroSAT class {classes[i]}") for i in order[:3] if (cmap == i).any()]
        facts: dict[str, Any] = {
            "model": "ResNet-18 fine-tuned on EuroSAT (Sentinel-2, 10 classes)",
            "eurosat_test_accuracy": metrics.get("test_accuracy"),
            "tiles": len(keep), "tile_px": tile,
            "scene_distribution_pct": {classes[i]: round(100 * float(scene[i]), 1) for i in order[:5]},
            "tile_share_pct": {classes[i]: round(100 * float(share[i]), 1) for i in order[:5] if share[i] > 0},
        }
        notes = []
        if r.gsd_m is None:
            notes.append("Pixel size unknown; the classifier was trained on 10 m Sentinel-2 patches, so results on "
                         "very different resolutions are less reliable.")
        desc = (f"Land use (EuroSAT-adapted classifier, {len(keep)} tile{'s' if len(keep) > 1 else ''}): mainly "
                f"{LABELS[top[0][0]]} ({100 * top[0][1]:.0f}%)"
                + (", with " + ", ".join(f"{LABELS[c]} ({100 * p:.0f}%)" for c, p in top[1:]) if len(top) > 1 else "")
                + ".")
        q = ctx.query.lower()
        mode = params.get("mode", "auto")
        matched = match_classes(q)
        thr = float(params.get("presence_threshold", 0.25))
        if mode != "describe" and matched and _PRESENCE.search(q):
            cls, specific = matched[0]
            i = classes.index(cls)
            val = float(share[i]) if len(keep) > 1 else float(scene[i])
            yes = val >= thr or (len(keep) > 1 and share[i] > 0 and scene[i] >= thr / 2)
            ans = "yes" if yes else "no"
            where = f"{100 * share[i]:.0f}% of tiles" if len(keep) > 1 else f"probability {100 * scene[i]:.0f}%"
            facts.update(question_type="presence", short_answer=ans, scene_specific=specific)
            margin = float(np.clip(abs(val - thr) / max(thr, 1e-6), 0, 1))
            return ToolResult(self.spec.name, answer=f"{ans.capitalize()}: {LABELS[cls]} ({where}). {desc}", facts=facts,
                              layers=layers, confidence=float((0.55 + 0.4 * margin) * conf ** 0.5), notes=notes)
        if mode != "describe" and (_LAND_USE.search(q) or re.search(r"\b(main|dominant|primary) land[- ]?use\b", q)):
            facts.update(question_type="land_use", short_answer=LABELS[top[0][0]], scene_specific=True)
            return ToolResult(self.spec.name, answer=desc, facts=facts, layers=layers,
                              confidence=float(top[0][1] ** 0.5 * conf ** 0.5), notes=notes)
        if mode == "qa":
            facts.update(question_type="unsupported")
            return ToolResult(self.spec.name, "unsupported", facts=facts, layers=layers,
                              notes=["Question is not about land use; scene distribution kept as evidence."])
        facts.update(question_type="describe")
        return ToolResult(self.spec.name, answer=desc, facts=facts, layers=layers, confidence=conf, notes=notes)
