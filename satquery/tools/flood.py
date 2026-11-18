"""flood_mapper: newly flooded land from a before/after pair, with water segmenters trained on Sen1Floods11.

Water is mapped on each date with the seg_net variant that fits the image (radar, multispectral or true colour);
flooded = water in the second image that was not water in the first. Rivers and lakes present on both dates are
reported as permanent water, not as flooding. A variant is used only if it beat SatQuery's classic water method
on Sen1Floods11's held-out test split (training/train_flood.py writes that comparison into the checkpoint).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.validate import InputConfig
from satquery.tools.base import Box, Layer, ParamSpec, TaskType, Tool, ToolContext, ToolResult, ToolSpec
from satquery.tools.landcover import clean_mask, regions, spatial_distribution
from satquery.tools.vocab import CLASS_COLORS

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(__file__).resolve().parents[2] / q


def _load():
    key = str(_path(settings.flood_checkpoint))
    with _lock:
        if key not in _cache:
            import torch

            from satquery.models.seg_net import build_unet
            ck = torch.load(key, map_location="cpu", weights_only=False)
            models = {}
            for v, entry in ck["variants"].items():
                m = build_unet(entry["in_ch"], ck.get("base", 16))
                m.load_state_dict(entry["state"])
                models[v] = (m.eval(), entry.get("metrics", {}))
            _cache[key] = models
        return _cache[key]


class FloodMapper(Tool):
    spec = ToolSpec(
        name="flood_mapper",
        title="Flood mapper (trained on Sen1Floods11)",
        description="Maps open water on both dates with a U-Net trained on hand-labelled flood events from 11 "
                    "countries (radar, multispectral or true-colour variant, chosen from the image) and reports "
                    "newly flooded land, permanent water and receded water, with a flood map.",
        tasks=(TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION),
        input_configs=(InputConfig.BITEMPORAL,),
        kind="learned (RS-adapted)",
        params=(ParamSpec("threshold", "float", 0.5, "Water probability above which a pixel counts as water",
                          0.2, 0.8),
                ParamSpec("min_flood_pct", "float", 1.0, "Newly flooded share of the scene needed to answer yes",
                          0.1, 20.0)),
    )

    def available(self) -> tuple[bool, str]:
        if not _path(settings.flood_checkpoint).exists():
            return False, "checkpoint not found (flood_seg.pt); run training/train_flood.py"
        return True, "trained on Sen1Floods11"

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        from satquery.models.seg_net import model_input, predict_water, variant_for

        models = _load()
        a, b = ctx.rasters
        variants = [variant_for(r) for r in (a, b)]
        for v in set(variants):
            if v not in models:
                return ToolResult(self.spec.name, "unsupported", notes=[f"No trained {v} water model in the checkpoint."])
            if not models[v][1].get("beats_baseline", False):
                return ToolResult(self.spec.name, "unsupported",
                                  notes=[f"The {v} water model did not beat the classic method on its test set; "
                                         "the measured change tools answer instead."])
        thr = float(params.get("threshold", 0.5))
        pa = predict_water(models[variants[0]][0], model_input(a, variants[0]))
        pb = predict_water(models[variants[1]][0], model_input(b, variants[1]))
        valid = a.valid_mask & b.valid_mask
        wa, wb = (pa > thr) & valid, (pb > thr) & valid
        new, perm, gone = wb & ~wa, wa & wb, wa & ~wb
        n = max(int(valid.sum()), 1)
        flood, perm_pct, gone_pct = (100 * float(m.sum()) / n for m in (new, perm, gone))
        h, w = new.shape
        regs = regions(clean_mask(new, 1), min_px=max(6, int(0.0005 * h * w)), label="flooded")
        where = spatial_distribution(new) if new.any() else ""
        metrics = [models[v][1] for v in variants]
        test_iou = float(np.mean([m.get("test_iou", 0.0) for m in metrics]))
        base_iou = float(np.mean([m.get("baseline_iou", 0.0) for m in metrics]))
        water = wa | wb
        margin = float(np.mean(np.abs(2 * np.stack([pa, pb])[:, water] - 1))) if water.any() else 0.5
        conf = 0.5 * test_iou + 0.5 * margin
        notes = []
        if any(r.meta.get("resized_to_ref") for r in ctx.rasters):
            conf *= 0.55
            notes.append("The two images had different sizes and were stretched to match.")
        yes = flood >= float(params.get("min_flood_pct", 1.0))
        model = (f"water model trained on Sen1Floods11 ({'/'.join(sorted(set(variants)))}); test IoU {test_iou:.2f} "
                 f"vs {base_iou:.2f} for the classic method")
        if yes:
            answer = (f"Yes: about {flood:.1f}% of the scene is newly under water in the second image, {where}. "
                      f"Water present in both images (rivers, lakes) covers {perm_pct:.1f}% and is not counted as "
                      f"flooding." + (f" Water receded from {gone_pct:.1f}%." if gone_pct >= 0.5 else "") +
                      f" ({model})")
        else:
            answer = (f"No clear flooding: only {flood:.1f}% of the scene is newly under water in the second image; "
                      f"permanent water covers {perm_pct:.1f}%. ({model})")
        boxes = [Box(r["x0"], r["y0"], r["x1"], r["y1"], label=f"flooded {i + 1}", source=self.spec.name,
                     score=float(np.mean(pb[r["y0"]:r["y1"], r["x0"]:r["x1"]])))
                 for i, r in enumerate(regs[:10])]
        layers = [Layer("flooded area", "mask", new, color=CLASS_COLORS["water"], legend="newly under water"),
                  Layer("water probability (second image)", "heatmap", np.where(valid, pb, np.nan), cmap="Blues",
                        vmin=0, vmax=1, legend="water probability")]
        return ToolResult(self.spec.name, answer=answer, boxes=boxes, layers=layers,
                          confidence=float(np.clip(conf, 0.05, 0.97)), notes=notes,
                          facts={"short_answer": "yes" if yes else "no", "question_type": "flood",
                                 "flooded_pct": round(flood, 2), "permanent_water_pct": round(perm_pct, 2),
                                 "receded_pct": round(gone_pct, 2), "variants": variants, "n_regions": len(regs),
                                 "model_test_iou": round(test_iou, 4), "classic_method_iou": round(base_iou, 4)})
