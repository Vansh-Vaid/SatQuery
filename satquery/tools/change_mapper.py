"""change_mapper: any before/after question answered from a trained, global land-cover model.

Each date is mapped into Dynamic World's nine classes by a U-Net trained on expert labels from every biome
(training/train_landcover.py); the question is then read as a class change: an event from satquery.tools.events
(flood, forest loss, fire scars, urban growth, drought, new cropland, snow loss, ...), a single class ("did the
trees decrease?"), or, failing both, a description of the main transitions. No event has its own model.
Radar pairs use the radar water model (training/train_flood.py, Sen1Floods11), so they answer water questions.
A model variant is used only if it beat SatQuery's rule-based land cover on held-out labels.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.validate import InputConfig
from satquery.tools.base import Box, Layer, ParamSpec, TaskType, Tool, ToolContext, ToolResult, ToolSpec
from satquery.tools.events import LABELS, match_classes, match_event
from satquery.tools.landcover import clean_mask, regions, spatial_distribution
from satquery.tools.vocab import CLASS_COLORS

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(__file__).resolve().parents[2] / q


def _load(path: str) -> dict:
    key = str(_path(path))
    with _lock:
        if key not in _cache:
            import torch

            from satquery.models.seg_net import build_unet
            ck = torch.load(key, map_location="cpu", weights_only=False)
            models = {}
            for v, e in ck["variants"].items():
                m = build_unet(e["in_ch"], ck.get("base", 16), e.get("n_out", 1))
                m.load_state_dict(e["state"])
                models[v] = (m.eval(), e.get("metrics", {}), e.get("classes"))
            _cache[key] = models
        return _cache[key]


def _usable(path: str, variant: str):
    if not _path(path).exists():
        return None
    entry = _load(path).get(variant)
    return entry if entry and entry[1].get("beats_baseline") else None


class ChangeMapper(Tool):
    spec = ToolSpec(
        name="change_mapper",
        title="Land-cover change mapper (trained, global)",
        description="Maps both dates into nine land-cover classes (water, trees, grass, flooded vegetation, crops, "
                    "scrub, built-up, bare, snow/ice) with a U-Net trained on Dynamic World expert labels from every "
                    "biome, then answers any change question as a class change: floods, forest loss, fire scars, "
                    "urban growth, drought, new cropland, or a single class increasing or decreasing.",
        tasks=(TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION),
        input_configs=(InputConfig.BITEMPORAL,),
        kind="learned (RS-adapted)",
        params=(ParamSpec("min_change_pct", "float", 1.0, "Share of the scene an event must cover to answer yes",
                          0.1, 20.0),),
    )

    def available(self) -> tuple[bool, str]:
        if _path(settings.landcover_checkpoint).exists() or _path(settings.flood_checkpoint).exists():
            return True, "trained land-cover / water models"
        return False, "checkpoint not found (landcover_seg.pt); run training/train_landcover.py"

    def _maps(self, r):
        """(class names, per-class probabilities (K,H,W), metrics) for one image, or None if no usable model."""
        from satquery.models.seg_net import model_input, predict_classes, predict_water, variant_for
        v = variant_for(r)
        if v == "s1":   # radar: water only
            e = _usable(settings.flood_checkpoint, "s1")
            if e is None:
                return None
            p = predict_water(e[0], model_input(r, "s1"))
            return ["water", "other"], np.stack([p, 1 - p]), e[1]
        e = _usable(settings.landcover_checkpoint, v)
        if e is None:
            return None
        return e[2], predict_classes(e[0], model_input(r, v)), e[1]

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        a, b = ctx.rasters
        ma, mb = self._maps(a), self._maps(b)
        if ma is None or mb is None or ma[0] != mb[0]:
            return ToolResult(self.spec.name, "unsupported",
                              notes=["No trained model that beat the rule-based method fits these images; the "
                                     "measured change tools answer instead."])
        classes, pa, metrics = ma[0], ma[1], ma[2]
        pb = mb[1]
        valid = a.valid_mask & b.valid_mask
        ca, cb = pa.argmax(0), pb.argmax(0)
        n = max(int(valid.sum()), 1)
        idx = {c: i for i, c in enumerate(classes)}
        pct = lambda m: 100 * float((m & valid).sum()) / n   # noqa: E731
        trans = sorted(((pct((ca == i) & (cb == j)), classes[i], classes[j])
                        for i in range(len(classes)) for j in range(len(classes)) if i != j), reverse=True)
        trans = [t for t in trans if t[0] >= 0.1]
        event, asked = match_event(ctx.query), [c for c in match_classes(ctx.query) if c in idx]
        thr = float(params.get("min_change_pct", 1.0))
        conf_model = float(metrics.get("test_miou", metrics.get("test_iou", 0.5)))
        margin = float(np.mean(np.maximum(pa.max(0), pb.max(0))[valid])) if valid.any() else 0.5
        conf = 0.5 * conf_model + 0.5 * margin
        notes = []
        if any(r.meta.get("resized_to_ref") for r in ctx.rasters):
            conf *= 0.55
            notes.append("The two images had different sizes and were stretched to match.")
        facts: dict[str, Any] = {"classes": classes, "model_variant": metrics.get("variant"),
                                 "per_class": {c: {"before_pct": round(pct(ca == i), 2), "after_pct": round(pct(cb == i), 2)}
                                               for c, i in idx.items()},
                                 "top_transitions": [{"from": f, "to": t, "pct": round(p, 2)} for p, f, t in trans[:6]]}
        mask = np.zeros_like(valid)
        gain = [idx[c] for c in (event.gain if event else ()) if c in idx]   # radar models know water only
        loss = [idx[c] for c in (event.loss if event else ()) if c in idx]
        if event is not None and (gain or loss) and (not event.gain or gain) and (not event.loss or loss):
            g = np.isin(cb, gain) & ~np.isin(ca, gain) if gain else np.ones_like(valid)
            lo = np.isin(ca, loss) & ~np.isin(cb, loss) if loss else np.ones_like(valid)
            mask = g & lo & valid
            share = pct(mask)
            yes = share >= thr
            where = f", {spatial_distribution(mask)}" if mask.any() else ""
            text = (f"Yes: {event.name} on about {share:.1f}% of the scene{where}." if yes else
                    f"No clear sign of {event.name}: it covers only {share:.1f}% of the scene.")
            facts.update(short_answer="yes" if yes else "no", question_type="event", event=event.key,
                         event_pct=round(share, 2))
        elif asked:
            c = asked[0]
            i = idx[c]
            before, after = pct(ca == i), pct(cb == i)
            gained, lost = pct((cb == i) & (ca != i)), pct((ca == i) & (cb != i))
            word = "increased" if after - before >= thr else "decreased" if before - after >= thr else "stayed about the same"
            mask = ((cb == i) != (ca == i)) & valid
            text = (f"{LABELS[c].capitalize()} {word}: {before:.1f}% to {after:.1f}% of the scene "
                    f"(gained {gained:.1f}%, lost {lost:.1f}%).")
            facts.update(short_answer=word, question_type="class", cls=c)
        else:
            mask = (ca != cb) & valid
            parts = [f"{LABELS.get(f, f)} to {LABELS.get(t, t)} ({p:.1f}%)" for p, f, t in trans[:3]]
            text = (f"About {pct(mask):.1f}% of the scene changed class. Main changes: " + "; ".join(parts) + "."
                    if parts else "No land-cover change was found between the two images.")
            facts.update(short_answer="description", question_type="describe")
        h, w = mask.shape
        regs = regions(clean_mask(mask, 1), min_px=max(6, int(0.0005 * h * w)), label="change")
        boxes = [Box(r["x0"], r["y0"], r["x1"], r["y1"], label=f"change {k + 1}", source=self.spec.name)
                 for k, r in enumerate(regs[:10])]
        model = metrics.get("description", "land-cover model trained on Dynamic World expert labels")
        return ToolResult(self.spec.name, answer=f"{text} ({model})", boxes=boxes, notes=notes,
                          layers=[Layer("changed area", "mask", mask, color=CLASS_COLORS["change"], legend="change")],
                          confidence=float(np.clip(conf, 0.05, 0.97)), facts=facts)
