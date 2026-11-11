"""hr_grounder: object grounding for high-resolution aerial images, trained on VRSBench.

Finds one box for a referring expression ("the large yellow vehicle closest to the green area").
Two checkpoint formats load:
- "dense" (current, satquery/models/hr_ground_net.py): a 14x14 ResNet-18 feature map modulated by the
  expression; the best-scoring cell gives the box. Confidence is read from the checkpoint's test-set table
  of Acc@0.5 by cell probability, so it is a measured hit rate, not a guess.
- legacy: one box regressed from a 3x3 grid (14.5% Acc@0.5); confidence capped at CAP.
Abstains on images coarser than 5 m per pixel.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.render import to_rgb
from satquery.imagery.validate import InputConfig
from satquery.models.hr_features import encode_batch, load_encoder
from satquery.models.rsvqa_features import encode_question, tokens
from satquery.tools.base import Box, ParamSpec, TaskType, Tool, ToolContext, ToolResult, ToolSpec

_lock = threading.Lock()
_cache: dict[str, Any] = {}
MAX_GSD_M = 5.0
OBJECTS = re.compile(r"\b(vehicles?|cars?|trucks?|bus(es)?|ships?|boats?|vessels?|planes?|airplanes?|aircraft|"
                     r"bridges?|harbou?rs?|tanks?|storage|courts?|fields?|diamonds?|roundabouts?|windmills?|"
                     r"stations?|pools?|chimneys?|dams?|stadiums?|terminals?|buildings?|houses?)\b", re.I)
CAP = 0.35   # 14.5% Acc@0.5 on the test set: never report more than low confidence


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(__file__).resolve().parents[2] / q


def _load():
    key = str(_path(settings.vrsbench_ground_checkpoint))
    with _lock:
        if key not in _cache:
            import torch
            import torch.nn as nn
            state = torch.load(key, map_location="cpu", weights_only=False)
            if state.get("arch") == "dense":
                from satquery.models.hr_ground_net import DenseGrounder, load_dense_encoder
                model = DenseGrounder(len(state["vocab"]))
                model.load_state_dict(state["model"])
                model.eval()
                _cache[key] = (model, state, load_dense_encoder(), state.get("metrics", {}))
                return _cache[key]
            nv, d = len(state["vocab"]), state["img_dim"]

            class Grounder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.img = nn.Sequential(nn.Linear(d, 768), nn.Tanh(), nn.Dropout(0.3))
                    self.q = nn.Sequential(nn.Linear(nv, 768), nn.Tanh(), nn.Dropout(0.3))
                    self.head = nn.Sequential(nn.Linear(768 * 2, 512), nn.ReLU(), nn.Linear(512, 4))

                def forward(self, f, q):
                    i, w = self.img(f), self.q(q)
                    o = torch.sigmoid(self.head(torch.cat([i * w, w], 1)))
                    cx, cy, bw, bh = o.unbind(1)
                    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).clamp(0, 1)

            model = Grounder()
            model.load_state_dict(state["model"])
            model.eval()
            mp = Path(key + ".metrics.json")
            _cache[key] = (model, state, load_encoder(), json.loads(mp.read_text()) if mp.exists() else {})
        return _cache[key]


def measured_confidence(cell_p: float, table: list[dict]) -> float:
    """Test-set Acc@0.5 of the probability bin `cell_p` falls in (bins sorted by p_max)."""
    if not table:
        return float(min(CAP, cell_p))
    for row in table:
        if cell_p <= row["p_max"]:
            return float(row["acc"])
    return float(table[-1]["acc"])


class HighResGrounder(Tool):
    spec = ToolSpec(
        name="hr_grounder",
        title="Approximate object locator (VRSBench)",
        description="Draws one box for a referred object (vehicle, ship, airplane, bridge, harbor, storage "
                    "tank, court, field...) in a sub-metre aerial image. Confidence is the measured hit rate "
                    "for similar predictions on VRSBench. Abstains on images coarser than 5 m per pixel.",
        tasks=(TaskType.GROUNDING,),
        input_configs=(InputConfig.SINGLE,),
        kind="learned (RS-adapted)",
        params=(ParamSpec("min_known_words", "int", 2, "Minimum expression words known to the model", 1, 10),),
    )

    def available(self) -> tuple[bool, str]:
        if not _path(settings.vrsbench_ground_checkpoint).exists():
            return False, "checkpoint not found (vrsbench_ground.pt); run training/train_vrsbench_grounding.py"
        return True, "trained on VRSBench referring expressions"

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        import torch

        r = next((x for x in ctx.rasters if x.modality.is_optical), None)
        if r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["Needs an optical image."])
        if r.gsd_m and r.gsd_m > MAX_GSD_M:
            return ToolResult(self.spec.name, "unsupported",
                              notes=[f"Trained on sub-metre aerial images; this image is {r.gsd_m:g} m per pixel."])
        obj = OBJECTS.search(ctx.query)
        model, state, body, metrics = _load()
        vocab = state["vocab"]
        if len([t for t in tokens(ctx.query) if t in vocab and "_" not in t]) < int(params.get("min_known_words", 2)):
            return ToolResult(self.spec.name, "unsupported", notes=["Expression is outside the VRSBench vocabulary."])
        if state.get("arch") == "dense":
            from satquery.models.hr_ground_net import encode_dense, project, token_ids
            f = project(encode_dense(body, [to_rgb(r)]), state["pca_mean"], state["pca_comp"], state["pca_scale"])
            bt, pt = model.predict(torch.from_numpy(f), torch.from_numpy(token_ids(ctx.query, vocab)[None]))
            b, cell_p = bt[0].numpy(), float(pt[0])
            conf = measured_confidence(cell_p, metrics.get("confidence_table") or [])
        else:
            img = (encode_batch(body, [to_rgb(r)])[0] - state["feat_mean"]) / state["feat_std"]
            with torch.inference_mode():
                b = model(torch.from_numpy(img[None].astype(np.float32)),
                          torch.from_numpy(encode_question(ctx.query, vocab)[None]))[0].numpy()
            cell_p = None
            conf = float(min(CAP, 0.15 + 0.6 * np.sqrt(float((b[2] - b[0]) * (b[3] - b[1])))))
        h, w = r.shape
        label = obj.group(0).lower() if obj else "referred object"
        area = float((b[2] - b[0]) * (b[3] - b[1]))
        box = Box(float(b[0] * w), float(b[1] * h), float(b[2] * w), float(b[3] * h), label=label, score=conf,
                  source=self.spec.name)
        acc = metrics.get("test_acc_at_0.5")
        answer = (f"Approximate location of the {label}: the box covers {100 * area:.1f}% of the image"
                  f" (x {100 * b[0]:.0f}-{100 * b[2]:.0f}%, y {100 * b[1]:.0f}-{100 * b[3]:.0f}% from the top-left)."
                  + (f" This locator finds the right object about {100 * acc:.0f}% of the time on VRSBench, "
                     "mostly for large objects." if acc else ""))
        return ToolResult(self.spec.name, answer=answer, boxes=[box], confidence=conf,
                          facts={"short_answer": "approximate box", "object_mentioned": bool(obj),
                                 "box_norm": [round(float(v), 3) for v in b],
                                 "cell_probability": None if cell_p is None else round(cell_p, 4),
                                 "vrsbench_acc_at_0.5": acc, "vrsbench_acc_by_class": metrics.get("acc_by_class")})
