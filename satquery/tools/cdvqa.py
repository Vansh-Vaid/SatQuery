"""cdvqa_vqa: change-question specialist trained on CDVQA (training/train_cdvqa.py).

Answers questions about what changed between two co-registered optical images: did a class change / increase /
decrease, the percentage of changed area (overall or per class), what a class changed into, and the largest or
smallest change. Its classes are those of CDVQA/SECOND: buildings, low vegetation, non-vegetated ground surface,
playgrounds, trees, water. Test accuracy is stored with the checkpoint and reported in every answer's facts.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.validate import InputConfig
from satquery.models.cdvqa_features import label_text, pair_features
from satquery.models.rsvqa_features import build_model, encode_question, load_backbone, tokens
from satquery.tools.base import ParamSpec, TaskType, Tool, ToolContext, ToolResult, ToolSpec

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(__file__).resolve().parents[2] / q


def _load():
    key = str(_path(settings.cdvqa_checkpoint))
    with _lock:
        if key not in _cache:
            import torch
            state = torch.load(key, map_location="cpu", weights_only=False)
            model = build_model(len(state["vocab"]), img_dim=state["img_dim"], n_answers=len(state["answers"]))
            model.load_state_dict(state["model"])
            model.eval()
            backbone, mean, std = load_backbone(str(_path(settings.scene_checkpoint)))
            mp = Path(key + ".metrics.json")
            _cache[key] = (model, state, backbone, mean, std, json.loads(mp.read_text()) if mp.exists() else {})
        return _cache[key]


class CDVQASpecialist(Tool):
    spec = ToolSpec(
        name="cdvqa_vqa",
        title="CDVQA-trained change-question specialist",
        description="Answers questions about change between two dates of optical imagery, trained on CDVQA: did a "
                    "class change, increase or decrease; percentage of changed area overall or per class; what a "
                    "class changed into; largest / smallest change. Classes: buildings, low vegetation, "
                    "non-vegetated ground surface, playgrounds, trees, water.",
        tasks=(TaskType.CHANGE_VQA,),
        input_configs=(InputConfig.BITEMPORAL,),
        kind="learned (RS-adapted)",
        params=(ParamSpec("min_known_words", "int", 3, "Minimum question words known to the model before it "
                                                       "answers (otherwise it abstains)", 1, 10),),
    )

    def available(self) -> tuple[bool, str]:
        for p in (settings.cdvqa_checkpoint, settings.scene_checkpoint):
            if not _path(p).exists():
                return False, f"checkpoint not found ({Path(p).name}); run training/train_cdvqa.py"
        return True, "trained on CDVQA"

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        import torch

        opt = [r for r in ctx.rasters if r.modality.is_optical]
        if len(opt) < 2:
            return ToolResult(self.spec.name, "unsupported", notes=["Trained on optical image pairs only."])
        # CDVQA's questions all start with What / Have / Did / How (+ "Has", "Is", "Are" variants) and never
        # offer choices. "Increased, decreased or unchanged?" is outside what it learned: the measured tools answer.
        import re
        ql = ctx.query.lower().strip()
        if re.search(r"\bor\b", ql) or not re.match(r"(what|have|has|did|does|how|is|are|was|were)\b", ql):
            return ToolResult(self.spec.name, "unsupported",
                              notes=["Question form not covered by CDVQA (choice or open question); abstaining."])
        model, state, backbone, mean, std, metrics = _load()
        vocab = state["vocab"]
        if len([t for t in tokens(ctx.query) if t in vocab and "_" not in t]) < int(params.get("min_known_words", 3)):
            return ToolResult(self.spec.name, "unsupported", notes=["Question is outside the CDVQA vocabulary; abstaining."])
        img = (pair_features(opt[0], opt[1], backbone, mean, std) - state["feat_mean"]) / state["feat_std"]
        q = encode_question(ctx.query, vocab)
        with torch.inference_mode():
            logits = model(torch.from_numpy(img[None].astype(np.float32)), torch.from_numpy(q[None]))[0].numpy()
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(p.argmax())
        label = state["answers"][k]
        text = label_text(label)
        top = {label_text(state["answers"][i]): round(float(p[i]), 3) for i in np.argsort(-p)[:4]}
        facts = {"short_answer": text, "label": label, "answer_distribution": top,
                 "model": "EuroSAT ResNet-18 pair features + bag-of-words, trained on CDVQA",
                 "cdvqa_test_accuracy": metrics.get("test_overall_accuracy"),
                 "cdvqa_test_per_type": metrics.get("test_per_type")}
        answer = f"{text[0].upper()}{text[1:]}. ({100 * p[k]:.0f}% model confidence)"
        notes = ["CDVQA was built from 0.5-3 m aerial image pairs; other resolutions are less reliable."] \
            if opt[0].gsd_m and opt[0].gsd_m > 5 else []
        return ToolResult(self.spec.name, answer=answer, facts=facts, confidence=float(p[k]), notes=notes)
