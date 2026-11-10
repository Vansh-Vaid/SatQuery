"""rsvqa_vqa: VQA specialist trained on RSVQA-LR (training/train_rsvqa_lr.py).

Frozen EuroSAT-adapted ResNet-18 image features + land-cover fractions, fused with a bag-of-words question
encoding. Answers presence, comparison, rural/urban and count questions about optical scenes. Test accuracy
on the official RSVQA-LR split is stored in the checkpoint's metrics file and reported in every answer's facts.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.validate import InputConfig
from satquery.models.rsvqa_features import (ANSWERS, BIN_TEXT, answer_mask, build_model, encode_question,
                                            image_features, load_backbone, question_type, tokens)
from satquery.tools.base import ParamSpec, Tool, ToolContext, ToolResult, ToolSpec, TaskType

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(__file__).resolve().parents[2] / q


def _load():
    key = str(_path(settings.rsvqa_checkpoint))
    with _lock:
        if key not in _cache:
            import torch
            state = torch.load(key, map_location="cpu", weights_only=False)
            model = build_model(len(state["vocab"]))
            model.load_state_dict(state["model"])
            model.eval()
            backbone, mean, std = load_backbone(str(_path(settings.scene_checkpoint)))
            mp = Path(key + ".metrics.json")
            metrics = json.loads(mp.read_text()) if mp.exists() else {}
            _cache[key] = (model, state, backbone, mean, std, metrics)
        return _cache[key]


class RSVQASpecialist(Tool):
    spec = ToolSpec(
        name="rsvqa_vqa",
        title="RSVQA-trained VQA specialist",
        description="Visual question answering model trained on RSVQA-LR (Sentinel-2 scenes with OpenStreetMap-"
                    "derived questions). Answers presence ('is there a road?'), comparison ('are there more "
                    "buildings than water areas?'), rural/urban and count questions (count answers in ranges). "
                    "Uses the EuroSAT-adapted image encoder. Optical input only.",
        tasks=(TaskType.VQA,),
        input_configs=(InputConfig.SINGLE,),
        kind="learned (RS-adapted)",
        params=(ParamSpec("min_known_words", "int", 2, "Minimum question words known to the model before it "
                                                       "answers (otherwise it abstains)", 1, 10),),
    )

    def available(self) -> tuple[bool, str]:
        for p in (settings.rsvqa_checkpoint, settings.scene_checkpoint):
            if not _path(p).exists():
                return False, f"checkpoint not found ({Path(p).name}); run training/train_rsvqa_lr.py"
        return True, "trained on RSVQA-LR"

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        import torch

        r = next((x for x in ctx.rasters if x.modality.is_optical), None)
        if r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["Trained on optical imagery; no optical image supplied."])
        if r.gsd_m and r.gsd_m < 3:
            return ToolResult(self.spec.name, "unsupported",
                              notes=[f"Trained on ~10 m Sentinel-2 scenes; this image is {r.gsd_m:g} m per pixel."])
        model, state, backbone, mean, std, metrics = _load()
        vocab = state["vocab"]
        known = [t for t in tokens(ctx.query) if t in vocab and "_" not in t]
        if len(known) < int(params.get("min_known_words", 2)):
            return ToolResult(self.spec.name, "unsupported",
                              notes=["Question is outside the RSVQA vocabulary; abstaining."])
        qtype = question_type(ctx.query)
        # RSVQA questions are closed-form: yes/no, rural/urban or counts. Abstain on open questions.
        import re
        if qtype == "yesno" and not re.match(r"^\s*(is|are|does|do|was|were|can|has|have)\b", ctx.query.lower()):
            return ToolResult(self.spec.name, "unsupported",
                              notes=["Open-ended question; the RSVQA specialist only answers yes/no, rural/urban "
                                     "and count questions."])
        img = (image_features(r, backbone, mean, std) - state["feat_mean"]) / state["feat_std"]
        q = encode_question(ctx.query, vocab)
        with torch.inference_mode():
            logits = model(torch.from_numpy(img[None].astype(np.float32)), torch.from_numpy(q[None]))[0].numpy()
        mask = answer_mask(qtype)
        logits = np.where(mask, logits, -1e9)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(p.argmax())
        ans = ANSWERS[k]
        text_ans = BIN_TEXT.get(ans, ans)
        dist = {ANSWERS[i]: round(float(p[i]), 3) for i in np.argsort(-p) if mask[i]}
        facts = {"short_answer": text_ans, "question_type": qtype, "answer_distribution": dist,
                 "model": "EuroSAT ResNet-18 features + bag-of-words, trained on RSVQA-LR",
                 "rsvqa_lr_test_accuracy": metrics.get("test_overall_accuracy"),
                 "rsvqa_lr_test_per_type": metrics.get("test_per_type")}
        notes = []
        if r.gsd_m and not (5 <= r.gsd_m <= 20):
            notes.append(f"Pixel size {r.gsd_m} m differs from RSVQA-LR (~10 m); answers are less reliable.")
        sentence = {"count": f"About {text_ans} (count range).", "rural_urban": f"This looks like a{'n' if ans == 'urban' else ''} {ans} area."}
        answer = sentence.get(qtype, f"{ans.capitalize()}.") + f" ({100 * p[k]:.0f}% model confidence)"
        return ToolResult(self.spec.name, answer=answer, facts=facts, confidence=float(p[k]), notes=notes)
