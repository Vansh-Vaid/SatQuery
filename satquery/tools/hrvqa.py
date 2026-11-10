"""hr_vqa: VQA specialist for high-resolution (sub-metre) aerial images, trained on VRSBench.

Frozen ImageNet ResNet-18 with a 3x3 spatial grid + bag-of-words question encoding, fused by elementwise
product, over the 300 most frequent VRSBench answers (object existence, quantity, colour, position, category,
size, shape, direction, scene type, rural/urban). It abstains on images coarser than 5 m per pixel, where the
RSVQA specialist (trained on 10 m Sentinel-2) applies instead.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.render import to_rgb
from satquery.imagery.validate import InputConfig
from satquery.models.hr_features import encode_batch, load_encoder
from satquery.models.rsvqa_features import build_model, encode_question, tokens
from satquery.tools.base import ParamSpec, TaskType, Tool, ToolContext, ToolResult, ToolSpec

_lock = threading.Lock()
_cache: dict[str, Any] = {}
MAX_GSD_M = 5.0


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(__file__).resolve().parents[2] / q


def _load():
    key = str(_path(settings.vrsbench_checkpoint))
    with _lock:
        if key not in _cache:
            import torch
            state = torch.load(key, map_location="cpu", weights_only=False)
            model = build_model(len(state["vocab"]), img_dim=state["img_dim"], n_answers=len(state["answers"]),
                                hidden=state.get("hidden", 512))
            model.load_state_dict(state["model"])
            model.eval()
            mp = Path(key + ".metrics.json")
            _cache[key] = (model, state, load_encoder(), json.loads(mp.read_text()) if mp.exists() else {})
        return _cache[key]


class HighResVQA(Tool):
    spec = ToolSpec(
        name="hr_vqa",
        title="High-resolution VQA specialist (VRSBench)",
        description="Visual question answering for sub-metre aerial images, trained on VRSBench: whether objects "
                    "exist (planes, ships, vehicles, bridges...), how many, their colour, position, size, shape, "
                    "direction, the scene type and rural/urban. Abstains on images coarser than 5 m per pixel.",
        tasks=(TaskType.VQA, TaskType.CAPTION),
        input_configs=(InputConfig.SINGLE,),
        kind="learned (RS-adapted)",
        params=(ParamSpec("min_known_words", "int", 2, "Minimum question words known to the model before it "
                                                       "answers (otherwise it abstains)", 1, 10),
                ParamSpec("mode", "enum", "qa", "qa: answer the question | describe: compose a scene description "
                                                "from the model's own answers", choices=("qa", "describe"))),
    )

    def available(self) -> tuple[bool, str]:
        if not _path(settings.vrsbench_checkpoint).exists():
            return False, "checkpoint not found (vrsbench_vqa.pt); run training/train_vrsbench_vqa.py"
        return True, "trained on VRSBench"

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        import torch

        r = next((x for x in ctx.rasters if x.modality.is_optical), None)
        if r is None:
            return ToolResult(self.spec.name, "unsupported", notes=["Trained on optical imagery; no optical image supplied."])
        if r.gsd_m and r.gsd_m > MAX_GSD_M:
            return ToolResult(self.spec.name, "unsupported",
                              notes=[f"Trained on sub-metre aerial images; this image is {r.gsd_m:g} m per pixel."])
        model, state, body, metrics = _load()
        vocab = state["vocab"]
        describe = params.get("mode") == "describe"
        if not describe and len([t for t in tokens(ctx.query) if t in vocab and "_" not in t]) < int(params.get("min_known_words", 2)):
            return ToolResult(self.spec.name, "unsupported", notes=["Question is outside the VRSBench vocabulary; abstaining."])
        img = torch.from_numpy(((encode_batch(body, [to_rgb(r)])[0] - state["feat_mean"]) / state["feat_std"])[None].astype(np.float32))

        def ask(question: str) -> tuple[str, float, np.ndarray]:
            with torch.inference_mode():
                lg = model(img, torch.from_numpy(encode_question(question, vocab)[None]))[0].numpy()
            pr = np.exp(lg - lg.max())
            pr /= pr.sum()
            i = int(pr.argmax())
            return state["answers"][i], float(pr[i]), pr

        if describe:
            return self._describe(ask, metrics)
        ans, _, p = ask(ctx.query)
        k = int(p.argmax())
        top = {state["answers"][i]: round(float(p[i]), 3) for i in np.argsort(-p)[:4]}
        facts = {"short_answer": ans, "answer_distribution": top,
                 "model": "ImageNet ResNet-18 (3x3 grid) + bag-of-words, trained on VRSBench",
                 "vrsbench_test_accuracy": metrics.get("test_accuracy"),
                 "vrsbench_test_per_type": metrics.get("test_per_type")}
        notes = [] if r.gsd_m else ["Pixel size unknown; assumed to be high-resolution aerial imagery."]
        return ToolResult(self.spec.name, answer=f"{ans[0].upper()}{ans[1:]}. ({100 * p[k]:.0f}% model confidence)",
                          facts=facts, confidence=float(p[k]), notes=notes)

    def _describe(self, ask, metrics) -> ToolResult:
        """A scene description built only from the model's own answers about this image (no templates from
        other images): setting, main object, how many, where. Parts the model is unsure of are left out."""
        asked: dict[str, tuple[str, float]] = {}
        area, pa, _ = ask("Does the image show a rural or urban area?")
        main, pm, _ = ask("What is the main object in the image?")
        main = main.replace("-", " ")   # dataset labels such as "small-vehicle", "tennis-court"
        asked["setting"], asked["main object"] = (area, pa), (main, pm)
        parts = []
        setting = area if area in ("rural", "urban") and pa >= 0.4 else ""
        if pm >= 0.2 and not main.isdigit() and main not in ("yes", "no"):
            plural = main if main.endswith("s") else main + "s"
            n, pn, _ = ask(f"How many {plural} are visible in the image?")
            pos, pp, _ = ask(f"Where is the {main} located in the image?")
            asked["count"], asked["position"] = (n, pn), (pos, pp)
            lead = f"{'An' if setting.startswith('u') else 'A'} {setting} scene" if setting else "An aerial scene"
            if n.isdigit() and pn >= 0.3:
                k = int(n)
                obj = f"{k} {main if k == 1 else plural}"
                parts.append(f"{lead} with {obj} visible")
            else:
                parts.append(f"{lead} whose main features are {main}" if main.endswith("s")
                             else f"{lead} whose main feature is {'an' if main[0] in 'aeiou' else 'a'} {main}")
            if pp >= 0.3 and pos not in ("yes", "no") and not pos.isdigit():
                parts.append(f"mainly in the {pos} of the image")
        elif setting:
            parts.append(f"{'An' if setting.startswith('u') else 'A'} {setting} aerial scene")
        if not parts:
            return ToolResult(self.spec.name, "unsupported", notes=["The model was not confident about this scene."])
        caption = ", ".join(parts) + "."
        conf = float(np.mean([v[1] for v in asked.values()]))
        return ToolResult(self.spec.name, answer=caption, confidence=conf,
                          facts={"short_answer": caption, "asked": {k: {"answer": a, "p": round(p, 3)} for k, (a, p) in asked.items()},
                                 "model": "VRSBench-trained VQA specialist, describe mode",
                                 "vrsbench_test_accuracy": metrics.get("test_accuracy")})
