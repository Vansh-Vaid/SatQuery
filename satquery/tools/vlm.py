"""rs_vlm: the remote-sensing-adapted vision-language model as a registry tool."""
from __future__ import annotations

import logging
import re
from typing import Any

from satquery.config import settings
from satquery.imagery.validate import InputConfig
from satquery.models.vlm_backend import HFBackend, RemoteBackend, VLMBackend
from satquery.models.vlm_prompts import box_to_pixels, build_prompt, parse_boxes, render_for_vlm
from satquery.tools.base import Box, ParamSpec, TaskType, Tool, ToolContext, ToolResult, ToolSpec

log = logging.getLogger(__name__)


def make_backend() -> VLMBackend | None:
    if settings.vlm_backend == "hf":
        return HFBackend(settings.vlm_base_model, settings.vlm_adapter)
    if settings.vlm_backend == "remote" and settings.vlm_endpoint:
        return RemoteBackend(settings.vlm_endpoint, settings.vlm_timeout_s)
    return None


def short_answer(text: str) -> str:
    t = text.strip().lower()
    m = re.match(r"^(yes|no)\b", t)
    if m:
        return m.group(1)
    return re.sub(r"[.\s]+$", "", t.split("\n")[0])[:120]


class RSVLM(Tool):
    spec = ToolSpec(
        name="rs_vlm",
        title="SatQuery-VLM (BigEarthNet.txt-adapted)",
        description="Vision-language model (Qwen3-VL-2B base) LoRA-fine-tuned on BigEarthNet.txt Sentinel-1/2 "
                    "image-text data. Free-form VQA, captioning and referring-expression grounding (boxes) on one "
                    "image; joint question answering over an optical+SAR pair (multi-image input); auxiliary "
                    "change description over a bi-temporal pair.",
        tasks=(TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING, TaskType.CROSS_MODAL,
               TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA),
        input_configs=(InputConfig.SINGLE, InputConfig.CROSS_MODAL, InputConfig.BITEMPORAL),
        kind="learned (RS-adapted)",
        params=(
            ParamSpec("mode", "enum", "auto", "vqa | caption | grounding | auto",
                      choices=("auto", "vqa", "caption", "grounding")),
            ParamSpec("max_new_tokens", "int", 160, "Generation length cap", 8, 512),
            ParamSpec("expression", "str", "", "Referring expression for grounding (default: the query)"),
        ),
    )

    def __init__(self, backend: VLMBackend | None = None, backend_factory=make_backend):
        self._backend = backend
        self._factory = backend_factory
        self._resolved = backend is not None

    @property
    def backend(self) -> VLMBackend | None:
        if not self._resolved:
            self._backend = self._factory()
            self._resolved = True
        return self._backend

    def available(self) -> tuple[bool, str]:
        b = self.backend
        if b is None:
            return False, ("no VLM backend configured (set SATQUERY_VLM_BACKEND=hf with SATQUERY_VLM_ADAPTER, or "
                           "=remote with SATQUERY_VLM_ENDPOINT)")
        return True, b.name

    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        ok, why = self.available()
        if not ok:
            return ToolResult(self.spec.name, "unavailable", notes=[why])
        mode = params.get("mode", "auto")
        if mode != "auto":          # an explicit task wins over the guess from the image pair
            task = mode
        elif ctx.config is InputConfig.BITEMPORAL:
            task = "change"
        else:
            q = ctx.query.lower()
            task = ("grounding" if re.search(r"\b(highlight|locate|where is|find|mark|box|point out|show me)\b", q)
                    else "caption" if re.search(r"\b(describe|caption|summar|overview)\b", q) else "vqa")
        rasters = ctx.rasters
        images = [render_for_vlm(r) for r in rasters]
        # `prompt` (benchmarks only, not exposed to the planner): the exact instruction the model was
        # trained with, so scoring does not depend on how a rendered PNG's modality is re-guessed on load
        prompt = params.get("prompt") or build_prompt(task, ctx.query, [r.modality for r in rasters],
                                                      [r.acquired for r in rasters],
                                                      expression=params.get("expression") or None)
        try:
            text, conf = self.backend.generate(images, prompt, int(params.get("max_new_tokens", 160)))
        except Exception as exc:  # network / OOM / missing weights -> degrade, never crash the request
            log.exception("VLM inference failed")
            return ToolResult(self.spec.name, "error", notes=[f"VLM inference failed: {type(exc).__name__}: {exc}"])
        facts: dict[str, Any] = {"task": task, "backend": self.backend.name, "raw_output": text[:2000]}
        boxes: list[Box] = []
        answer = text
        if task == "grounding":
            h, w = ctx.reference.shape
            for i, b in enumerate(parse_boxes(text)):
                x0, y0, x1, y1 = box_to_pixels(b["bbox_2d"], h, w)
                if x1 - x0 >= 1 and y1 - y0 >= 1:
                    boxes.append(Box(x0, y0, x1, y1, label=b["label"] or f"target {i + 1}", score=conf,
                                     source=self.spec.name))
            answer = (f"Located {len(boxes)} region(s) for '{params.get('expression') or ctx.query}'."
                      if boxes else "The model did not return a valid box.")
            facts["boxes_0_1000"] = [b["bbox_2d"] for b in parse_boxes(text)]
        else:
            facts["short_answer"] = short_answer(text)
        return ToolResult(self.spec.name, answer=answer, facts=facts, boxes=boxes, confidence=conf)
