"""SatQuery agentic controller.

    validate inputs -> plan (Claude | rules) -> validate plan against registry
    -> execute tools in order (sharing upstream outputs) -> integrate textual and
    spatial outputs -> confidence -> visual evidence -> auditable execution trace
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from satquery import __version__
from satquery.agent.plan import Plan, validate_plan
from satquery.agent.planner import ClaudePlanner, PlannerError, RulePlanner, has_anthropic_credentials
from satquery.agent.viewer import build_viewer
from satquery.config import settings
from satquery.imagery.raster import Raster
from satquery.imagery.render import colorize, draw_boxes, false_color, overlay, png_data_uri, to_rgb, upscale_for_display
from satquery.imagery.validate import InputConfig, validate_inputs
from satquery.tools.base import Box, TaskType, ToolContext, ToolResult
from satquery.tools.registry import Registry, default_registry


@dataclass
class AgentResponse:
    request_id: str
    status: str                      # ok | partial | rejected
    query: str
    task: str | None
    answer: str
    short_answer: str | None
    confidence: float | None
    confidence_label: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    visuals: list[dict[str, Any]] = field(default_factory=list)
    boxes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    viewer: dict[str, Any] = field(default_factory=dict)   # layered payload for the interactive canvas
    # report images are expensive to encode and only needed for downloads: built on first use
    visual_builder: Any = field(default=None, repr=False, compare=False)

    def get_visuals(self) -> list[dict[str, Any]]:
        if not self.visuals and self.visual_builder is not None:
            self.visuals = self.visual_builder()
            self.visual_builder = None
        return self.visuals

    def to_dict(self, include_visuals: bool = True) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "visual_builder"}
        if not include_visuals:
            d["visuals"] = [{k: v for k, v in vis.items() if k != "png"} for vis in self.visuals]
            d["viewer"] = {k: v for k, v in self.viewer.items() if k in ("width", "height", "boxes", "geo")}
        return d


def confidence_label(c: float | None) -> str:
    if c is None:
        return "unknown"
    return "high" if c >= 0.75 else "medium" if c >= 0.5 else "low"


class SatQueryAgent:
    def __init__(self, registry: Registry | None = None, planner: Any = None):
        self.registry = registry or default_registry()
        self._planner = planner
        self.rules = RulePlanner()

    # ------------------------------------------------------------- planning
    def _primary_planner(self) -> Any:
        if self._planner is not None:
            return self._planner
        mode = settings.planner
        if mode == "rules" or (mode == "auto" and not has_anthropic_credentials()):
            return self.rules
        self._planner = ClaudePlanner(self.registry)
        return self._planner

    def plan(self, query: str, report) -> tuple[Plan, float]:
        t = time.perf_counter()
        planner = self._primary_planner()
        notes: list[str] = []
        try:
            plan = planner.plan(query, report, self.registry)
        except PlannerError as e:
            notes.append(f"{planner.name} unavailable ({e}); fell back to the rule planner")
            plan = self.rules.plan(query, report, self.registry)
        plan.corrections = notes + plan.corrections
        plan = validate_plan(plan, self.registry, report.config)
        if not plan.steps:
            fallback = validate_plan(self.rules.plan(query, report, self.registry), self.registry, report.config)
            fallback.corrections = plan.corrections + ["proposed plan had no executable steps; used rule planner"]
            plan = fallback
        return plan, (time.perf_counter() - t) * 1000

    # ------------------------------------------------------------- main entry
    def run(self, query: str, rasters: list[Raster], pair_hint: str | None = None) -> AgentResponse:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        query = (query or "").strip()
        report = validate_inputs(rasters, pair_hint)
        trace: dict[str, Any] = {
            "request_id": rid, "started_utc": started, "satquery_version": __version__, "query": query,
            "input": {"configuration": report.config.value, "pair_hint": pair_hint,
                      "images": [r.summary() for r in report.rasters]},
            "validation": [c.to_dict() for c in report.checks],
        }
        if not query:
            report.add("query", "fail", "Empty query.")
            trace["validation"] = [c.to_dict() for c in report.checks]
        if not report.ok:
            fails = [c.message for c in report.checks if c.status == "fail"]
            trace["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            trace["outcome"] = "rejected at input validation"
            viewer = build_viewer(report.rasters, {}, []) if report.rasters and \
                all(r.shape == report.rasters[0].shape for r in report.rasters) else {}
            return AgentResponse(rid, "rejected", query, None, "Input rejected: " + " ".join(fails), None, None,
                                 "unknown", warnings=report.warnings, trace=trace,
                                 visuals=self._input_visuals(report.rasters), viewer=viewer)

        plan, plan_ms = self.plan(query, report)
        trace["planner"] = {"name": plan.planner, "duration_ms": round(plan_ms, 1)}
        trace["plan"] = plan.to_dict()

        ctx = ToolContext(query=query, config=report.config, rasters=report.rasters)
        steps_trace = []
        for i, step in enumerate(plan.steps, start=1):
            tool = self.registry.get(step.tool)
            params = {**tool.defaults(), **step.params}
            ts = time.perf_counter()
            try:
                res = tool.run(ctx, params)
            except Exception as exc:  # a failing specialist must not take down the request
                res = ToolResult(tool.spec.name, "error", notes=[f"{type(exc).__name__}: {exc}"])
            dur = (time.perf_counter() - ts) * 1000
            ctx.upstream[tool.spec.name] = res
            steps_trace.append({
                "step": i, "tool": tool.spec.name, "title": tool.spec.title, "kind": tool.spec.kind,
                "purpose": step.purpose, "params": _jsonable(params), "status": res.status,
                "duration_ms": round(dur, 1),
                "confidence": None if res.confidence is None else round(float(res.confidence), 3),
                "output": res.answer, "facts": _jsonable(res.facts), "notes": res.notes,
                "n_boxes": len(res.boxes), "layers": [l.name for l in res.layers],
            })
        trace["steps"] = steps_trace

        answer, short, conf, primary, evidence, boxes, notes = self._integrate(plan, ctx, report.warnings)
        status = "ok" if any(r.ok for r in ctx.upstream.values()) else "partial"
        trace["result"] = {"task": plan.task.value, "intent": plan.intent, "primary_tool": primary,
                           "answer": answer, "short_answer": short,
                           "confidence": None if conf is None else round(conf, 3),
                           "confidence_label": confidence_label(conf), "integration_notes": notes}
        viewer = build_viewer(ctx.rasters, ctx.upstream, boxes)
        trace["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        trace["outcome"] = status
        return AgentResponse(rid, status, query, plan.task.value, answer, short, conf, confidence_label(conf),
                             evidence=evidence, visual_builder=lambda: self._visuals(ctx, boxes), boxes=viewer["boxes"],
                             warnings=report.warnings, trace=trace, viewer=viewer)

    # ------------------------------------------------------------- integration
    def _integrate(self, plan: Plan, ctx: ToolContext, warnings: list[str]):
        up = {k: v for k, v in ctx.upstream.items()}
        ok = {k: v for k, v in up.items() if v.ok}
        notes: list[str] = []
        task = plan.task
        vlm = ok.get("rs_vlm")
        primary_name: str | None = None
        answer, short, conf = None, None, None
        boxes: list[Box] = []

        if task in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION):
            primary_name = "change_qa" if "change_qa" in ok else next(
                (n for n in ("change_semantics", "change_detector") if n in ok), None)
            cv = ok.get("cdvqa_vqa")
            if task is TaskType.CHANGE_VQA and cv is not None:
                primary_name = "cdvqa_vqa"     # trained change specialist; measured change evidence cross-checks it
                a = cv.facts.get("short_answer", "")
                b = ok["change_qa"].facts.get("short_answer", "") if "change_qa" in ok else ""
                if a in ("yes", "no") and b in ("yes", "no"):
                    notes.append("change specialist confirmed by measured change" if a == b else
                                 f"change specialist ('{a}') and measured change ('{b}') disagree")
            boxes = ok["change_detector"].boxes if "change_detector" in ok else []
        elif task is TaskType.CROSS_MODAL:
            rq = ok.get("landcover_reasoner")
            if plan.intent == "locate" and "region_grounder" in ok:
                primary_name = "region_grounder"
            elif rq is not None:
                primary_name = "landcover_reasoner"
            else:
                primary_name = next((n for n in ("optical_sar_fusion", "landcover_optical", "landcover_sar") if n in ok),
                                    None)
            if "region_grounder" in ok:
                boxes = ok["region_grounder"].boxes
        elif task is TaskType.GROUNDING:
            g = ok.get("region_grounder")
            if vlm is not None and vlm.boxes:
                primary_name = "rs_vlm"
                boxes = self._verify_vlm_boxes(vlm.boxes, ctx, notes)
                if g is not None:
                    boxes += g.boxes
            else:
                hg = ok.get("hr_grounder")   # answers only on sub-metre / unknown-scale imagery
                if hg is not None and (g is None or hg.facts.get("object_mentioned")):
                    # a named object (vehicle, ship, bridge...) beats a land-cover word in the same expression
                    primary_name, boxes = "hr_grounder", list(hg.boxes)
                else:
                    primary_name = "region_grounder" if g is not None else None
                    boxes = g.boxes if g is not None else []
        elif task is TaskType.CAPTION:
            # hr_vqa only answers on sub-metre imagery, where the EuroSAT classifier (10 m) is out of its domain
            primary_name = next((n for n in ("rs_vlm", "hr_vqa", "rs_scene_classifier", "landcover_reasoner") if n in ok), None)
        else:  # VQA
            rq = ok.get("landcover_reasoner")
            if vlm is not None and rq is not None:
                a, b = vlm.facts.get("short_answer", ""), rq.facts.get("short_answer", "")
                if a in ("yes", "no") and b in ("yes", "no"):
                    if a == b:
                        primary_name = "rs_vlm"
                        notes.append("VLM answer confirmed by land-cover evidence")
                    else:
                        primary_name = "rs_vlm" if (vlm.confidence or 0) >= (rq.confidence or 0) else "landcover_reasoner"
                        notes.append(f"VLM ('{a}') and land-cover evidence ('{b}') disagree; reporting the more "
                                     f"confident source and lowering confidence")
                else:
                    primary_name = "rs_vlm"
            else:
                sc = ok.get("rs_scene_classifier")
                rv = ok.get("rsvqa_vqa")
                hv = ok.get("hr_vqa")      # answers only on sub-metre / unknown-scale imagery
                if vlm is not None:
                    primary_name = "rs_vlm"
                elif hv is not None:
                    primary_name = "hr_vqa"   # objects and layout in high-resolution aerial images (VRSBench-trained)
                elif sc is not None and (sc.facts.get("question_type") == "land_use"
                                         or (sc.facts.get("scene_specific") and rv is None)):
                    # "what kind of area is this?", or a land-use class nothing else can answer
                    primary_name = "rs_scene_classifier"
                elif rv is not None:
                    primary_name = "rsvqa_vqa"             # trained VQA specialist; land-cover evidence cross-checks it
                    a, b = rv.facts.get("short_answer", ""), (rq.facts.get("short_answer", "") if rq else "")
                    if a in ("yes", "no") and b in ("yes", "no"):
                        notes.append("VQA specialist confirmed by land-cover evidence" if a == b else
                                     f"VQA specialist ('{a}') and land-cover evidence ('{b}') disagree")
                elif sc is not None and rq is None:
                    primary_name = "rs_scene_classifier"   # land-use question the land-cover reasoner cannot answer
                    if rq is not None:
                        notes.append("land-use question answered by the EuroSAT-adapted classifier")
                else:
                    primary_name = "landcover_reasoner" if rq is not None else None

        if primary_name is None or primary_name not in ok:
            failed = [f"{n}: {', '.join(r.notes) or r.status}" for n, r in up.items() if not r.ok]
            answer = ("SatQuery could not answer this query with the available tools. " +
                      ("Details: " + " | ".join(failed) if failed else ""))
            return answer, None, None, None, self._evidence(ok, None), boxes, notes

        prim = ok[primary_name]
        answer = prim.answer or ""
        short = prim.facts.get("short_answer")
        conf = prim.confidence if prim.confidence is not None else 0.5
        # agreement / disagreement adjustments
        if any("disagree" in n for n in notes):
            conf *= 0.7
        elif any("confirmed" in n for n in notes):
            xc = ok.get("change_qa") if primary_name == "cdvqa_vqa" else ok.get("landcover_reasoner")
            other = (xc.confidence if xc is not None else None) or 0.5
            conf = 1 - (1 - conf) * (1 - 0.5 * other)
        if task is TaskType.CAPTION and primary_name != "landcover_reasoner" and "landcover_reasoner" in ok:
            answer += "\n\nMeasured composition: " + ok["landcover_reasoner"].answer
        if task is TaskType.CROSS_MODAL:
            fus = ok.get("optical_sar_fusion")
            if fus is not None:
                if primary_name != "optical_sar_fusion":
                    answer += "\n\n" + (fus.answer or "")
                ins = fus.facts.get("insights") or []
                if ins:
                    answer += "\n" + " ".join(ins)
        if task is TaskType.GROUNDING and primary_name == "rs_vlm" and "region_grounder" in ok:
            answer += " " + (ok["region_grounder"].answer or "")
        if vlm is not None and primary_name != "rs_vlm" and vlm.answer:
            answer += f"\n\nSatQuery-VLM reading: {vlm.answer.strip()}"
        n_warn = len(warnings)
        if n_warn:
            conf *= max(0.7, 1 - 0.04 * n_warn)
            notes.append(f"confidence reduced for {n_warn} input warning(s)")
        return answer.strip(), short, float(np.clip(conf, 0.0, 0.99)), primary_name, \
            self._evidence(ok, primary_name), boxes, notes

    def _verify_vlm_boxes(self, boxes: list[Box], ctx: ToolContext, notes: list[str]) -> list[Box]:
        """Score VLM boxes by the fraction of land-cover evidence of the target class inside them."""
        g = ctx.upstream.get("region_grounder")
        if g is None or not g.ok or not g.layers:
            return boxes
        mask = np.zeros_like(g.layers[0].array, dtype=bool)
        for l in g.layers:
            mask |= l.array.astype(bool)
        out = []
        for b in boxes:
            y0, y1, x0, x1 = int(b.y0), max(int(b.y1), int(b.y0) + 1), int(b.x0), max(int(b.x1), int(b.x0) + 1)
            support = float(mask[y0:y1, x0:x1].mean()) if mask[y0:y1, x0:x1].size else 0.0
            out.append(Box(b.x0, b.y0, b.x1, b.y1, label=f"{b.label} (evidence {support:.0%})",
                           score=b.score, source=b.source))
            notes.append(f"VLM box '{b.label}': {support:.0%} of its area supported by land-cover evidence")
        return out

    @staticmethod
    def _evidence(ok: dict[str, ToolResult], primary: str | None) -> list[dict[str, Any]]:
        return [{"tool": n, "primary": n == primary, "text": r.answer,
                 "confidence": None if r.confidence is None else round(float(r.confidence), 3)}
                for n, r in ok.items() if r.answer]

    # ------------------------------------------------------------- visuals
    def _input_visuals(self, rasters: list[Raster]) -> list[dict[str, Any]]:
        out = []
        for i, r in enumerate(rasters):
            label = f"Input {i + 1}: {r.name} ({r.modality.value}" + (f", {r.acquired}" if r.acquired else "") + ")"
            out.append({"title": label, "kind": "input", "png": png_data_uri(upscale_for_display(to_rgb(r)))})
            fc = false_color(r)
            if fc is not None:
                out.append({"title": f"Input {i + 1}: false colour (NIR-R-G)", "kind": "input",
                            "png": png_data_uri(upscale_for_display(fc))})
        return out

    def _visuals(self, ctx: ToolContext, boxes: list[Box]) -> list[dict[str, Any]]:
        out = self._input_visuals(ctx.rasters)
        base = to_rgb(ctx.reference)
        if boxes:
            pal = {"rs_vlm": (0, 255, 255), "region_grounder": (255, 230, 0), "change_detector": (255, 0, 200)}
            img = upscale_for_display(base, 512)  # draw after upscaling so outlines/labels stay crisp
            k = img.shape[0] / base.shape[0]
            for src in {b.source for b in boxes}:
                scaled = [{**b.to_dict(), "x0": b.x0 * k, "y0": b.y0 * k, "x1": b.x1 * k, "y1": b.y1 * k}
                          for b in boxes if b.source == src]
                img = draw_boxes(img, scaled, pal.get(src, (255, 255, 0)))
            out.append({"title": "Detected / grounded regions", "kind": "boxes", "png": png_data_uri(img),
                        "legend": "cyan = VLM, yellow = grounder, magenta = change regions"})
        for res in ctx.upstream.values():
            if not res.ok:
                continue
            for layer in res.layers:
                if layer.kind == "mask":
                    if not layer.array.any():
                        continue
                    img = overlay(base, layer.array, layer.color, 0.55)
                else:
                    img = colorize(np.nan_to_num(layer.array.astype(np.float32), nan=layer.vmin or 0.0),
                                   layer.cmap, layer.vmin, layer.vmax)
                out.append({"title": layer.name, "kind": layer.kind, "tool": res.tool,
                            "legend": layer.legend, "png": png_data_uri(upscale_for_display(img))})
        return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return f"<array {x.shape}>"
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)
