"""Query interpretation and workflow planning.

``ClaudePlanner`` asks Claude (structured JSON output constrained to the
registry's tool names and the tasks allowed for the input configuration) to
classify the task and propose a tool sequence with permitted parameters.
``RulePlanner`` is the deterministic planner used offline or as fallback.
Either way the proposal is passed through ``validate_plan`` by the controller.
"""
from __future__ import annotations

import json
import re
from typing import Any

from satquery.agent.plan import ALLOWED_TASKS, INTENTS, Plan, PlanStep
from satquery.config import settings
from satquery.imagery.raster import Modality
from satquery.imagery.validate import InputConfig, ValidationReport
from satquery.tools.base import TaskType
from satquery.tools.registry import Registry

_LOCATE = re.compile(r"\b(highlight|locate|localize|localise|where is|where are|find|mark|outline|segment|box|"
                     r"bounding|point out|show me|delineate|identify (the|all)?\s*\w*\s*(regions?|areas?|zones?)|"
                     r"identify .* regions|map the)\b")
_DESCRIBE = re.compile(r"\b(describe|description|caption|summari[sz]e|overview|what (can you see|is visible|is in)|"
                       r"tell me about|what changed|what has changed|what are the changes|explain)\b")


def detect_intent(query: str, config: InputConfig) -> str:
    q = query.lower()
    if config is InputConfig.BITEMPORAL:
        return "describe" if (_DESCRIBE.search(q) or re.search(r"\bwhere\b", q)) and not re.match(
            r"^\s*(is|are|has|have|did|does|do)\b", q) else "qa"
    if _LOCATE.search(q):
        return "locate"
    if _DESCRIBE.search(q):
        return "describe"
    return "qa"


class RulePlanner:
    name = "rules"

    def plan(self, query: str, report: ValidationReport, registry: Registry) -> Plan:
        cfg = report.config
        intent = detect_intent(query, cfg)
        vlm_ok = registry.available("rs_vlm")[0]
        scene_ok = registry.available("rs_scene_classifier")[0]
        rsvqa_ok = registry.available("rsvqa_vqa")[0]
        hr_ok = registry.available("hr_vqa")[0]
        optical = any(r.modality.is_optical for r in report.rasters)
        steps: list[PlanStep] = []
        if cfg is InputConfig.SINGLE:
            lc = "landcover_sar" if report.rasters[0].modality is Modality.SAR else "landcover_optical"
            if intent == "locate":
                task = TaskType.GROUNDING
                if vlm_ok:
                    steps.append(PlanStep("rs_vlm", {"mode": "grounding"}, "referring-expression grounding (boxes)"))
                steps += [PlanStep(lc, {}, "land-cover masks as grounding evidence"),
                          PlanStep("region_grounder", {}, "select regions matching the referred class")]
                if registry.available("hr_grounder")[0] and lc == "landcover_optical":
                    steps.append(PlanStep("hr_grounder", {}, "approximate box for a referred object (high-resolution images)"))
            elif intent == "describe":
                task = TaskType.CAPTION
                if vlm_ok:
                    steps.append(PlanStep("rs_vlm", {"mode": "caption"}, "scene caption"))
                steps += [PlanStep(lc, {}, "quantitative land-cover evidence"),
                          PlanStep("landcover_reasoner", {"mode": "describe"}, "land-cover scene description")]
                if hr_ok and optical:
                    steps.append(PlanStep("hr_vqa", {"mode": "describe"}, "object-level description (high-resolution images)"))
                if scene_ok and optical:
                    steps.append(PlanStep("rs_scene_classifier", {"mode": "describe"}, "EuroSAT-adapted land-use classification"))
            else:
                task = TaskType.VQA
                if vlm_ok:
                    steps.append(PlanStep("rs_vlm", {"mode": "vqa"}, "answer the question"))
                steps += [PlanStep(lc, {}, "land-cover evidence"),
                          PlanStep("landcover_reasoner", {"mode": "qa"}, "evidence-based answer / cross-check")]
                if scene_ok and optical:
                    steps.append(PlanStep("rs_scene_classifier", {"mode": "qa"}, "land-use answer from the adapted classifier"))
                if rsvqa_ok and optical:
                    steps.append(PlanStep("rsvqa_vqa", {}, "answer from the RSVQA-trained VQA specialist"))
                if hr_ok and optical:
                    steps.append(PlanStep("hr_vqa", {}, "answer from the VRSBench-trained high-resolution specialist"))
        elif cfg is InputConfig.CROSS_MODAL:
            task = TaskType.CROSS_MODAL
            steps = [PlanStep("landcover_optical", {}, "optical/multispectral land cover + cloud mask"),
                     PlanStep("landcover_sar", {}, "SAR land cover (cloud-independent)"),
                     PlanStep("optical_sar_fusion", {}, "fuse complementary optical and SAR evidence")]
            if intent == "locate" or re.search(r"\b(regions?|areas?)\b", query.lower()):
                steps.append(PlanStep("region_grounder", {}, "highlight the requested classes on the fused map"))
            steps.append(PlanStep("landcover_reasoner", {"mode": "qa" if intent == "qa" else "describe"},
                                  "answer from fused evidence"))
            if scene_ok:
                steps.append(PlanStep("rs_scene_classifier", {"mode": "qa" if intent == "qa" else "describe"},
                                      "land use of the optical image (EuroSAT-adapted)"))
            if vlm_ok:
                steps.append(PlanStep("rs_vlm", {"mode": "caption" if intent != "qa" else "vqa"},
                                      "joint optical+SAR reading by the RS-adapted VLM"))
        else:
            task = TaskType.CHANGE_DESCRIPTION if intent == "describe" else TaskType.CHANGE_VQA
            steps = [PlanStep("change_detector", {}, "where did it change"),
                     PlanStep("change_semantics", {}, "what changed into what"),
                     PlanStep("change_qa", {"mode": "describe" if intent == "describe" else "auto"},
                              "answer / describe the change")]
            if intent != "describe" and registry.available("cdvqa_vqa")[0] and all(
                    r.modality.is_optical for r in report.rasters):
                steps.append(PlanStep("cdvqa_vqa", {}, "answer from the CDVQA-trained change specialist"))
            if vlm_ok and intent == "describe":
                steps.append(PlanStep("rs_vlm", {}, "auxiliary visual change description"))
        return Plan(task=task, intent=intent, steps=steps, planner=self.name,
                    rationale=f"rule-based: {cfg.value} input, intent '{intent}'")


# ====================================================================== Claude
_SYSTEM = """You are the planning controller of SatQuery AI, an agentic remote-sensing assistant.
Given a user's natural-language query and a description of the validated input images, you:
1. classify the requested task (only tasks allowed for the input configuration),
2. classify the intent: "qa" (answer a question), "describe" (describe a scene or change), "locate" (highlight/box regions),
3. choose an ordered sequence of tools from the registry below, setting ONLY declared parameters.

Policy:
- Use only tools whose input_configs include the input configuration and that are listed as available.
- Place producers before consumers ("requires").
- Prefer combining the RS-adapted VLM (rs_vlm) with algorithmic evidence tools so answers are evidence-grounded.
- For cross-modal pairs always use landcover_optical, landcover_sar and optical_sar_fusion.
- For bi-temporal pairs always use change_detector, change_semantics and change_qa.
- Leave params empty unless the query clearly calls for a non-default (e.g. "subtle changes" -> higher sensitivity).
- Parameter values are strings; they will be type-checked and bounded against the registry.

Tasks: {tasks}
Allowed tasks per input configuration: {allowed}

Tool registry:
{registry}
"""


def _schema(registry: Registry) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": [t.value for t in TaskType]},
            "intent": {"type": "string", "enum": list(INTENTS)},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": registry.names()},
                        "purpose": {"type": "string"},
                        "params": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                            "required": ["name", "value"], "additionalProperties": False}},
                    },
                    "required": ["tool", "purpose", "params"],
                    "additionalProperties": False,
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["task", "intent", "steps", "rationale"],
        "additionalProperties": False,
    }


class PlannerError(RuntimeError):
    pass


class ClaudePlanner:
    def __init__(self, registry: Registry, model: str | None = None, effort: str | None = None, client: Any = None):
        self.model = model or settings.planner_model
        self.effort = effort or settings.planner_effort
        self.name = f"claude:{self.model}"
        self._client = client
        specs = [{k: v for k, v in t.spec.to_dict().items() if k != "version"} for t in registry.tools()]
        self._system = _SYSTEM.format(
            tasks=json.dumps(registry.tasks()),
            allowed=json.dumps({c.value: [t.value for t in ts] for c, ts in ALLOWED_TASKS.items()}),
            registry=json.dumps(specs, indent=1))
        self._schema = _schema(registry)

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def plan(self, query: str, report: ValidationReport, registry: Registry) -> Plan:
        import anthropic

        unavailable = {n: registry.available(n)[1] for n in registry.names() if not registry.available(n)[0]}
        user = {
            "query": query,
            "input_configuration": report.config.value,
            "images": [r.summary() for r in report.rasters],
            "validation_warnings": report.warnings,
            "unavailable_tools": unavailable,
        }
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self._schema}},
                system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": json.dumps(user)}],
            )
        except anthropic.AuthenticationError as e:
            raise PlannerError(f"authentication failed ({e.status_code})") from e
        except anthropic.RateLimitError as e:
            raise PlannerError("rate limited") from e
        except anthropic.APIStatusError as e:
            raise PlannerError(f"API error {e.status_code}") from e
        except anthropic.APIConnectionError as e:
            raise PlannerError("cannot reach the Claude API") from e
        if resp.stop_reason == "refusal":
            raise PlannerError("planner request was declined")
        if resp.stop_reason == "max_tokens":
            raise PlannerError("planner output truncated")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if not text:
            raise PlannerError("empty planner response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise PlannerError("planner returned invalid JSON") from e
        try:
            task = TaskType(data["task"])
        except (KeyError, ValueError):
            task = ALLOWED_TASKS[report.config][0]
        steps = [PlanStep(s["tool"], {p["name"]: p["value"] for p in s.get("params", [])}, s.get("purpose", ""))
                 for s in data.get("steps", [])]
        return Plan(task=task, intent=data.get("intent", "qa"), steps=steps, planner=self.name,
                    rationale=str(data.get("rationale", ""))[:500])


def has_anthropic_credentials() -> bool:
    import os
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    from pathlib import Path
    return (Path.home() / ".config" / "anthropic").exists()
