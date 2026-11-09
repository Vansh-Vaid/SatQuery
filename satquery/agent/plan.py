"""Execution plans and their validation against the tool registry.

Whatever proposes a plan (Claude or the rule planner), ``validate_plan`` is the
gate: only registered, available, input-compatible tools survive; dependencies
are auto-inserted in order; only declared parameters are kept, coerced and
bounded. Every correction is recorded for the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from satquery.imagery.validate import InputConfig
from satquery.tools.base import TaskType
from satquery.tools.registry import Registry

ALLOWED_TASKS: dict[InputConfig, tuple[TaskType, ...]] = {
    InputConfig.SINGLE: (TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING),
    InputConfig.CROSS_MODAL: (TaskType.CROSS_MODAL,),
    InputConfig.BITEMPORAL: (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION),
}
INTENTS = ("qa", "describe", "locate")


@dataclass
class PlanStep:
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "params": self.params, "purpose": self.purpose}


@dataclass
class Plan:
    task: TaskType
    intent: str
    steps: list[PlanStep]
    planner: str
    rationale: str = ""
    corrections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task.value, "intent": self.intent, "planner": self.planner,
                "rationale": self.rationale, "steps": [s.to_dict() for s in self.steps],
                "corrections": self.corrections}


def validate_plan(plan: Plan, registry: Registry, config: InputConfig) -> Plan:
    fixes = list(plan.corrections)
    allowed = ALLOWED_TASKS[config]
    task = plan.task
    if task not in allowed:
        fixes.append(f"task '{task.value}' is not valid for {config.value} input; using '{allowed[0].value}'")
        task = allowed[0]
    intent = plan.intent if plan.intent in INTENTS else "qa"

    kept: list[PlanStep] = []
    seen: set[str] = set()
    for step in plan.steps:
        name = step.tool
        if name not in registry:
            fixes.append(f"dropped unknown tool '{name}'")
            continue
        tool = registry.get(name)
        if config not in tool.spec.input_configs:
            fixes.append(f"dropped '{name}': does not accept {config.value} input")
            continue
        ok, why = tool.available()
        if not ok:
            fixes.append(f"skipped '{name}': unavailable ({why})")
            continue
        if name in seen:
            fixes.append(f"dropped duplicate '{name}'")
            continue
        params: dict[str, Any] = {}
        for k, v in (step.params or {}).items():
            spec = tool.spec.param(k)
            if spec is None:
                fixes.append(f"{name}: rejected undeclared parameter '{k}'")
                continue
            if v is None or v == "":
                continue
            coerced, note = spec.coerce(v)
            if note:
                fixes.append(f"{name}.{k}: {note}")
            params[k] = coerced
        seen.add(name)
        kept.append(PlanStep(name, params, step.purpose))

    # dependency insertion (requires) - keep producer before consumer
    ordered: list[PlanStep] = []
    placed: set[str] = set()

    def place(step: PlanStep, chain: tuple[str, ...] = ()) -> None:
        if step.tool in placed:
            return
        for dep in registry.get(step.tool).spec.requires:
            if dep in placed or dep in chain:
                continue
            dep_step = next((s for s in kept if s.tool == dep), None)
            if dep_step is None:
                dep_tool = registry.get(dep) if dep in registry else None
                if dep_tool and config in dep_tool.spec.input_configs and dep_tool.available()[0]:
                    fixes.append(f"inserted required tool '{dep}' before '{step.tool}'")
                    dep_step = PlanStep(dep, {}, f"required input for {step.tool}")
                else:
                    continue
            place(dep_step, chain + (step.tool,))
        ordered.append(step)
        placed.add(step.tool)

    for s in kept:
        place(s)
    return Plan(task=task, intent=intent, steps=ordered, planner=plan.planner, rationale=plan.rationale,
                corrections=fixes)
