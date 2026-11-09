"""Tool contract shared by every specialist model / algorithm in the registry.

A tool declares *what it can do* (tasks, input configurations, modalities) and
*which parameters may be configured* (typed, bounded). The controller may only
set declared parameters; anything else is rejected and logged in the trace.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from satquery.imagery.raster import Raster
from satquery.imagery.validate import InputConfig


class TaskType(str, Enum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    CHANGE_DESCRIPTION = "change_description"
    CROSS_MODAL = "cross_modal_analysis"


TASK_DESCRIPTIONS = {
    TaskType.VQA: "Answer a question about one image.",
    TaskType.CAPTION: "Describe / caption the scene in one image.",
    TaskType.GROUNDING: "Locate (highlight / box) the region referred to in the query in one image.",
    TaskType.CHANGE_VQA: "Answer a question about changes between two dates.",
    TaskType.CHANGE_DESCRIPTION: "Describe what changed between two dates and where.",
    TaskType.CROSS_MODAL: "Jointly analyse a co-registered optical/multispectral + SAR pair.",
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str                      # float | int | bool | enum | str
    default: Any
    description: str
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None

    def coerce(self, value: Any) -> tuple[Any, str | None]:
        """Return (value, note). Out-of-range numbers are clipped; invalid -> default."""
        try:
            if self.type == "float":
                v: Any = float(value)
            elif self.type == "int":
                v = int(round(float(value)))
            elif self.type == "bool":
                v = value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}
                return v, None
            elif self.type == "enum":
                v = str(value).strip().lower()
                if self.choices and v not in self.choices:
                    return self.default, f"'{value}' not in {list(self.choices)}; default '{self.default}' used"
                return v, None
            else:
                return str(value)[:200], None
        except (TypeError, ValueError):
            return self.default, f"'{value}' is not a valid {self.type}; default used"
        note = None
        if self.min is not None and v < self.min:
            v, note = type(v)(self.min), f"clipped to minimum {self.min}"
        if self.max is not None and v > self.max:
            v, note = type(v)(self.max), f"clipped to maximum {self.max}"
        return v, note

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type, "default": self.default,
                             "description": self.description}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.choices:
            d["choices"] = list(self.choices)
        return d


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    tasks: tuple[TaskType, ...]
    input_configs: tuple[InputConfig, ...]
    kind: str                      # "learned (RS-adapted)" | "algorithmic" | "reasoning"
    params: tuple[ParamSpec, ...] = ()
    requires: tuple[str, ...] = ()  # tools whose output this tool consumes (auto-inserted if missing)
    version: str = "1.0"

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description,
                "tasks": [t.value for t in self.tasks], "input_configs": [c.value for c in self.input_configs],
                "kind": self.kind, "params": [p.to_dict() for p in self.params],
                "requires": list(self.requires), "version": self.version}


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    score: float | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"x0": round(self.x0, 1), "y0": round(self.y0, 1), "x1": round(self.x1, 1),
                "y1": round(self.y1, 1), "label": self.label,
                "score": None if self.score is None else round(self.score, 3), "source": self.source}

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        ua = (self.x1 - self.x0) * (self.y1 - self.y0) + (other.x1 - other.x0) * (other.y1 - other.y0) - inter
        return inter / ua if ua > 0 else 0.0


@dataclass
class Layer:
    """A spatial output on the reference image grid."""
    name: str
    kind: str                      # mask | heatmap
    array: np.ndarray
    color: tuple[int, int, int] = (255, 0, 0)
    legend: str = ""
    cmap: str = "magma"
    vmin: float | None = None
    vmax: float | None = None


@dataclass
class ToolResult:
    tool: str
    status: str = "ok"             # ok | unavailable | error | unsupported
    answer: str | None = None      # natural-language contribution
    facts: dict[str, Any] = field(default_factory=dict)  # structured evidence (JSON-serialisable)
    layers: list[Layer] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)   # in-memory payload for downstream tools (not exported)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> dict[str, Any]:
        return {"tool": self.tool, "status": self.status, "answer": self.answer,
                "confidence": None if self.confidence is None else round(float(self.confidence), 3),
                "facts": self.facts, "boxes": [b.to_dict() for b in self.boxes],
                "layers": [l.name for l in self.layers], "notes": self.notes}


@dataclass
class ToolContext:
    query: str
    config: InputConfig
    rasters: list[Raster]
    upstream: dict[str, ToolResult] = field(default_factory=dict)  # keyed by tool name

    @property
    def reference(self) -> Raster:
        return self.rasters[0]


class Tool(abc.ABC):
    spec: ToolSpec

    def available(self) -> tuple[bool, str]:
        return True, "ready"

    @abc.abstractmethod
    def run(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult: ...

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.spec.params}
