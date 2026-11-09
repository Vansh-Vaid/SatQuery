"""The predefined tool registry: the only tools the controller may select."""
from __future__ import annotations

from typing import Any

from satquery.imagery.validate import InputConfig
from satquery.tools.algorithmic import (LandcoverOptical, LandcoverReasoner, LandcoverSAR, OpticalSARFusion,
                                        RegionGrounder)
from satquery.tools.base import TASK_DESCRIPTIONS, TaskType, Tool
from satquery.tools.change import ChangeDetector, ChangeQA, ChangeSemantics
from satquery.tools.rsvqa import RSVQASpecialist
from satquery.tools.scene import SceneClassifier
from satquery.tools.vlm import RSVLM


class Registry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.spec.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def available(self, name: str) -> tuple[bool, str]:
        return self._tools[name].available() if name in self._tools else (False, "unknown tool")

    def for_config(self, config: InputConfig) -> list[Tool]:
        return [t for t in self._tools.values() if config in t.spec.input_configs]

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for t in self._tools.values():
            ok, why = t.available()
            out.append({**t.spec.to_dict(), "available": ok, "status": why})
        return out

    @staticmethod
    def tasks() -> dict[str, str]:
        return {t.value: d for t, d in TASK_DESCRIPTIONS.items()}


def default_registry(vlm: RSVLM | None = None) -> Registry:
    return Registry([
        vlm or RSVLM(),
        RSVQASpecialist(),
        SceneClassifier(),
        LandcoverOptical(),
        LandcoverSAR(),
        OpticalSARFusion(),
        RegionGrounder(),
        LandcoverReasoner(),
        ChangeDetector(),
        ChangeSemantics(),
        ChangeQA(),
    ])


__all__ = ["Registry", "default_registry", "TaskType"]
