"""EuroSAT-adapted scene classifier: availability, planning, answers. Skips when the checkpoint is not trained."""
import pytest

from satquery.agent.controller import SatQueryAgent
from satquery.agent.planner import RulePlanner
from satquery.imagery.validate import validate_inputs
from satquery.tools.base import ToolContext
from satquery.tools.registry import default_registry
from satquery.tools.scene import SceneClassifier, match_classes
from satquery.tools.vlm import RSVLM

tool = SceneClassifier()
needs_ckpt = pytest.mark.skipif(not tool.available()[0], reason="run training/train_scene_classifier.py first")


def test_query_matching():
    assert match_classes("Is there a highway near the river?")[0] == ("Highway", True)
    assert ("River", False) in match_classes("Is there a highway near the river?")
    assert match_classes("any vineyards?") == [("PermanentCrop", True)]


@needs_ckpt
def test_land_use_question_answered(s2):
    rep = validate_inputs([s2])
    res = tool.run(ToolContext("What kind of area is this?", rep.config, rep.rasters), {**tool.defaults(), "mode": "qa"})
    assert res.ok and res.facts["question_type"] == "land_use" and res.facts["tiles"] >= 1
    assert sum(res.facts["scene_distribution_pct"].values()) <= 100.5


@needs_ckpt
def test_sar_only_is_unsupported(s1):
    rep = validate_inputs([s1])
    assert tool.run(ToolContext("What kind of area is this?", rep.config, rep.rasters), tool.defaults()).status == "unsupported"


@needs_ckpt
def test_agent_routes_land_use_question_to_classifier(s2):
    agent = SatQueryAgent(default_registry(RSVLM(backend_factory=lambda: None)), planner=RulePlanner())
    r = agent.run("What kind of area is this?", [s2])
    assert "rs_scene_classifier" in [s["tool"] for s in r.trace["steps"]]
    assert r.trace["result"]["primary_tool"] == "rs_scene_classifier"


@needs_ckpt
def test_agent_routes_presence_question_to_vqa_specialist(s2):
    from satquery.tools.rsvqa import RSVQASpecialist
    if not RSVQASpecialist().available()[0]:
        pytest.skip("run training/train_rsvqa_lr.py first")
    agent = SatQueryAgent(default_registry(RSVLM(backend_factory=lambda: None)), planner=RulePlanner())
    r = agent.run("Is there a road in this image?", [s2])
    assert r.trace["result"]["primary_tool"] == "rsvqa_vqa" and r.short_answer in ("yes", "no")
    r = agent.run("How many buildings are there?", [s2])
    assert r.trace["result"]["primary_tool"] == "rsvqa_vqa" and "between" in r.short_answer or r.short_answer == "0"
