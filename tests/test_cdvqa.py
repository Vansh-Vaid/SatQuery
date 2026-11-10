"""CDVQA change-question specialist: label text, scoring, routing. Model tests skip until it is trained."""
import sys
from pathlib import Path

import pytest

from satquery.agent.controller import SatQueryAgent
from satquery.agent.planner import RulePlanner
from satquery.models.cdvqa_features import label_text
from satquery.tools.cdvqa import CDVQASpecialist
from satquery.tools.registry import default_registry
from satquery.tools.vlm import RSVLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from metrics import vqa_correct  # noqa: E402

needs_ckpt = pytest.mark.skipif(not CDVQASpecialist().available()[0], reason="run training/train_cdvqa.py first")


def test_label_text():
    assert label_text("NVG_surface") == "non-vegetated ground surface"
    assert label_text("0_to_10") == "0-10%"
    assert label_text("0") == "0%"
    assert label_text("yes") == "yes"


def test_cdvqa_scoring_maps_readable_answers_back():
    assert vqa_correct("0-10%", "0_to_10", "change_ratio")
    assert vqa_correct("Non-vegetated ground surface.", "NVG_surface", "change_to_what")
    assert vqa_correct("0%", "0", "change_ratio_types")
    assert not vqa_correct("20-30%", "0_to_10", "change_ratio")
    assert vqa_correct("Yes. (91% model confidence)", "yes", "change_or_not")


@needs_ckpt
def test_change_question_routed_to_specialist(s2, s2_t2):
    agent = SatQueryAgent(default_registry(RSVLM(backend_factory=lambda: None)), planner=RulePlanner())
    r = agent.run("Did the areas of buildings increase?", [s2, s2_t2], "bitemporal")
    assert r.trace["result"]["primary_tool"] == "cdvqa_vqa"
    assert r.short_answer in ("yes", "no")
    # a three-way choice is not a CDVQA question form: the specialist abstains, measured tools answer
    r = agent.run("Has the built-up area increased, decreased, or remained unchanged?", [s2, s2_t2], "bitemporal")
    assert r.trace["result"]["primary_tool"] == "change_qa"
    # a describe request still goes to the measured change tools
    r = agent.run("Describe what changed between these dates.", [s2, s2_t2], "bitemporal")
    assert r.trace["result"]["primary_tool"] == "change_qa"
