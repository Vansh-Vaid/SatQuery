"""High-resolution specialists (VRSBench): resolution gating and agent routing. Skips until trained."""
import dataclasses

import pytest

from satquery.agent.controller import SatQueryAgent
from satquery.agent.planner import RulePlanner
from satquery.imagery.validate import validate_inputs
from satquery.tools.base import ToolContext
from satquery.tools.hrground import HighResGrounder
from satquery.tools.hrvqa import HighResVQA
from satquery.tools.registry import default_registry
from satquery.tools.vlm import RSVLM

vqa, ground = HighResVQA(), HighResGrounder()
needs = pytest.mark.skipif(not (vqa.available()[0] and ground.available()[0]),
                           reason="run training/train_vrsbench_vqa.py and train_vrsbench_grounding.py first")


@pytest.fixture()
def aerial(s2):
    """The synthetic scene as an ungeoreferenced RGB image (scale unknown), like a VRSBench PNG."""
    rgb = s2.data[[s2.bands.index(b) for b in ("B04", "B03", "B02")]]
    return dataclasses.replace(s2, data=rgb, bands=["R", "G", "B"], source_format="png", crs=None, transform=None,
                               bounds=None, name="aerial.png")


def agent():
    return SatQueryAgent(default_registry(RSVLM(backend_factory=lambda: None)), planner=RulePlanner())


@needs
def test_abstains_on_sentinel_scale(s2):
    rep = validate_inputs([s2])
    ctx = ToolContext("How many ships are visible in the image?", rep.config, rep.rasters)
    assert vqa.run(ctx, vqa.defaults()).status == "unsupported"          # 10 m per pixel: not its domain
    assert ground.run(ctx, ground.defaults()).status == "unsupported"


@needs
def test_high_res_question_routed_to_vrsbench_specialist(aerial):
    r = agent().run("How many ships are visible in the image?", [aerial])
    assert r.trace["result"]["primary_tool"] == "hr_vqa" and r.short_answer


@needs
def test_named_object_beats_land_cover_word_in_grounding(aerial):
    r = agent().run("Locate the large vehicle closest to the green area", [aerial])
    assert r.trace["result"]["primary_tool"] == "hr_grounder"
    assert len(r.boxes) == 1 and r.confidence is not None and r.confidence <= 0.35 + 1e-6


@needs
def test_high_res_caption_uses_describe_mode(aerial):
    r = agent().run("Describe this image in detail.", [aerial])
    assert r.trace["result"]["primary_tool"] in ("hr_vqa", "landcover_reasoner")   # hr abstains if unsure


@needs
def test_sentinel_scene_still_uses_rsvqa(s2):
    from satquery.tools.rsvqa import RSVQASpecialist
    if not RSVQASpecialist().available()[0]:
        pytest.skip("RSVQA specialist not trained")
    r = agent().run("Is there a road in this image?", [s2])
    assert r.trace["result"]["primary_tool"] == "rsvqa_vqa"
