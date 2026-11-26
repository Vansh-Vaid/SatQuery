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
    assert len(r.boxes) == 1 and r.confidence is not None and 0 < r.confidence < 1


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


# ---- dense grounder: checkpoint format and confidence lookup (no trained model needed) ----------------------
def test_measured_confidence_reads_the_bin_the_probability_falls_in():
    from satquery.tools.hrground import CAP, measured_confidence
    table = [{"p_max": 0.1, "acc": 0.05}, {"p_max": 0.4, "acc": 0.3}, {"p_max": 0.9, "acc": 0.7}]
    assert measured_confidence(0.05, table) == 0.05
    assert measured_confidence(0.4, table) == 0.3
    assert measured_confidence(0.95, table) == 0.7          # above the last bin: last bin's rate
    assert measured_confidence(0.9, []) == CAP              # no table: legacy cap


def test_direction_words_flip_with_the_image():
    from satquery.models.hr_ground_net import MAX_TOKENS, flip_words, token_ids
    assert flip_words("The ship at the top left", True, False) == "the ship at the top right"
    assert flip_words("The ship at the top left", False, True) == "the ship at the bottom left"
    ids = token_ids("the ship ship at the top", {"ship": 0, "top": 1, "the_ship": 2})
    assert ids.shape == (MAX_TOKENS,) and list(ids[:3]) == [0, 1, 2] and (ids[3:] == -1).all()


def test_dense_checkpoint_runs_end_to_end(aerial, tmp_path, monkeypatch):
    import numpy as np
    import torch

    from satquery.config import settings
    from satquery.models.hr_ground_net import PROJ_DIM, RAW_DIM, DenseGrounder
    from satquery.tools import hrground

    vocab = {w: i for i, w in enumerate(["large", "vehicle", "green", "area", "closest", "the"])}
    torch.manual_seed(0)
    comp = np.linalg.qr(np.random.default_rng(0).normal(size=(RAW_DIM, PROJ_DIM)))[0].astype(np.float32)
    table = [{"p_max": 0.02, "acc": 0.1}, {"p_max": 1.0, "acc": 0.4}]
    ckpt = tmp_path / "ground.pt"
    torch.save({"arch": "dense", "model": DenseGrounder(len(vocab)).state_dict(), "vocab": vocab,
                "pca_mean": np.zeros(RAW_DIM, np.float32), "pca_comp": comp,
                "pca_scale": np.ones(PROJ_DIM, np.float32), "metrics": {"confidence_table": table}}, ckpt)
    monkeypatch.setattr(hrground, "settings", dataclasses.replace(settings, vrsbench_ground_checkpoint=str(ckpt)))

    tool = HighResGrounder()
    rep = validate_inputs([aerial])
    res = tool.run(ToolContext("Locate the large vehicle closest to the green area", rep.config, rep.rasters),
                   tool.defaults())
    assert res.ok and len(res.boxes) == 1
    b, (h, w) = res.boxes[0], aerial.shape
    assert 0 <= b.x0 < b.x1 <= w and 0 <= b.y0 < b.y1 <= h
    assert res.confidence in (0.1, 0.4) and 0 < res.facts["cell_probability"] <= 1


def test_dense_vqa_checkpoint_runs_end_to_end(aerial, tmp_path, monkeypatch):
    import numpy as np
    import torch

    from satquery.config import settings
    from satquery.models.hr_ground_net import PROJ_DIM, RAW_DIM, DenseVQA
    from satquery.tools import hrvqa

    vocab = {w: i for i, w in enumerate(["how", "many", "vehicles", "are", "visible", "where", "is", "the"])}
    answers = ["2", "3", "top-left", "yes"]
    torch.manual_seed(0)
    comp = np.linalg.qr(np.random.default_rng(0).normal(size=(RAW_DIM, PROJ_DIM)))[0].astype(np.float32)
    ckpt = tmp_path / "vqa.pt"
    torch.save({"arch": "dense", "model": DenseVQA(len(vocab), len(answers)).state_dict(), "vocab": vocab,
                "answers": answers, "pca_mean": np.zeros(RAW_DIM, np.float32), "pca_comp": comp,
                "pca_scale": np.ones(PROJ_DIM, np.float32)}, ckpt)
    monkeypatch.setattr(hrvqa, "settings", dataclasses.replace(settings, vrsbench_checkpoint=str(ckpt)))

    tool = HighResVQA()
    rep = validate_inputs([aerial])
    res = tool.run(ToolContext("How many vehicles are visible?", rep.config, rep.rasters), tool.defaults())
    assert res.ok and res.facts["short_answer"] in answers and 0 < res.confidence <= 1
    desc = tool.run(ToolContext("Describe the image", rep.config, rep.rasters), {**tool.defaults(), "mode": "describe"})
    assert desc.status in ("ok", "unsupported")   # describe mode runs on the dense model too


def test_where_questions_use_the_grounder_box_only_when_it_measures_better(aerial, tmp_path, monkeypatch):
    import json

    import numpy as np
    import torch

    from satquery.config import settings
    from satquery.models.hr_ground_net import PROJ_DIM, RAW_DIM, DenseGrounder, DenseVQA
    from satquery.tools import hrground, hrvqa

    comp = np.linalg.qr(np.random.default_rng(0).normal(size=(RAW_DIM, PROJ_DIM)))[0].astype(np.float32)
    pca = {"pca_mean": np.zeros(RAW_DIM, np.float32), "pca_comp": comp, "pca_scale": np.ones(PROJ_DIM, np.float32)}
    gv = {w: i for i, w in enumerate(["the", "large", "vehicle"])}
    gck = tmp_path / "ground.pt"
    torch.save({"arch": "dense", "model": DenseGrounder(len(gv)).state_dict(), "vocab": gv, **pca,
                "metrics": {"test_acc_at_0.5": 0.38, "confidence_table": [{"p_max": 1.0, "acc": 0.4}]}}, gck)
    monkeypatch.setattr(hrground, "settings", dataclasses.replace(settings, vrsbench_ground_checkpoint=str(gck)))
    qv = {w: i for i, w in enumerate(["where", "is", "the", "large", "vehicle", "located"])}
    answers = ["top-left", "center", "yes"]

    def vqa_with_position_accuracy(acc: float) -> HighResVQA:
        ck = tmp_path / f"vqa_{acc}.pt"
        torch.save({"arch": "dense", "model": DenseVQA(len(qv), len(answers)).state_dict(), "vocab": qv,
                    "answers": answers, **pca}, ck)
        (tmp_path / f"vqa_{acc}.pt.metrics.json").write_text(json.dumps({"test_per_type": {"object position": acc}}))
        monkeypatch.setattr(hrvqa, "settings", dataclasses.replace(settings, vrsbench_checkpoint=str(ck)))
        return HighResVQA()

    rep = validate_inputs([aerial])
    ctx = ToolContext("Where is the large vehicle located?", rep.config, rep.rasters)
    tool = vqa_with_position_accuracy(0.2)                    # grounder 38% > 20%: answer from its box
    res = tool.run(ctx, tool.defaults())
    assert res.facts["answered_by"] == "hr_grounder box" and len(res.boxes) == 1
    assert res.facts["short_answer"] in {f"{r}-{c}" for r in ("top", "middle", "bottom")
                                         for c in ("left", "middle", "right")} | {"center"}
    assert "answered_by" not in tool.run(ctx, {**tool.defaults(), "position_from_box": False}).facts
    tool = vqa_with_position_accuracy(0.5)                    # grounder 38% < 50%: the VQA model answers
    assert "answered_by" not in tool.run(ctx, tool.defaults()).facts


def test_grid_label_uses_the_dataset_place_words():
    from satquery.tools.hrvqa import grid_label
    assert grid_label(0.1, 0.1) == "top-left" and grid_label(0.5, 0.5) == "center"
    assert grid_label(0.9, 0.5) == "middle-right" and grid_label(0.5, 0.9) == "bottom-middle"


# ---- DINOv2 encoder (checkpoints that name it; the others keep ResNet-18) ------------------------------------
def _dino_grounder(tmp_path, monkeypatch, weights=True):
    import numpy as np
    import torch

    from satquery.config import settings
    from satquery.models import dino, hr_ground_net
    from satquery.models.hr_ground_net import PROJ_DIM, RAW_DIM, DenseGrounder
    from satquery.tools import hrground

    torch.manual_seed(0)
    enc = tmp_path / "dinov2.pt"
    if weights:
        torch.save({k: v.half() for k, v in dino.DinoV2S().state_dict().items()}, enc)
    vocab = {w: i for i, w in enumerate(["large", "vehicle", "the"])}
    comp = np.linalg.qr(np.random.default_rng(0).normal(size=(RAW_DIM, PROJ_DIM)))[0].astype(np.float32)
    ckpt = tmp_path / "ground.pt"
    torch.save({"arch": "dense", "encoder": "dinov2_s", "model": DenseGrounder(len(vocab)).state_dict(), "vocab": vocab,
                "pca_mean": np.zeros(RAW_DIM, np.float32), "pca_comp": comp, "pca_scale": np.ones(PROJ_DIM, np.float32),
                "metrics": {"confidence_table": [{"p_max": 1.0, "acc": 0.4}]}}, ckpt)
    new = dataclasses.replace(settings, vrsbench_ground_checkpoint=str(ckpt), dino_checkpoint=str(enc))
    monkeypatch.setattr(hrground, "settings", new)
    monkeypatch.setattr(hr_ground_net, "settings", new)
    hr_ground_net.load_dense_encoder.cache_clear()
    return HighResGrounder()


def test_dino_encoder_gives_the_same_map_shape_as_resnet(tmp_path):
    import numpy as np
    import torch

    from satquery.models.dino import DinoV2S
    from satquery.models.hr_ground_net import GRID, RAW_DIM, encode_dense
    torch.manual_seed(0)
    f = encode_dense(DinoV2S().eval(), [np.zeros((300, 500, 3), np.uint8), np.full((64, 64, 3), 200, np.uint8)])
    assert f.shape == (2, RAW_DIM, GRID, GRID) and f.dtype == np.float32 and np.isfinite(f).all()


def test_dino_checkpoint_runs_and_is_unavailable_without_its_encoder_weights(aerial, tmp_path, monkeypatch):
    rep = validate_inputs([aerial])
    tool = _dino_grounder(tmp_path, monkeypatch)
    assert tool.available()[0]
    res = tool.run(ToolContext("Locate the large vehicle", rep.config, rep.rasters), tool.defaults())
    assert res.ok and len(res.boxes) == 1 and res.confidence == 0.4
    (tmp_path / "dinov2.pt").unlink()
    ok, why = tool.available()
    assert not ok and "dinov2" in why
