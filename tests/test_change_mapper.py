"""change_mapper: any change question read as land-cover class changes from one trained model (no event models)."""
import dataclasses

import pytest

from satquery.imagery.validate import validate_inputs
from satquery.tools.base import ToolContext
from satquery.tools.events import match_classes, match_event


def _ckpt(path, variants, n_out, classes, beats=True):
    import torch

    from satquery.models.seg_net import IN_CH, build_unet
    torch.manual_seed(0)
    torch.save({"arch": "seg_unet", "base": 4, "variants": {
        v: {"state": build_unet(IN_CH[v], 4, n_out).state_dict(), "in_ch": IN_CH[v], "n_out": n_out,
            "classes": classes, "metrics": {"test_miou": 0.6, "beats_baseline": beats}} for v in variants}}, path)


@pytest.fixture()
def models(tmp_path, monkeypatch):
    from satquery.config import settings
    from satquery.models.seg_net import DW_CLASSES
    from satquery.tools import change_mapper

    def use(beats=True, landcover=True, radar=True):
        lc, fl = tmp_path / f"lc_{beats}.pt", tmp_path / f"fl_{beats}.pt"
        if landcover:
            _ckpt(lc, ["s2", "rgb"], len(DW_CLASSES), DW_CLASSES, beats)
        if radar:
            _ckpt(fl, ["s1"], 1, None, beats)
        monkeypatch.setattr(change_mapper, "settings", dataclasses.replace(
            settings, landcover_checkpoint=str(lc if landcover else tmp_path / "none.pt"),
            flood_checkpoint=str(fl if radar else tmp_path / "none.pt")))
        change_mapper._cache.clear()
    return use


def _rgb(r, name):
    return dataclasses.replace(r, data=r.data[[r.bands.index(b) for b in ("B04", "B03", "B02")]], bands=["R", "G", "B"],
                               source_format="png", crs=None, transform=None, bounds=None, name=name)


def test_events_are_vocabulary_not_models():
    assert match_event("Has the area been flooded?").key == "flood"
    assert match_event("Was there deforestation here?").key == "deforestation"
    assert match_event("Did a wildfire burn this valley?").key == "fire"
    assert match_event("How much urban sprawl happened?").key == "urban"
    assert match_event("What changed?") is None
    assert match_classes("Did the forest shrink near the river?") == ["water", "trees"]


@pytest.mark.parametrize("q,qtype", [("Has the area been flooded?", "event"), ("Was there deforestation?", "event"),
                                     ("Did the trees decrease?", "class"), ("What changed?", "describe")])
def test_any_question_on_multispectral_and_true_colour_pairs(models, s2, s2_t2, q, qtype):
    from satquery.models.seg_net import DW_CLASSES
    from satquery.tools.change_mapper import ChangeMapper
    models()
    tool = ChangeMapper()
    for pair in ([s2, s2_t2], [_rgb(s2, "a.png"), _rgb(s2_t2, "b.png")]):
        rep = validate_inputs(pair)
        res = tool.run(ToolContext(q, rep.config, rep.rasters), tool.defaults())
        assert res.ok and res.facts["question_type"] == qtype and list(res.facts["per_class"]) == DW_CLASSES
        assert 0 < res.confidence < 1 and res.layers[0].array.shape == rep.rasters[0].shape


def test_radar_pairs_answer_water_events_with_the_radar_water_model(models, s1, s2_t2):
    from satquery.tools.change_mapper import ChangeMapper
    models()
    s1b = dataclasses.replace(s1, name="s1_b.tif", acquired=s2_t2.acquired)
    rep = validate_inputs([s1, s1b])
    res = ChangeMapper().run(ToolContext("Has the area been flooded?", rep.config, rep.rasters), ChangeMapper().defaults())
    assert res.ok and res.facts["classes"] == ["water", "other"] and res.facts["question_type"] == "event"


def test_steps_aside_when_its_model_lost_to_the_rule_based_method(models, s2, s2_t2):
    from satquery.tools.change_mapper import ChangeMapper
    models(beats=False)
    rep = validate_inputs([s2, s2_t2])
    assert ChangeMapper().run(ToolContext("What changed?", rep.config, rep.rasters), {}).status == "unsupported"


def test_agent_uses_the_trained_model_for_every_change_question_once_available(models, s2, s2_t2):
    from satquery.agent.controller import SatQueryAgent
    from satquery.agent.planner import RulePlanner
    from satquery.tools.registry import default_registry
    from satquery.tools.vlm import RSVLM
    agent = SatQueryAgent(default_registry(RSVLM(backend_factory=lambda: None)), planner=RulePlanner())
    models(landcover=False, radar=False)                      # nothing trained: measured change tools answer
    assert agent.run("Has the area been flooded?", [s2, s2_t2], "bitemporal").trace["result"]["primary_tool"] == "change_qa"
    models()
    for q in ("Has the area been flooded?", "Was there deforestation?", "What changed between these dates?"):
        assert agent.run(q, [s2, s2_t2], "bitemporal").trace["result"]["primary_tool"] == "change_mapper", q


def test_landcover_trainer_runs_end_to_end_on_fake_tiles(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    import numpy as np
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("test", 2)):
        (tmp_path / "dw" / split).mkdir(parents=True)
        for k in range(n):
            y = np.ones((48, 48), np.uint8)              # water on the left, trees and built-up on the right
            y[:, 16:32], y[:, 32:] = 2, 7
            y[:2] = 10                                     # cloud: ignored
            x = np.full((6, 48, 48), 0.2, np.float32)
            x[:, y == 1] = [0.05, 0.08, 0.04, 0.02, 0.01, 0.01][0]
            x += rng.normal(0, 0.01, x.shape).astype(np.float32)
            np.savez_compressed(tmp_path / "dw" / split / f"t{k}.npz", x=x.astype(np.float16), y=y)
    out = tmp_path / "lc.pt"
    monkeypatch.setattr(sys, "argv", ["train_landcover.py", "--data", str(tmp_path / "dw"), "--out", str(out),
                                      "--epochs", "1", "--crop", "32", "--bs", "2", "--base", "4"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    import train_landcover
    train_landcover.main()
    m = json.loads(Path(str(out) + ".metrics.json").read_text())
    for v in ("s2", "rgb"):
        r = m["variants"][v]
        assert 0 <= r["test_miou"] <= 1 and 0 <= r["baseline_acc_4class"] <= 1 and "beats_baseline" in r


def test_landcover_gate_also_needs_better_flood_water_than_the_app():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from train_landcover import passes
    assert passes(0.88, 0.66, {"test": (0.85, 0.83), "bolivia": (0.80, 0.79)})
    assert not passes(0.88, 0.66, {"test": (0.60, 0.83), "bolivia": (0.80, 0.79)})   # the first full run, test split
    assert not passes(0.88, 0.66, {"test": (0.85, 0.83), "bolivia": (0.63, 0.79)})
    assert not passes(0.60, 0.66, {})
    assert passes(0.88, 0.66, {})                                                    # flood scenes not given
