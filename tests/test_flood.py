"""Flood specialist: trainer smoke test on a fake Sen1Floods11 folder, and the flood_mapper tool end to end."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _fake_sen1floods11(root: Path, n_train=3, n_valid=1, n_test=2, n_bolivia=1, size=64):
    """Chips with a water stripe: low radar backscatter, high green / low SWIR, label 1."""
    import rasterio
    rng = np.random.default_rng(0)
    for sub in ("S1Hand", "S2Hand", "LabelHand"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "valid": [], "test": [], "bolivia": []}
    k = 0
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test), ("bolivia", n_bolivia)):
        for _ in range(n):
            cid = f"Testland_{k:04d}"
            k += 1
            lab = np.zeros((size, size), np.int16)
            c0 = int(rng.integers(8, size - 24))
            lab[:, c0:c0 + 16] = 1
            lab[:2, :] = -1
            s1 = np.where(lab == 1, -22.0, -8.0)[None].repeat(2, 0).astype(np.float32)
            s1 += rng.normal(0, 1.0, s1.shape).astype(np.float32)
            refl = np.full((13, size, size), 0.25, np.float32)
            refl[2][lab == 1], refl[11][lab == 1], refl[12][lab == 1] = 0.12, 0.02, 0.01   # B03 up, SWIR down
            refl[7][lab == 1] = 0.03                                                       # NIR down
            s2 = np.clip(refl * 10000 + rng.normal(0, 50, refl.shape), 0, 10000).astype(np.uint16)
            for sub, arr in (("S1Hand", s1), ("S2Hand", s2), ("LabelHand", lab[None])):
                with rasterio.open(root / sub / f"{cid}_{sub}.tif", "w", driver="GTiff", height=size, width=size,
                                   count=arr.shape[0], dtype=arr.dtype) as dst:
                    dst.write(arr)
            splits[split].append(cid)
    for split, ids in splits.items():
        (root / f"flood_{split}_data.csv").write_text(
            "\n".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif" for c in ids) + "\n")


def test_trainer_runs_end_to_end_on_a_tiny_fake_dataset(tmp_path, monkeypatch):
    import json
    _fake_sen1floods11(tmp_path / "s1f")
    out = tmp_path / "flood_seg.pt"
    monkeypatch.setattr(sys, "argv", ["train_flood.py", "--root", str(tmp_path / "s1f"), "--out", str(out),
                                      "--epochs", "2", "--crop", "48", "--bs", "2", "--base", "4"])
    sys.path.insert(0, str(ROOT / "training"))
    import train_flood
    train_flood.main()
    m = json.loads(Path(str(out) + ".metrics.json").read_text())
    assert set(m["variants"]) == {"s1", "s2", "rgb"} and m["test_countries"] == ["Testland"]
    for v in m["variants"].values():
        assert 0 <= v["test_iou"] <= 1 and 0 <= v["baseline_iou"] <= 1 and v["n_test_chips"] == 2
        assert 0 <= v["bolivia_iou"] <= 1 and v["n_bolivia_chips"] == 1   # held-out country reported


def _flood_checkpoint(path, beats=True):
    """A checkpoint with untrained weights but the real format (all three variants)."""
    import torch

    from satquery.models.seg_net import IN_CH, build_unet
    torch.manual_seed(0)
    ck = {"arch": "seg_unet", "base": 4, "variants": {
        v: {"state": build_unet(IN_CH[v], 4).state_dict(), "in_ch": IN_CH[v],
            "metrics": {"test_iou": 0.7, "baseline_iou": 0.5, "beats_baseline": beats}} for v in IN_CH}}
    torch.save(ck, path)


def _use(monkeypatch, path):
    import dataclasses

    from satquery.config import settings
    from satquery.tools import flood
    monkeypatch.setattr(flood, "settings", dataclasses.replace(settings, flood_checkpoint=str(path)))
    flood._cache.clear()


def test_flood_mapper_runs_on_each_input_family(tmp_path, monkeypatch, s2, s2_t2, s1):
    import dataclasses

    from satquery.imagery.validate import validate_inputs
    from satquery.tools.base import ToolContext
    from satquery.tools.flood import FloodMapper
    _flood_checkpoint(tmp_path / "f.pt")
    _use(monkeypatch, tmp_path / "f.pt")
    s1_t2 = dataclasses.replace(s1, name="s1_b.tif", acquired=s2_t2.acquired)
    rgb = lambda r, n: dataclasses.replace(r, data=r.data[[r.bands.index(b) for b in ("B04", "B03", "B02")]],  # noqa: E731
                                           bands=["R", "G", "B"], source_format="png", crs=None, transform=None,
                                           bounds=None, name=n)
    tool = FloodMapper()
    for pair, family in (([s2, s2_t2], "s2"), ([s1, s1_t2], "s1"), ([rgb(s2, "a.png"), rgb(s2_t2, "b.png")], "rgb")):
        rep = validate_inputs(pair)
        res = tool.run(ToolContext("Has the area been flooded?", rep.config, rep.rasters), tool.defaults())
        assert res.ok and res.facts["variants"] == [family, family], family
        assert res.facts["short_answer"] in ("yes", "no") and 0 < res.confidence < 1
        assert res.layers[0].array.shape == rep.rasters[0].shape


def test_flood_mapper_steps_aside_when_its_model_lost_to_the_classic_method(tmp_path, monkeypatch, s2, s2_t2):
    from satquery.imagery.validate import validate_inputs
    from satquery.tools.base import ToolContext
    from satquery.tools.flood import FloodMapper
    _flood_checkpoint(tmp_path / "lost.pt", beats=False)
    _use(monkeypatch, tmp_path / "lost.pt")
    rep = validate_inputs([s2, s2_t2])
    res = FloodMapper().run(ToolContext("Was it flooded?", rep.config, rep.rasters), FloodMapper().defaults())
    assert res.status == "unsupported"


def test_agent_routes_flood_questions_to_the_flood_model_when_trained(tmp_path, monkeypatch, s2, s2_t2):
    from satquery.agent.controller import SatQueryAgent
    from satquery.agent.planner import RulePlanner
    from satquery.tools.registry import default_registry
    from satquery.tools.vlm import RSVLM
    agent = SatQueryAgent(default_registry(RSVLM(backend_factory=lambda: None)), planner=RulePlanner())
    _use(monkeypatch, tmp_path / "missing.pt")          # no checkpoint: measured change tools answer
    assert agent.run("Has the area been flooded?", [s2, s2_t2], "bitemporal").trace["result"]["primary_tool"] == "change_qa"
    _flood_checkpoint(tmp_path / "f.pt")
    _use(monkeypatch, tmp_path / "f.pt")
    r = agent.run("Has the area been flooded?", [s2, s2_t2], "bitemporal")
    assert r.trace["result"]["primary_tool"] == "flood_mapper" and r.short_answer in ("yes", "no")
    r = agent.run("What changed between these dates?", [s2, s2_t2], "bitemporal")   # not a flood question
    assert r.trace["result"]["primary_tool"] != "flood_mapper"
