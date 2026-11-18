"""Radar water model trainer (Sen1Floods11): smoke test on a fake dataset folder."""
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
