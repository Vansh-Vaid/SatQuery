"""Check SatQuery's optical tools on real Sentinel-2 imagery (EuroSAT100: 100 labelled 13-band GeoTIFFs).

Every tile goes through the production path a user upload takes: load_image (GeoTIFF, band layout,
DN -> reflectance) -> to_rgb rendering -> tool. Two checks:
  1. rs_scene_classifier accuracy on raw GeoTIFFs. The model was trained on EuroSAT's pre-rendered JPEGs,
     so this measures the domain gap introduced by our own rendering.
  2. landcover_optical sanity: water fraction on SeaLake/River tiles vs all others, and vegetation
     fraction on Forest tiles.

  python eval/real_sentinel2_check.py --root data/eurosat/ms100 --out eval/results/real_s2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.io import load_image  # noqa: E402
from satquery.imagery.validate import validate_inputs  # noqa: E402
from satquery.tools.base import ToolContext  # noqa: E402
from satquery.tools.landcover import classify_optical  # noqa: E402
from satquery.tools.scene import SceneClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/eurosat/ms100"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/real_s2"))
    args = ap.parse_args()
    files = sorted(args.root.rglob("*.tif"))
    if not files:
        sys.exit(f"no GeoTIFFs under {args.root}")
    scene = SceneClassifier()
    ok, why = scene.available()
    if not ok:
        sys.exit(why)

    rows, per_cls = [], defaultdict(lambda: [0, 0])
    water = {"water_classes": [], "other": []}
    veg_forest = []
    modality_sources: dict[str, int] = defaultdict(int)
    for f in files:
        truth = f.parent.name
        r = load_image(f)
        modality_sources[str(r.meta.get("modality_source"))] += 1
        rep = validate_inputs([r])
        res = scene.run(ToolContext("What kind of area is this?", rep.config, rep.rasters), {**scene.defaults(), "mode": "qa"})
        dist = res.facts.get("scene_distribution_pct", {})
        pred = max(dist, key=dist.get) if dist else None
        correct = pred == truth
        per_cls[truth][0] += int(correct)
        per_cls[truth][1] += 1
        fr = classify_optical(r).fractions()
        (water["water_classes"] if truth in ("SeaLake", "River") else water["other"]).append(fr["water"])
        if truth == "Forest":
            veg_forest.append(fr["vegetation"])
        rows.append({"file": f.name, "truth": truth, "pred": pred, "correct": correct,
                     "top_pct": dist.get(pred) if pred else None, "water_frac": round(fr["water"], 3),
                     "veg_frac": round(fr["vegetation"], 3), "built_frac": round(fr["built_up"], 3)})
    acc = sum(r["correct"] for r in rows) / len(rows)
    metrics = {
        "n": len(rows),
        "scene_classifier_accuracy_on_raw_geotiff": round(acc, 4),
        "per_class_accuracy": {c: {"acc": round(a / n, 3), "n": n} for c, (a, n) in sorted(per_cls.items())},
        "modality_inference": dict(modality_sources),
        "landcover_water_fraction_mean": {
            "SeaLake+River tiles": round(float(np.mean(water["water_classes"])), 3) if water["water_classes"] else None,
            "all other tiles": round(float(np.mean(water["other"])), 3) if water["other"] else None},
        "landcover_vegetation_fraction_mean_on_Forest": round(float(np.mean(veg_forest)), 3) if veg_forest else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.out / "per_tile.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
