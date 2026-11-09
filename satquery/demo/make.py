"""Write the bundled demo scenarios (GeoTIFFs + manifest) used by the web UI and tests."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from satquery.demo import synth as S
from satquery.imagery.io import save_geotiff

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "demo_data"

SCENARIOS = [
    {"id": "single_s2", "title": "Single Sentinel-2 scene (12 bands)", "files": ["s2_2024-06-01.tif"],
     "queries": ["Describe the land-cover and major objects visible in this image.",
                 "Highlight the water body referred to in the query.",
                 "Is there more vegetation than built-up area?", "How many water bodies are there?"]},
    {"id": "single_s1", "title": "Single Sentinel-1 SAR scene (VV/VH)", "files": ["s1_2024-06-02.tif"],
     "queries": ["Is there any water in this image?", "Highlight the built-up area.",
                 "Describe this SAR image."]},
    {"id": "cross_modal", "title": "Cross-modal: cloudy Sentinel-2 + Sentinel-1",
     "files": ["s2_cloudy_2024-06-01.tif", "s1_2024-06-02.tif"],
     "queries": ["Use the optical and SAR images together to identify built-up and water-covered regions.",
                 "How much of the scene is water?", "Describe the scene using both sensors."]},
    {"id": "bitemporal_s2", "title": "Bi-temporal Sentinel-2 (2024 → 2025)",
     "files": ["s2_2024-06-01.tif", "s2_2025-06-05.tif"],
     "queries": ["What changed between these two dates, and where did the change occur?",
                 "Has the built-up area increased, decreased, or remained unchanged?",
                 "What did the water turn into?", "How much of the area changed?"]},
    {"id": "bitemporal_s1", "title": "Bi-temporal Sentinel-1 (2024 → 2025)",
     "files": ["s1_2024-06-02.tif", "s1_2025-06-06.tif"],
     "queries": ["What changed between these two dates, and where did the change occur?",
                 "Has the water area decreased?"]},
]


def make_demo_data(out_dir: Path | str = DEFAULT_DIR, size: int = 256) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab = S.base_layout(size, size)
    lab2 = S.evolve(lab)
    d1, d1s, d2, d2s = date(2024, 6, 1), date(2024, 6, 2), date(2025, 6, 5), date(2025, 6, 6)
    rasters = {
        "s2_2024-06-01.tif": S.render_s2(lab, seed=1, acquired=d1),
        "s2_2025-06-05.tif": S.render_s2(lab2, seed=2, gain=1.05, acquired=d2),
        "s2_cloudy_2024-06-01.tif": S.render_s2(lab, seed=3, cloud=S.cloud_mask(size, size), acquired=d1),
        "s1_2024-06-02.tif": S.render_s1(lab, seed=4, acquired=d1s),
        "s1_2025-06-06.tif": S.render_s1(lab2, seed=5, acquired=d2s),
    }
    for name, r in rasters.items():
        save_geotiff(r, out / name)
    (out / "manifest.json").write_text(json.dumps({"note": "Synthetic Sentinel-1/2 scenes with known ground truth "
                                                           "(satquery/demo/synth.py).",
                                                   "scenarios": SCENARIOS}, indent=2), encoding="utf-8")
    return out


def load_manifest(out_dir: Path | str = DEFAULT_DIR) -> dict:
    out = Path(out_dir)
    if not (out / "manifest.json").exists():
        make_demo_data(out)
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(f"demo data written to {make_demo_data()}")
