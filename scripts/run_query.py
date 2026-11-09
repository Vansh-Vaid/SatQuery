"""Command-line SatQuery: run one query on 1-2 images and save the report.

  python scripts/run_query.py "What changed between these two dates?" demo_data/s2_2024-06-01.tif demo_data/s2_2025-06-05.tif
  python scripts/run_query.py "Is there water?" scene.tif --modality sar --report out/report.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.agent.controller import SatQueryAgent  # noqa: E402
from satquery.imagery.io import load_image  # noqa: E402
from satquery.reports.report import render_html, render_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--modality", nargs="*", default=[], help="per-image modality hints (multispectral|optical_rgb|sar)")
    ap.add_argument("--date", nargs="*", default=[], help="per-image acquisition dates YYYY-MM-DD")
    ap.add_argument("--pair", choices=["cross_modal", "bitemporal"], default=None)
    ap.add_argument("--report", type=Path, default=None, help="write HTML report here (JSON trace alongside)")
    a = ap.parse_args()
    rasters = [load_image(p, a.modality[i] if i < len(a.modality) else None, a.date[i] if i < len(a.date) else None)
               for i, p in enumerate(a.images)]
    r = SatQueryAgent().run(a.query, rasters, a.pair)
    print(f"[{r.status}] task={r.task} confidence={r.confidence_label}"
          f"{f' ({r.confidence:.0%})' if r.confidence is not None else ''}")
    print(r.answer)
    print("\nexecution:", " -> ".join(f"{s['tool']}[{s['status']}]" for s in r.trace.get("steps", [])))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(render_html(r), encoding="utf-8")
        a.report.with_suffix(".json").write_text(render_json(r), encoding="utf-8")
        print(f"report: {a.report}")


if __name__ == "__main__":
    main()
