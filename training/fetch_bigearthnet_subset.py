"""Fetch only the BigEarthNet v2.0 patches a BigEarthNet.txt training subset needs.

The full imagery is 118 GB (Zenodo record 10891137: S2 63 GB + S1 54 GB as .tar.zst). That does not fit
on a Kaggle or Colab disk, and the subset uses only a few percent of the patches. This script:

  1. picks the annotations exactly as prepare_bigearthnet_txt.py will (same function, same seed);
  2. streams each archive once, decompressing on the fly, without writing it to disk;
  3. renders each needed patch straight to <out>/images/<patch_id>_s2.png / _s1.png, with the same
     rendering code the app uses at inference.

It is safe to re-run: finished patches are skipped, and a stream stops as soon as every needed patch
is rendered. Then run prepare_bigearthnet_txt.py --rendered with the same --out, --splits,
--per-type, --bench-per-type and --seed.

  python training/fetch_bigearthnet_subset.py --out data/bentxt --per-type 3000 --bench-per-type 500
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_bigearthnet_txt import (PATCH, S2_BANDS, load_annotations, select_rows,  # noqa: E402
                                     to_s1_raster, to_s2_raster)
from satquery.models.vlm_prompts import render_for_vlm  # noqa: E402

ZENODO = "https://zenodo.org/api/records/10891137/files/{}/content"
ARCHIVES = {"s2": "BigEarthNet-S2.tar.zst", "s1": "BigEarthNet-S1.tar.zst"}


def read_band(raw: bytes) -> np.ndarray:
    from rasterio.io import MemoryFile
    with MemoryFile(raw) as mf, mf.open() as src:
        return src.read(1, out_shape=(PATCH, PATCH)).astype(np.float32)


class Progress:
    """Wraps the HTTP stream to report download progress."""

    def __init__(self, raw, label: str, total: int):
        self.raw, self.label, self.total, self.n, self.t0, self.next = raw, label, total, 0, time.time(), 0

    def read(self, size: int = -1) -> bytes:
        b = self.raw.read(size)
        self.n += len(b)
        if self.n >= self.next:
            rate = self.n / max(time.time() - self.t0, 1e-6) / 1e6
            print(f"  {self.label}: {self.n / 1e9:5.1f} / {self.total / 1e9:.1f} GB  ({rate:.0f} MB/s)", flush=True)
            self.next += 2_000_000_000
        return b


def stream(kind: str, wanted: dict[str, list[str]], img_dir: Path) -> None:
    """Stream one archive and render every patch folder whose name is in ``wanted``.

    ``wanted`` maps a folder name (the S2 patch id, or the S1 name) to the S2 patch ids whose
    images it provides; output files are always named after the S2 patch id."""
    import requests
    import zstandard

    todo = {k: v for k, v in wanted.items() if not all((img_dir / f"{p}_{kind}.png").exists() for p in v)}
    print(f"{kind}: {len(wanted)} patches needed, {len(todo)} still to render", flush=True)
    if not todo:
        return
    bands = S2_BANDS if kind == "s2" else ("VV", "VH")
    with requests.get(ZENODO.format(ARCHIVES[kind]), stream=True, timeout=120) as r:
        r.raise_for_status()
        src = Progress(r.raw, ARCHIVES[kind], int(r.headers.get("content-length", 0)))
        dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
        with dctx.stream_reader(src) as zr, tarfile.open(fileobj=zr, mode="r|") as tar:
            state = {"folder": None, "buf": {}}

            def flush() -> None:
                folder, buf = state["folder"], state["buf"]
                if folder not in todo or not buf:
                    return
                if kind == "s2" and not {"B02", "B03", "B04"} <= set(buf):
                    return
                img = render_for_vlm(to_s2_raster(buf) if kind == "s2" else to_s1_raster(buf))
                for p in todo.pop(folder):
                    img.save(img_dir / f"{p}_{kind}.png")

            for m in tar:
                if not m.isfile() or not m.name.endswith(".tif"):
                    continue
                parent = Path(m.name).parent.name
                if parent != state["folder"]:
                    flush()
                    state["folder"], state["buf"] = parent, {}
                    if not todo:
                        print(f"{kind}: all needed patches rendered; stopping the stream early", flush=True)
                        return
                if parent in todo:
                    band = Path(m.name).stem.rsplit("_", 1)[-1].upper()
                    if band in bands:
                        state["buf"][band] = read_band(tar.extractfile(m).read())
            flush()
    if todo:
        print(f"{kind}: {len(todo)} needed patches were not in the archive", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "bench"])
    ap.add_argument("--per-type", type=int, default=6000)
    ap.add_argument("--bench-per-type", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-s1", action="store_true", help="optical only (skips the 54 GB radar archive)")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    import pandas as pd

    df = load_annotations(args.parquet)
    parts = [select_rows(df, s, args.per_type, args.bench_per_type, args.seed) for s in args.splits]
    sel = pd.concat(parts).drop_duplicates("patch_id")
    print(f"{sum(len(p) for p in parts)} annotations over {len(sel)} distinct patches", flush=True)
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"s2": {p: [p] for p in sel["patch_id"]}}
    if not args.no_s1:
        s1: dict[str, list[str]] = {}
        for p, n in zip(sel["patch_id"], sel["s1_name"]):
            if isinstance(n, str):
                s1.setdefault(n, []).append(p)
        wanted["s1"] = s1
    for kind, w in wanted.items():
        for attempt in range(1, args.retries + 1):
            try:
                stream(kind, w, img_dir)
                break
            except Exception as e:  # dropped connection: re-stream; finished patches are skipped
                print(f"{kind}: stream failed ({type(e).__name__}: {e}); attempt {attempt}/{args.retries}", flush=True)
                if attempt == args.retries:
                    raise
                time.sleep(30)
    print("done. Next: python training/prepare_bigearthnet_txt.py --rendered with the same arguments")


if __name__ == "__main__":
    main()
