"""Build land-cover training tiles: Dynamic World expert labels + the matching Sentinel-2 scene, on one grid.

Labels (CC-BY 4.0, Tait et al. 2021, PANGAEA 933475) come without imagery; each tile's date, location and
Sentinel-2 grid are in the metadata table. The matching Level-2A scene is fetched from Microsoft Planetary
Computer and warped onto the label tile's exact 510 x 510, 10 m grid. Output: one .npz per tile in --out/<split>/
with x = (6, H, W) float16 reflectance for B02 B03 B04 B08 B11 B12 and y = (H, W) uint8 Dynamic World labels
(0 no data, 1..9 classes, 10 cloud). Splits: "train" = expert-labelled training tiles (stratified by biome),
"test" = the held-out validation set (three experts per tile).

  pip install pystac-client planetary-computer rasterio openpyxl
  python training/fetch_dynamic_world.py --out data/dynamic_world --train-tiles 1500
"""
from __future__ import annotations

import argparse
import io
import random
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PANGAEA = "https://download.pangaea.de/dataset/933475/files/"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def download(name: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    if not path.exists():
        req = urllib.request.Request(PANGAEA + name, headers={"User-Agent": "SatQuery-trainer/1.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=600).read())
    return path


def metadata(xlsx: Path) -> list[dict]:
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    rows = ws.iter_rows(values_only=True)
    head = [str(h) for h in next(rows)]
    return [dict(zip(head, r)) for r in rows if r and r[0]]


def label_index(zpath: Path) -> dict[str, str]:
    """dw_id -> member path of its label GeoTIFF (composites preferred in the validation set)."""
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if not m.lower().endswith((".tif", ".tiff")):
                continue
            mid = re.search(r"(dw_[-0-9.]+_[-0-9.]+-\d{8})", m)
            if mid and (mid.group(1) not in out or "composite" in m.lower()):
                out[mid.group(1)] = m
    return out


def scene_on_grid(item, crs, transform, shape) -> np.ndarray:
    """The item's six bands warped onto the label grid, as reflectance (DN / 10000)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    out = []
    for b in BANDS:
        with rasterio.open(item.assets[b].href) as src, WarpedVRT(src, crs=crs, transform=transform, width=shape[1],
                                                                 height=shape[0], resampling=Resampling.bilinear) as v:
            out.append(v.read(1).astype(np.float32) / 10000.0)
    return np.clip(np.stack(out), 0, 1.5)


def fetch_tile(row: dict, zpath: Path, member: str, out: Path, catalog) -> str:
    import rasterio
    dest = out / f"{row['dw_id']}.npz"
    if dest.exists():
        return "cached"
    with zipfile.ZipFile(zpath) as z, rasterio.open(io.BytesIO(z.read(member))) as lab:
        y, crs, transform = lab.read(1).astype(np.uint8), lab.crs, lab.transform
    date = str(row["date"])[:10]
    grid = str(row.get("S2_GRID") or "").lstrip("T")
    query = {"s2:mgrs_tile": {"eq": grid}} if grid else None
    items = list(catalog.search(collections=["sentinel-2-l2a"], datetime=f"{date}/{date}", query=query,
                                intersects={"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]}).items())
    if not items:
        return "no scene"
    x = scene_on_grid(items[0], crs, transform, y.shape)
    np.savez_compressed(dest, x=x.astype(np.float16), y=y)
    return "ok"


def main() -> None:
    import planetary_computer
    import pystac_client

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/dynamic_world"))
    ap.add_argument("--train-tiles", type=int, default=1500)
    ap.add_argument("--max-cloud", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    raw = args.out / "raw"
    meta = metadata(download("v1_dw_tile_metadata_for_public_release.xlsx", raw))
    experts = download("Experts_tiles.zip", raw)
    valid = download("validation_set_tiles.zip", raw)
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier=planetary_computer.sign_inplace)
    rng = random.Random(0)
    by_id = {r["dw_id"]: r for r in meta}
    for split, zpath in (("train", experts), ("test", valid)):
        index = label_index(zpath)
        rows = [by_id[i] for i in index if i in by_id and float(by_id[i].get("Clouds") or 0) <= args.max_cloud]
        if split == "train":   # stratify across biomes so every kind of landscape is represented
            biomes: dict = {}
            for r in rows:
                biomes.setdefault(r.get("biome"), []).append(r)
            per = max(1, args.train_tiles // max(len(biomes), 1))
            rows = [r for b in biomes.values() for r in rng.sample(b, min(per, len(b)))]
        out = args.out / split
        out.mkdir(parents=True, exist_ok=True)

        def job(r, zpath=zpath, index=index, out=out):
            try:
                return fetch_tile(r, zpath, index[r["dw_id"]], out, catalog)
            except Exception as e:   # one bad scene must not stop the run
                return f"error: {type(e).__name__}"
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(job, rows))
        print(split, {k: results.count(k) for k in sorted(set(results))}, flush=True)
    sys.exit(0 if any((args.out / "train").glob("*.npz")) else "no training tiles were built")


if __name__ == "__main__":
    main()
