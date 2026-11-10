"""Test optical + SAR analysis on real Sentinel-2 / Sentinel-1 pairs, scored against ESA WorldCover.

For each site the script downloads, from Microsoft Planetary Computer (free, no account):
  * a Sentinel-2 L2A scene (optical, 10 m; B02 B03 B04 B08 B11 B12, plus the SCL cloud layer),
  * the Sentinel-1 RTC scene (radar, VV + VH) closest in date to it,
  * the ESA WorldCover 2021 10 m land-cover map as the answer key,
all resampled onto one 10 m grid. The pair is saved as two ordinary GeoTIFFs and then goes through the
same path a user upload takes (load_image -> validate_inputs -> tools). It scores per pixel:

  optical only   landcover_optical
  SAR only       landcover_sar
  fused          optical_sar_fusion (what the app uses for an optical + radar question)

and runs the full agent on one cross-modal question per site.

WorldCover classes are mapped to SatQuery's four classes: water (80), vegetation (10 tree, 20 shrub,
30 grass, 40 crop, 90 wetland, 95 mangrove, 100 moss), built-up (50), bare soil (60). Snow (70) is ignored.

  python eval/real_pair_test.py                    # all sites
  python eval/real_pair_test.py --sites rotterdam  # one site
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.io import load_image  # noqa: E402
from satquery.imagery.validate import validate_inputs  # noqa: E402
from satquery.tools.algorithmic import OpticalSARFusion  # noqa: E402
from satquery.tools.base import ToolContext  # noqa: E402
from satquery.tools.landcover import classify_optical, classify_sar  # noqa: E402
from satquery.tools.vocab import CLASSES  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{}"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# name: (lon, lat, date window, max cloud %, description)
# Windows avoid each region's rainy season, except dhaka_cloudy, which is picked *for* its clouds.
SITES = {
    "rotterdam": (4.40, 51.90, ("2023-05-01", "2023-09-30"), 10, "port, river Maas, city, farmland"),
    "geneva": (6.15, 46.21, ("2023-06-01", "2023-09-30"), 10, "lake, city, forest, fields"),
    "cairo": (31.23, 30.05, ("2023-03-01", "2023-06-30"), 5, "Nile, dense city, desert, irrigated fields"),
    "hyderabad": (78.47, 17.42, ("2023-01-15", "2023-04-30"), 10, "lakes, dense city, scrub"),
    "sacramento": (-121.50, 38.58, ("2023-06-01", "2023-09-30"), 5, "rivers, city, farmland, dry grass"),
    "dhaka_cloudy": (90.40, 23.78, ("2023-06-15", "2023-09-30"), -1, "monsoon clouds over city and rivers"),
}
HALF_M = 3000          # 6 km x 6 km = 600 x 600 pixels at 10 m
RES = 10.0
QUERY = "Use the optical and SAR images together to identify built-up and water-covered regions."

WC_MAP = {80: 0, 10: 1, 20: 1, 30: 1, 40: 1, 90: 1, 95: 1, 100: 1, 50: 2, 60: 3}  # -> CLASSES index


# ------------------------------------------------------------------ download
_tokens: dict[str, str] = {}


def sign(href: str, collection: str) -> str:
    if collection not in _tokens:
        _tokens[collection] = requests.get(SAS.format(collection), timeout=60).json()["token"]
    return f"{href}?{_tokens[collection]}"


def search(collection: str, lon: float, lat: float, start: str, end: str, extra: dict | None = None) -> list[dict]:
    body = {"collections": [collection], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    if extra:
        body.update(extra)
    for attempt in range(3):
        try:
            r = requests.post(STAC, json=body, timeout=120)
            r.raise_for_status()
            return r.json()["features"]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    return []


def item_date(item: dict) -> date:
    return datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()


def pick_s2(lon: float, lat: float, window: tuple[str, str], max_cloud: int) -> dict:
    items = search("sentinel-2-l2a", lon, lat, *window)
    if not items:
        raise RuntimeError("no Sentinel-2 scene in window")
    cc = lambda it: it["properties"].get("eo:cloud_cover", 100.0)  # noqa: E731
    if max_cloud >= 0:   # clearest scene
        return min(items, key=cc)
    # cloudy test: partly cloudy (25-70 %), so there is both clear and hidden ground to compare
    part = [it for it in items if 25 <= cc(it) <= 70]
    return min(part or items, key=lambda it: abs(cc(it) - 45))


def pick_s1(lon: float, lat: float, when: date) -> dict:
    lo, hi = (when - timedelta(days=12)).isoformat(), (when + timedelta(days=12)).isoformat()
    items = search("sentinel-1-rtc", lon, lat, lo, hi)
    items = [it for it in items if {"vv", "vh"} <= set(it["assets"])]
    if not items:
        raise RuntimeError("no Sentinel-1 RTC scene within 12 days")
    return min(items, key=lambda it: abs((item_date(it) - when).days))


def read_on_grid(href: str, crs: str, transform, size: int, resampling, dtype="float32", nodata=None) -> np.ndarray:
    import rasterio
    from rasterio.vrt import WarpedVRT

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(href) as src:
            nd = nodata if nodata is not None else src.nodata
            with WarpedVRT(src, crs=crs, transform=transform, width=size, height=size, resampling=resampling,
                           src_nodata=nd, nodata=nd) as vrt:
                arr = vrt.read(1).astype(dtype)
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr) if np.issubdtype(arr.dtype, np.floating) else arr
    return arr


def fetch_site(name: str, out: Path) -> dict:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_xy

    lon, lat, window, max_cloud, _ = SITES[name]
    d = out / name
    meta_path = d / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    d.mkdir(parents=True, exist_ok=True)
    s2 = pick_s2(lon, lat, window, max_cloud)
    s2_date = item_date(s2)
    s1 = pick_s1(lon, lat, s2_date)
    s1_date = item_date(s1)
    wc = search("esa-worldcover", lon, lat, "2021-01-01", "2021-12-31")
    if not wc:
        raise RuntimeError("no WorldCover tile")
    epsg = s2["properties"]["proj:epsg"]
    crs = f"EPSG:{epsg}"
    xs, ys = warp_xy("EPSG:4326", crs, [lon], [lat])
    x0, y0 = round((xs[0] - HALF_M) / RES) * RES, round((ys[0] + HALF_M) / RES) * RES
    size = int(2 * HALF_M / RES)
    tr = from_origin(x0, y0, RES, RES)
    print(f"  {name}: S2 {s2_date} ({s2['properties'].get('eo:cloud_cover', 0):.0f}% cloud), "
          f"S1 {s1_date}, grid {crs}", flush=True)

    # Sentinel-2: processing baseline >= 04.00 adds +1000 to every L2A value
    baseline = float(s2["properties"].get("s2:processing_baseline", "0") or 0)
    offset = 1000 if baseline >= 4.0 else 0
    bands = []
    for b in S2_BANDS:
        a = read_on_grid(sign(s2["assets"][b]["href"], "sentinel-2-l2a"), crs, tr, size,
                         Resampling.bilinear, nodata=0)
        bands.append(np.clip(np.nan_to_num(a, nan=0) - offset, 1, 10000))
    s2_arr = np.stack(bands).astype(np.uint16)
    scl = read_on_grid(sign(s2["assets"]["SCL"]["href"], "sentinel-2-l2a"), crs, tr, size,
                       Resampling.nearest, dtype="uint8", nodata=0)

    sar = []
    for b in ("vv", "vh"):
        a = read_on_grid(sign(s1["assets"][b]["href"], "sentinel-1-rtc"), crs, tr, size, Resampling.average)
        sar.append(np.where(np.isfinite(a) & (a > 0), a, np.nan))
    s1_arr = np.stack(sar).astype(np.float32)

    wc_arr = read_on_grid(sign(wc[0]["assets"]["map"]["href"], "esa-worldcover"), crs, tr, size,
                          Resampling.mode, dtype="uint8", nodata=0)

    common = dict(driver="GTiff", height=size, width=size, crs=crs, transform=tr, compress="deflate")
    s2_path = d / f"s2_{s2_date.isoformat()}.tif"
    with rasterio.open(s2_path, "w", count=len(S2_BANDS), dtype="uint16", nodata=0, **common) as dst:
        dst.write(s2_arr)
        dst.descriptions = tuple(S2_BANDS)
        dst.update_tags(ACQUISITION_DATE=s2_date.isoformat(), SOURCE=s2["id"])
    s1_path = d / f"s1_{s1_date.isoformat()}.tif"
    with rasterio.open(s1_path, "w", count=2, dtype="float32", nodata=float("nan"), **common) as dst:
        dst.write(s1_arr)
        dst.descriptions = ("VV", "VH")
        dst.update_tags(ACQUISITION_DATE=s1_date.isoformat(), SOURCE=s1["id"])
    np.save(d / "worldcover.npy", wc_arr)
    np.save(d / "scl.npy", scl)
    meta = {"site": name, "lon": lon, "lat": lat, "description": SITES[name][4], "crs": crs,
            "s2": s2["id"], "s2_date": s2_date.isoformat(),
            "s2_cloud_cover_pct": s2["properties"].get("eo:cloud_cover"), "s2_processing_baseline": baseline,
            "s1": s1["id"], "s1_date": s1_date.isoformat(), "days_apart": abs((s1_date - s2_date).days),
            "worldcover": wc[0]["id"], "s2_file": s2_path.name, "s1_file": s1_path.name}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# ------------------------------------------------------------------ scoring
def label_map(lcm) -> np.ndarray:
    lab = np.full(lcm.valid.shape, -1, np.int8)
    for i, c in enumerate(CLASSES):
        lab[lcm.masks[c]] = i
    return lab


def score(pred: np.ndarray, truth: np.ndarray, region: np.ndarray) -> dict:
    """Per-class IoU / F1 and accuracy over ``region``. Unclassified pixels (-1) count as wrong."""
    p, t = pred[region], truth[region]
    n = int(region.sum())
    out = {"pixels": n, "coverage_pct": round(100 * float((p >= 0).mean()), 1) if n else None,
           "accuracy_pct": round(100 * float((p == t).mean()), 1) if n else None, "per_class": {}}
    f1s = []
    for i, c in enumerate(CLASSES):
        tp = int(((p == i) & (t == i)).sum())
        fp = int(((p == i) & (t != i)).sum())
        fn = int(((p != i) & (t == i)).sum())
        share = float((t == i).mean()) if n else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        out["per_class"][c] = {"truth_pct": round(100 * share, 1),
                               "iou": None if iou is None else round(iou, 3),
                               "f1": None if f1 is None else round(f1, 3)}
        if share >= 0.01 and f1 is not None:   # macro-F1 over classes that cover at least 1 % of the site
            f1s.append(f1)
    out["macro_f1"] = round(float(np.mean(f1s)), 3) if f1s else None
    return out


def evaluate(meta: dict, root: Path, agent=None) -> dict:
    d = root / meta["site"]
    opt = load_image(d / meta["s2_file"])
    sar = load_image(d / meta["s1_file"])
    rep = validate_inputs([opt, sar])
    opt_r, sar_r = rep.rasters[0], rep.rasters[1]
    lo, ls = classify_optical(opt_r), classify_sar(sar_r)
    ctx = ToolContext(query=QUERY, config=rep.config, rasters=rep.rasters)
    fused = OpticalSARFusion().run(ctx, {})
    lf = fused.data["lcm"]

    wc = np.load(d / "worldcover.npy")
    scl = np.load(d / "scl.npy")
    truth = np.full(wc.shape, -1, np.int8)
    for k, v in WC_MAP.items():
        truth[wc == k] = v
    known = truth >= 0
    scl_cloud = np.isin(scl, (8, 9, 10))          # ESA's own cloud mask (medium/high prob., cirrus)
    preds = {"optical_only": label_map(lo), "sar_only": label_map(ls), "fused": label_map(lf)}
    res = {"meta": meta, "config": rep.config.value,
           "validation_warnings": rep.warnings,
           "esa_cloud_pct": round(100 * float(scl_cloud.mean()), 1),
           "our_cloud_pct": round(100 * float(lo.cloud.mean()), 1) if lo.cloud is not None else 0.0,
           "all_pixels": {k: score(v, truth, known) for k, v in preds.items()},
           "fusion_facts": {k: fused.facts[k] for k in ("pixel_agreement_pct", "cloud_pixels_filled_by_sar_pct",
                                                        "per_class_cross_sensor")}}
    if scl_cloud.mean() > 0.02:
        res["cloudy_pixels"] = {k: score(v, truth, known & scl_cloud) for k, v in preds.items()}
        res["clear_pixels"] = {k: score(v, truth, known & ~scl_cloud) for k, v in preds.items()}
    if agent is not None:
        t0 = time.perf_counter()
        r = agent.run(QUERY, [load_image(d / meta["s2_file"]), load_image(d / meta["s1_file"])])
        res["agent"] = {"query": QUERY, "status": r.status, "answer": r.answer, "confidence": r.confidence,
                        "confidence_label": r.confidence_label,
                        "tools": [s["tool"] for s in r.trace.get("steps", [])],
                        "seconds": round(time.perf_counter() - t0, 1)}
    res["_maps"] = {"rgb": opt_r, "sar": sar_r, "truth": truth, **preds}
    return res


# ------------------------------------------------------------------ figure
PALETTE = np.array([[30, 100, 220], [60, 160, 60], [220, 50, 40], [210, 180, 120]], np.uint8)


def colour(lab: np.ndarray) -> np.ndarray:
    img = np.full(lab.shape + (3,), 255, np.uint8)
    for i in range(len(CLASSES)):
        img[lab == i] = PALETTE[i]
    return img


def figure(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cols = ["Sentinel-2 (true colour)", "Sentinel-1 (VV, dB)", "WorldCover (answer key)",
            "Optical only", "SAR only", "Fused"]
    fig, axes = plt.subplots(len(results), 6, figsize=(18, 3.1 * len(results)), squeeze=False)
    for row, res in zip(axes, results):
        m = res["_maps"]
        r = m["rgb"]
        rgb = np.stack([r.band("B04"), r.band("B03"), r.band("B02")], -1)
        rgb = np.clip(np.nan_to_num(rgb) / max(np.nanpercentile(rgb, 98), 1e-3), 0, 1)
        vv = np.nan_to_num(m["sar"].band("VV"), nan=-25)
        panels = [rgb, vv, colour(m["truth"]), colour(m["optical_only"]), colour(m["sar_only"]), colour(m["fused"])]
        for j, (ax, img) in enumerate(zip(row, panels)):
            ax.imshow(img, cmap="gray" if j == 1 else None, vmin=-25 if j == 1 else None, vmax=0 if j == 1 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if j >= 3:
                key = ["optical_only", "sar_only", "fused"][j - 3]
                s = res["all_pixels"][key]
                ax.set_xlabel(f"acc {s['accuracy_pct']}%  ·  macro-F1 {s['macro_f1']}", fontsize=9)
        meta = res["meta"]
        row[0].set_ylabel(f"{meta['site']}\nS2 {meta['s2_date']} / S1 {meta['s1_date']}", fontsize=9)
    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=10)
    fig.legend(handles=[Patch(color=PALETTE[i] / 255, label=c.replace("_", " ")) for i, c in enumerate(CLASSES)]
               + [Patch(facecolor="white", edgecolor="grey", label="unclassified / cloud")],
               loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--data", type=Path, default=Path("data/real_pairs"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/real_pairs"))
    ap.add_argument("--figure", type=Path, default=Path("docs/images/real_pairs.jpg"))
    ap.add_argument("--no-agent", action="store_true", help="skip the full-agent run per site")
    args = ap.parse_args()

    agent = None
    if not args.no_agent:
        from satquery.agent.controller import SatQueryAgent
        agent = SatQueryAgent()
    results = []
    for name in args.sites:
        print(f"[{name}]", flush=True)
        try:
            meta = fetch_site(name, args.data)
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}", flush=True)
            continue
        res = evaluate(meta, args.data, agent)
        results.append(res)
        for k in ("optical_only", "sar_only", "fused"):
            s = res["all_pixels"][k]
            print(f"  {k:13s} acc {s['accuracy_pct']:5.1f}%  macro-F1 {s['macro_f1']}  coverage {s['coverage_pct']}%  "
                  + "  ".join(f"{c} IoU {v['iou']}" for c, v in s["per_class"].items()), flush=True)
        if "cloudy_pixels" in res:
            print("  under cloud:  " + "  ".join(f"{k} acc {v['accuracy_pct']}%"
                                                 for k, v in res["cloudy_pixels"].items()), flush=True)
    if not results:
        sys.exit("no site could be processed")

    summary = {}
    for k in ("optical_only", "sar_only", "fused"):
        summary[k] = {
            "mean_accuracy_pct": round(float(np.mean([r["all_pixels"][k]["accuracy_pct"] for r in results])), 1),
            "mean_macro_f1": round(float(np.mean([r["all_pixels"][k]["macro_f1"] for r in results])), 3),
            "mean_iou": {c: round(float(np.mean(v)), 3) for c in CLASSES
                         if (v := [r["all_pixels"][k]["per_class"][c]["iou"] for r in results
                                   if r["all_pixels"][k]["per_class"][c]["truth_pct"] >= 1
                                   and r["all_pixels"][k]["per_class"][c]["iou"] is not None])}}
    args.out.mkdir(parents=True, exist_ok=True)
    figure(results, args.figure)
    for r in results:
        r.pop("_maps")
    (args.out / "results.json").write_text(json.dumps({"summary": summary, "sites": results}, indent=2, default=str))
    print("\nMean over sites:")
    for k, v in summary.items():
        print(f"  {k:13s} acc {v['mean_accuracy_pct']}%  macro-F1 {v['mean_macro_f1']}  IoU {v['mean_iou']}")
    print(f"\nwrote {args.out / 'results.json'} and {args.figure}")


if __name__ == "__main__":
    main()
