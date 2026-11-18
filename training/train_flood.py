"""Train SatQuery's water segmenters (flood_mapper) on Sen1Floods11's hand-labelled flood events.

Three variants of satquery.models.seg_net (radar "s1", multispectral "s2", true colour "rgb") are trained on the
same 446 chips from 11 countries, selected on the official validation split and scored on the official test
split with the dataset's usual metric (IoU of water over all labelled pixels). Each variant is also compared
with the classic method SatQuery already has, on the same test pixels:
  s1  - Otsu threshold on VV backscatter        s2 - MNDWI > 0        rgb - SatQuery's colour-based land cover
Images are read with SatQuery's own loader and turned into model inputs by seg_net.model_input, so training and
the app see identical numbers.

  python training/train_flood.py --root data/sen1floods11 --out checkpoints/flood_seg.pt          (GPU or CPU)
Expected layout under --root (searched recursively): *_S1Hand.tif, *_S2Hand.tif, *_LabelHand.tif and the split
lists flood_train_data.csv, flood_valid_data.csv, flood_test_data.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.io import load_image  # noqa: E402
from satquery.imagery.raster import Modality, Raster  # noqa: E402
from satquery.models.seg_net import IN_CH, S2_BANDS, build_unet, model_input, predict_water  # noqa: E402


def split_ids(root: Path, split: str, required: bool = True) -> list[str]:
    path = next(root.rglob(f"flood_{split}_data.csv"), None)
    if path is None:
        if not required:
            return []
        raise SystemExit(f"flood_{split}_data.csv not found under {root}")
    ids = []
    for line in path.read_text().splitlines():
        for tok in line.replace(";", ",").split(","):
            tok = tok.strip()
            if "_S1Hand" in tok:
                ids.append(tok.split("_S1Hand")[0])
    return ids


def load_chip(files: dict[str, Path], cid: str) -> tuple[Raster, Raster, np.ndarray] | None:
    import rasterio
    try:
        s1 = load_image(files[f"{cid}_S1Hand.tif"], "sar")
        s2 = load_image(files[f"{cid}_S2Hand.tif"], "s2")
        with rasterio.open(files[f"{cid}_LabelHand.tif"]) as src:
            y = src.read(1).astype(np.int8)
    except KeyError:
        return None
    return s1, s2, y


def rgb_raster(s2: Raster) -> Raster:
    """The S2 chip's true colour only, the way an RGB GeoTIFF would arrive."""
    data = np.stack([s2.band(b) for b in ("B04", "B03", "B02")])
    return Raster(data=data, bands=["B04", "B03", "B02"], modality=Modality.OPTICAL_RGB, source_format="geotiff",
                  name=s2.name, valid_mask=s2.valid_mask)


def inputs_for(variant: str, s1: Raster, s2: Raster) -> np.ndarray:
    return model_input(s1 if variant == "s1" else (s2 if variant == "s2" else rgb_raster(s2)), variant)


def baseline(variant: str, s1: Raster, s2: Raster) -> np.ndarray:
    """The classic water map SatQuery already produces for this input."""
    from satquery.tools.landcover import classify, otsu
    from satquery.tools.vocab import CLASSES
    if variant == "s1":
        vv = s1.band("VV") if s1.band("VV") is not None else s1.data[0]
        fin = vv[np.isfinite(vv)]
        thr = otsu(fin, -26.0, -8.0) if fin.size else None     # same window as SatQuery's landcover_sar
        return np.nan_to_num(vv, nan=0.0) < (thr if thr is not None else -18.0)
    if variant == "s2":
        g, sw = s2.band("B03"), s2.band("B11")
        return ((g - sw) / np.maximum(g + sw, 1e-6)) > 0
    return classify(rgb_raster(s2)).label_map() == CLASSES.index("water")


def iou(pred: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    m = y >= 0
    p, t = pred & m, (y == 1) & m
    return int((p & t).sum()), int((p | t).sum())


def augment(x, y, variant, rng):
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if variant == "rgb":   # screenshots and photos: different stretches, gamma and contrast
        x = np.clip(x, 0, 1) ** rng.uniform(0.6, 1.6)
        x = np.clip((x - 0.5) * rng.uniform(0.75, 1.3) + 0.5 + rng.uniform(-0.12, 0.12), 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y)


def main() -> None:
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/sen1floods11"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/flood_seg.pt"))
    ap.add_argument("--variants", nargs="+", default=["s1", "s2", "rgb"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    files = {p.name: p for p in args.root.rglob("*.tif")}
    chips = {}
    for split in ("train", "valid", "test", "bolivia"):   # Bolivia: a country kept out of training entirely
        loaded = [(cid, load_chip(files, cid)) for cid in split_ids(args.root, split, required=split != "bolivia")]
        chips[split] = [(cid, c) for cid, c in loaded if c is not None]
        print(f"{split}: {len(chips[split])} chips ({len(loaded) - len(chips[split])} missing files)", flush=True)
    if not chips["train"] or not chips["test"]:
        raise SystemExit("no training or test chips found; check --root")
    countries = sorted({cid.split("_")[0] for cid, _ in chips["test"]})

    result = {"dataset": "Sen1Floods11 hand-labelled flood events (official splits)", "test_countries": countries,
              "variants": {}}
    ckpt = {"arch": "seg_unet", "base": args.base, "variants": {}}
    for v in args.variants:
        data = {s: [(inputs_for(v, s1, s2), y) for _, (s1, s2, y) in chips[s]] for s in chips}
        model = build_unet(IN_CH[v], args.base).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * max(1, math.ceil(len(data["train"]) / args.bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

        def evaluate(split):
            model.eval()
            cpu_model = model.to("cpu")
            inter = union = 0
            for x, y in data[split]:
                a, b = iou(predict_water(cpu_model, x) > 0.5, y)
                inter, union = inter + a, union + b
            model.to(dev).train()
            return inter / max(union, 1)

        best, best_state = -1.0, None
        for ep in range(args.epochs):
            te = time.time()
            order = rng.permutation(len(data["train"]))
            for k in range(0, len(order), args.bs):
                xb, yb = [], []
                for i in order[k:k + args.bs]:
                    x, y = data["train"][i]
                    h, w = y.shape
                    c = min(args.crop, h, w)
                    cy, cx = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
                    x, y = augment(x[:, cy:cy + c, cx:cx + c], y[cy:cy + c, cx:cx + c], v, rng)
                    xb.append(x)
                    yb.append(y)
                x = torch.from_numpy(np.stack(xb)).to(dev)
                y = torch.from_numpy(np.stack(yb).astype(np.float32)).to(dev)
                m = (y >= 0).float()
                t = y.clamp(min=0)
                logit = model(x)
                bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * m).sum() / m.sum().clamp(min=1)
                p = torch.sigmoid(logit) * m
                dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
                loss = bce + dice
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            val = evaluate("valid") if data["valid"] else evaluate("train")
            if val > best:
                best, best_state = val, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            print(f"[{v}] epoch {ep + 1}/{args.epochs}: val IoU {val:.4f} ({time.time() - te:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        test = evaluate("test")
        by_country: dict[str, list[int]] = {}       # test IoU per country (India included)
        cpu_model = model.to("cpu")
        for (cid, _), (x, y) in zip(chips["test"], data["test"]):
            a, b = iou(predict_water(cpu_model, x) > 0.5, y)
            c = by_country.setdefault(cid.split("_")[0], [0, 0])
            c[0], c[1] = c[0] + a, c[1] + b
        model.to(dev)
        bi = bu = 0
        for _, (s1, s2, y) in chips["test"]:
            a, b = iou(baseline(v, s1, s2), y)
            bi, bu = bi + a, bu + b
        base_iou = bi / max(bu, 1)
        held = {}
        if data["bolivia"]:
            hi = hu = 0
            for _, (s1, s2, y) in chips["bolivia"]:
                a, b = iou(baseline(v, s1, s2), y)
                hi, hu = hi + a, hu + b
            held = {"bolivia_iou": round(evaluate("bolivia"), 4), "bolivia_baseline_iou": round(hi / max(hu, 1), 4),
                    "n_bolivia_chips": len(chips["bolivia"])}
        held["test_iou_by_country"] = {k: round(i / u, 4) for k, (i, u) in sorted(by_country.items()) if u}
        result["variants"][v] = {**held, "test_iou": round(test, 4), "baseline_iou": round(base_iou, 4),
                                 "best_val_iou": round(best, 4), "n_test_chips": len(chips["test"]),
                                 "baseline": {"s1": "Otsu threshold on VV", "s2": "MNDWI > 0",
                                              "rgb": "SatQuery colour land cover"}[v],
                                 "beats_baseline": bool(test > base_iou)}
        ckpt["variants"][v] = {"state": best_state, "in_ch": IN_CH[v], "metrics": result["variants"][v]}
        print(f"[{v}] test IoU {test:.4f} vs baseline {base_iou:.4f}", flush=True)
    result["s2_bands"] = S2_BANDS
    result["minutes"] = round((time.time() - t0) / 60, 1)
    ckpt["metrics"] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
