"""Train SatQuery's general land-cover model (change_mapper) on Dynamic World tiles built by fetch_dynamic_world.py.

One U-Net, nine Dynamic World classes, two input variants: "s2" (six multispectral bands; Sentinel-2 or Landsat)
and "rgb" (true colour as the app renders it, for photos and screenshots). Scored on Dynamic World's held-out
validation set (three experts per tile): pixel accuracy, mean IoU, per-class IoU. The gate: accuracy on SatQuery's
four measured classes must beat SatQuery's rule-based land cover on the same pixels. Optionally (--sen1floods11)
the model's water is also scored on Sen1Floods11 flood scenes, Bolivia included, as a test of events it was never
trained for.

  python training/train_landcover.py --data data/dynamic_world --out checkpoints/landcover_seg.pt \
      [--sen1floods11 data/sen1floods11]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from satquery.imagery.raster import Modality, Raster  # noqa: E402
from satquery.models.seg_net import DW_CLASSES, DW_TO_SATQUERY, IN_CH, build_unet, model_input, predict_classes  # noqa: E402

IGNORE = 255
SQ = ["water", "vegetation", "built_up", "bare_soil"]
COLLAPSE = np.array([SQ.index(DW_TO_SATQUERY[c]) for c in DW_CLASSES])


def raster_of(x: np.ndarray, variant: str) -> Raster:
    x = x.astype(np.float32)
    if variant == "rgb":
        return Raster(data=x[[2, 1, 0]], bands=["B04", "B03", "B02"], modality=Modality.OPTICAL_RGB,
                      source_format="geotiff", valid_mask=np.isfinite(x).all(0))
    return Raster(data=x, bands=["B02", "B03", "B04", "B08", "B11", "B12"], modality=Modality.S2_MS,
                  source_format="geotiff", valid_mask=np.isfinite(x).all(0))


def targets(y: np.ndarray) -> np.ndarray:
    t = np.full(y.shape, IGNORE, np.uint8)
    ok = (y >= 1) & (y <= 9)
    t[ok] = y[ok] - 1
    return t


def load(folder: Path, variant: str) -> list[tuple[np.ndarray, np.ndarray, Raster]]:
    out = []
    for f in sorted(folder.glob("*.npz")):
        d = np.load(f)
        r = raster_of(d["x"], variant)
        out.append((model_input(r, variant), targets(d["y"]), r))
    return out


def baseline_map(r: Raster) -> np.ndarray:
    from satquery.tools.landcover import classify
    from satquery.tools.vocab import CLASSES
    m = classify(r).label_map()
    return np.array([SQ.index(c) for c in CLASSES])[np.clip(m, 0, None)]


def main() -> None:
    import torch
    import torch.nn.functional as F
    from train_flood import augment, iou, load_chip, rgb_raster, split_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/dynamic_world"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/landcover_seg.pt"))
    ap.add_argument("--variants", nargs="+", default=["s2", "rgb"])
    ap.add_argument("--sen1floods11", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = len(DW_CLASSES)
    result = {"dataset": "Dynamic World expert labels (train) / expert validation set (test)", "classes": DW_CLASSES,
              "variants": {}}
    ckpt = {"arch": "seg_unet", "base": args.base, "variants": {}}
    for v in args.variants:
        train, test = load(args.data / "train", v), load(args.data / "test", v)
        if not train or not test:
            raise SystemExit(f"no tiles under {args.data}/train or /test")
        model = build_unet(IN_CH[v], args.base, K).to(dev)
        counts = np.bincount(np.concatenate([t[t != IGNORE] for _, t, _ in train]), minlength=K).astype(np.float64)
        weight = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
        weight = (weight / weight.mean()).to(dev)                       # rarer classes (snow, flooded veg.) count more
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, pct_start=0.1,
                                                    total_steps=args.epochs * max(1, math.ceil(len(train) / args.bs)))

        def evaluate(rows):
            model.eval()
            cpu = model.to("cpu")
            conf = np.zeros((K, K), np.int64)
            sq_m = sq_b = n = 0
            for x, t, r in rows:
                p = predict_classes(cpu, x).argmax(0)
                ok = t != IGNORE
                conf += np.bincount(t[ok] * K + p[ok], minlength=K * K).reshape(K, K)
                b = baseline_map(r)
                sq_m += int((COLLAPSE[p[ok]] == COLLAPSE[t[ok]]).sum())
                sq_b += int((b[ok] == COLLAPSE[t[ok]]).sum())
                n += int(ok.sum())
            model.to(dev).train()
            inter = np.diag(conf).astype(float)
            union = conf.sum(0) + conf.sum(1) - inter
            ious = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
            return {"pixel_acc": float(inter.sum() / max(conf.sum(), 1)), "miou": float(np.nanmean(ious)),
                    "class_iou": {c: (None if np.isnan(i) else round(float(i), 4)) for c, i in zip(DW_CLASSES, ious)},
                    "acc4": sq_m / max(n, 1), "baseline_acc4": sq_b / max(n, 1)}

        best, best_state = -1.0, None
        hold = train[: max(1, len(train) // 10)]
        for ep in range(args.epochs):
            te = time.time()
            for k in range(0, len(train), args.bs):
                xb, yb = [], []
                for i in rng.permutation(len(train))[: args.bs]:
                    x, t, _ = train[i]
                    h, w = t.shape
                    c = min(args.crop, h, w)
                    cy, cx = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
                    xa, ta = augment(x[:, cy:cy + c, cx:cx + c], t[cy:cy + c, cx:cx + c], v, rng)
                    xb.append(xa)
                    yb.append(ta.astype(np.int64))
                logits = model(torch.from_numpy(np.stack(xb)).to(dev))
                loss = F.cross_entropy(logits, torch.from_numpy(np.stack(yb)).to(dev), weight=weight, ignore_index=IGNORE)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            score = evaluate(hold)["miou"]
            if score > best:
                best, best_state = score, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            print(f"[{v}] epoch {ep + 1}/{args.epochs}: train-holdout mIoU {score:.4f} ({time.time() - te:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        m = evaluate(test)
        entry = {"variant": v, "test_pixel_acc": round(m["pixel_acc"], 4), "test_miou": round(m["miou"], 4),
                 "test_class_iou": m["class_iou"], "test_acc_4class": round(m["acc4"], 4),
                 "baseline_acc_4class": round(m["baseline_acc4"], 4), "n_test_tiles": len(test),
                 "beats_baseline": bool(m["acc4"] > m["baseline_acc4"]),
                 "description": f"land-cover model trained on Dynamic World expert labels; test mIoU {m['miou']:.2f}"}
        if args.sen1floods11:        # floods were never a training target: score the model's water on flood scenes
            files = {p.name: p for p in args.sen1floods11.rglob("*.tif")}
            water = [DW_CLASSES.index("water"), DW_CLASSES.index("flooded_vegetation")]
            for split in ("test", "bolivia"):
                inter = union = 0
                by_country: dict[str, list[int]] = {}
                for cid in split_ids(args.sen1floods11, split, required=False):
                    chip = load_chip(files, cid)
                    if chip is None:
                        continue
                    _, s2, y = chip
                    r = s2 if v == "s2" else rgb_raster(s2)
                    p = np.isin(predict_classes(model.cpu(), model_input(r, v)).argmax(0), water)
                    a, b = iou(p, y)
                    inter, union = inter + a, union + b
                    c = by_country.setdefault(cid.split("_")[0], [0, 0])
                    c[0], c[1] = c[0] + a, c[1] + b
                if union:
                    entry[f"sen1floods11_{split}_water_iou"] = round(inter / union, 4)
                    if split == "test":   # per country, India included: "any place" for floods
                        entry["sen1floods11_test_water_iou_by_country"] = {
                            k: round(i / u, 4) for k, (i, u) in sorted(by_country.items()) if u}
            model.to(dev)
        result["variants"][v] = entry
        ckpt["variants"][v] = {"state": best_state, "in_ch": IN_CH[v], "n_out": K, "classes": DW_CLASSES, "metrics": entry}
        print(json.dumps(entry, indent=2), flush=True)
    result["minutes"] = round((time.time() - t0) / 60, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
