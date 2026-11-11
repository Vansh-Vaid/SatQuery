"""Train SatQuery's referring-expression grounder on VRSBench (CPU).

Model: satquery/models/hr_ground_net.py (dense 14x14 ResNet-18 map, expression-modulated cells, one box per
cell; the best cell wins). Data: VRSBench_train.json + Images_train.zip (10% of training images held out
for epoch selection), VRSBench_EVAL_referring.json + Images_val.zip for the test score. Scored with the
standard Acc@0.5 (IoU >= 0.5) and mean IoU.

Steps (each cached and resumable, so a slow machine can stop and restart):
  1. fit a fixed PCA (768 -> 256 channels) on 600 training images
  2. encode every needed image once into float16 14x14x256 maps        (~30-45 min on a laptop CPU)
  3. train with horizontal / vertical flips (direction words swapped to match)

  python training/train_vrsbench_grounding.py --features-only     # steps 1-2
  python training/train_vrsbench_grounding.py --out checkpoints/vrsbench_ground.pt

The earlier version (one box regressed from a 3x3 grid, 14.5% Acc@0.5) is in git history; the tool still
loads its checkpoints.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.models.hr_ground_net import (  # noqa: E402
    GRID,
    PROJ_DIM,
    RAW_DIM,
    DenseGrounder,
    encode_dense,
    flip_words,
    load_dense_encoder,
    project,
    token_ids,
)
from satquery.models.rsvqa_features import tokens  # noqa: E402


def parse_box(s: str):
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", s)]
    return [min(max(v / 100, 0.0), 1.0) for v in n[:4]] if len(n) >= 4 else None


def load_refs(root: Path) -> tuple[list[dict], list[dict]]:
    train = []
    for x in json.loads((root / "VRSBench_train.json").read_text()):
        q = x["conversations"][0]["value"]
        if "[refer]" not in q:
            continue
        m = re.search(r"<p>(.*?)</p>", q)
        box = parse_box(x["conversations"][1]["value"])
        if m and box and box[2] > box[0] and box[3] > box[1]:
            train.append({"img": x["image"], "q": m.group(1), "box": box})
    test = [{"img": x["image_id"], "q": x["question"], "box": parse_box(x["ground_truth"]), "cls": x.get("obj_cls")}
            for x in json.loads((root / "VRSBench_EVAL_referring.json").read_text())]
    return train, [r for r in test if r["box"]]


class ZipImages:
    def __init__(self, path: Path):
        self.z = zipfile.ZipFile(path)
        self.members = {Path(m).name: m for m in self.z.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg"))}

    def __call__(self, name: str) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(self.z.read(self.members[name]))).convert("RGB"))


def fit_pca(path: Path, imgs: ZipImages, names: list[str], enc, n_img: int = 600) -> dict:
    if path.exists():
        return dict(np.load(path))
    import torch
    t0 = time.time()
    pick = random.Random(0).sample(names, min(n_img, len(names)))
    raws = [encode_dense(enc, [imgs(n) for n in pick[k:k + 32]]) for k in range(0, len(pick), 32)]
    x = torch.from_numpy(np.concatenate(raws).transpose(0, 2, 3, 1).reshape(-1, RAW_DIM)).double()
    mean = x.mean(0)
    evals, evecs = torch.linalg.eigh(torch.cov((x - mean).T))
    order = torch.argsort(evals, descending=True)[:PROJ_DIM]
    pca = {"mean": mean.float().numpy(), "comp": evecs[:, order].float().numpy(),
           "scale": evals[order].clamp(min=1e-6).sqrt().float().numpy(),
           "explained": np.float32(evals[order].sum() / evals.sum())}
    np.savez(path, **pca)
    print(f"PCA {RAW_DIM}->{PROJ_DIM}: {100 * float(pca['explained']):.1f}% variance kept ({time.time() - t0:.0f}s)",
          flush=True)
    return pca


def encode_all(cache: Path, imgs: ZipImages, names: list[str], enc, pca: dict, bs: int = 32) -> np.ndarray:
    """(N, PROJ_DIM, GRID, GRID) float16, resumable: progress is a row count next to a memory-mapped file."""
    done_marker = cache.with_suffix(".json")
    if cache.exists() and done_marker.exists() and json.loads(done_marker.read_text()) == names:
        return np.load(cache, mmap_mode="r")
    part, prog = cache.with_suffix(".part.npy"), cache.with_suffix(".progress")
    shape = (len(names), PROJ_DIM, GRID, GRID)
    if part.exists() and prog.exists():
        out = np.lib.format.open_memmap(part, mode="r+")
        start = int(prog.read_text())
    else:
        out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=shape)
        start = 0
    t0 = time.time()
    for k in range(start, len(names), bs):
        batch = names[k:k + bs]
        out[k:k + len(batch)] = project(encode_dense(enc, [imgs(n) for n in batch]),
                                        pca["mean"], pca["comp"], pca["scale"]).astype(np.float16)
        if (k // bs) % 25 == 0 or k + bs >= len(names):
            out.flush()
            prog.write_text(str(k + len(batch)))
            rate = (k + len(batch) - start) / max(time.time() - t0, 1e-6)
            print(f"  {cache.stem}: {k + len(batch)}/{len(names)}  {rate:.1f} img/s  "
                  f"~{(len(names) - k - len(batch)) / max(rate, 1e-6) / 60:.0f} min left", flush=True)
    out.flush()
    del out
    part.replace(cache)
    done_marker.write_text(json.dumps(names))
    prog.unlink(missing_ok=True)
    return np.load(cache, mmap_mode="r")


def main() -> None:
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/vrsbench"))
    ap.add_argument("--cache", type=Path, default=None, help="feature cache folder (default: <root>/features)")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vrsbench_ground.pt"))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-count", type=int, default=3, help="keep expression tokens seen at least this often")
    ap.add_argument("--features-only", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    cache = args.cache or args.root / "features"
    cache.mkdir(parents=True, exist_ok=True)

    train, test = load_refs(args.root)
    tr_names = sorted({r["img"] for r in train})
    te_names = sorted({r["img"] for r in test})
    print(f"{len(train)} train / {len(test)} test expressions over "
          f"{len(tr_names)} / {len(te_names)} images", flush=True)

    enc = load_dense_encoder()
    tr_imgs = ZipImages(args.root / "Images_train.zip")
    pca = fit_pca(cache / "dense_pca.npz", tr_imgs, tr_names, enc)
    F_tr = encode_all(cache / "train_dense.npy", tr_imgs, tr_names, enc, pca)
    F_te = encode_all(cache / "val_dense.npy", ZipImages(args.root / "Images_val.zip"), te_names, enc, pca)
    if args.features_only:
        print(f"features ready ({(time.time() - t0) / 60:.1f} min)")
        return
    print("loading feature maps into memory...", flush=True)
    F_tr, F_te = torch.from_numpy(np.ascontiguousarray(F_tr)), torch.from_numpy(np.ascontiguousarray(F_te))
    tr_idx, te_idx = {n: i for i, n in enumerate(tr_names)}, {n: i for i, n in enumerate(te_names)}

    held = set(random.Random(0).sample(tr_names, len(tr_names) // 10))
    splits = {"train": [r for r in train if r["img"] not in held], "val": [r for r in train if r["img"] in held],
              "test": test}
    cnt: dict[str, int] = defaultdict(int)
    for r in splits["train"]:
        for t in set(tokens(r["q"])) | set(tokens(flip_words(r["q"], True, True))):
            cnt[t] += 1
    vocab = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= args.min_count))}
    print({k: len(v) for k, v in splits.items()}, "vocab", len(vocab), flush=True)

    def tensors(s: str, flips: bool):
        rows, idx = splits[s], (te_idx if s == "test" else tr_idx)
        variants = [(False, False), (True, False), (False, True), (True, True)] if flips else [(False, False)]
        ids = torch.from_numpy(np.stack([np.stack([token_ids(flip_words(r["q"], h, v), vocab) for h, v in variants])
                                         for r in rows]))                       # (N, V, T)
        return (torch.tensor([idx[r["img"]] for r in rows]), ids,
                torch.tensor([r["box"] for r in rows], dtype=torch.float32))

    data = {"train": tensors("train", True), "val": tensors("val", False), "test": tensors("test", False)}
    feats = {"train": F_tr, "val": F_tr, "test": F_te}

    # soft target: Gaussian (sigma = 1 cell) around the cell holding the box centre
    g = GRID
    ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    cell_xy = torch.stack([(xs.flatten() + 0.5) / g, (ys.flatten() + 0.5) / g], 1)    # (G*G, 2)

    def soft_target(box: torch.Tensor) -> torch.Tensor:
        c = torch.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
        d2 = ((c[:, None, :] - cell_xy[None]) * g).pow(2).sum(-1)
        t = torch.exp(-d2 / 2)
        return t / t.sum(1, keepdim=True)

    def giou(a, b):
        lt, rb = torch.max(a[..., :2], b[..., :2]), torch.min(a[..., 2:], b[..., 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        ua = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1) + (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1) - inter
        enc_ = (torch.max(a[..., 2:], b[..., 2:]) - torch.min(a[..., :2], b[..., :2])).prod(-1)
        iou = inter / ua.clamp(min=1e-6)
        return iou, iou - (enc_ - ua) / enc_.clamp(min=1e-6)

    model = DenseGrounder(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    im, ids, y = data["train"]
    steps = args.epochs * math.ceil(len(y) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def evaluate(s: str):
        model.eval()
        im_, ids_, y_ = data[s]
        boxes, probs = [], []
        for k in range(0, len(y_), 512):
            b, p = model.predict(feats[s][im_[k:k + 512]].float(), ids_[k:k + 512, 0])
            boxes.append(b)
            probs.append(p)
        model.train()
        iou, _ = giou(torch.cat(boxes), y_)
        return float((iou >= 0.5).float().mean()), float(iou.mean()), iou.numpy(), torch.cat(probs).numpy()

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        te = time.time()
        perm = torch.randperm(len(y))
        tot = 0.0
        for k in range(0, len(y), args.bs):
            b = perm[k:k + args.bs]
            v = torch.randint(0, 4, (len(b),))
            f = feats["train"][im[b]].float()
            box = y[b].clone()
            hflip, vflip = (v == 1) | (v == 3), (v == 2) | (v == 3)
            f[hflip] = f[hflip].flip(-1)
            f[vflip] = f[vflip].flip(-2)
            box[hflip] = torch.stack([1 - box[hflip, 2], box[hflip, 1], 1 - box[hflip, 0], box[hflip, 3]], 1)
            box[vflip] = torch.stack([box[vflip, 0], 1 - box[vflip, 3], box[vflip, 2], 1 - box[vflip, 1]], 1)
            logits, pred = model(f, ids[b, v])
            tgt = soft_target(box)
            ce = -(tgt * F.log_softmax(logits, 1)).sum(1).mean()
            _, gi = giou(pred, box[:, None, :].expand_as(pred))
            reg = ((pred - box[:, None, :]).abs().sum(-1) * 2 + (1 - gi))              # (B, G*G)
            loss = ce + (tgt * reg).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        acc, miou, _, _ = evaluate("val")
        print(f"epoch {ep + 1}/{args.epochs}: loss {tot / len(y):.3f}  held-out Acc@0.5 {acc:.4f}  mIoU {miou:.4f}  "
              f"({time.time() - te:.0f}s)", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    acc, miou, ious, probs = evaluate("test")
    hit = ious >= 0.5
    by = defaultdict(list)
    for r, h in zip(test, hit):
        by[r["cls"]].append(h)
    # confidence table: Acc@0.5 by cell probability (equal-count bins), used by the tool as its confidence
    order = np.argsort(probs)
    table = []
    for chunk in np.array_split(order, 10):
        table.append({"p_max": round(float(probs[chunk].max()), 4), "acc": round(float(hit[chunk].mean()), 4)})
    metrics = {"dataset": "VRSBench referring eval set", "model": "dense 14x14 FiLM grounder (hr_ground_net)",
               "test_acc_at_0.5": round(acc, 4), "test_mean_iou": round(miou, 4), "n_test": len(test),
               "held_out_acc_at_0.5": round(best, 4),
               "acc_by_class": {k: round(float(np.mean(v)), 3)
                                for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]},
               "confidence_table": table, "pca_variance_kept": round(float(pca["explained"]), 4),
               "minutes": round((time.time() - t0) / 60, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "dense", "model": best_state, "vocab": vocab, "pca_mean": pca["mean"],
                "pca_comp": pca["comp"], "pca_scale": pca["scale"], "metrics": metrics}, args.out)
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
