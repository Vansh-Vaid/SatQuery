"""Train a referring-expression box regressor on VRSBench (CPU baseline for grounding).

Input: cached ImageNet ResNet-18 3x3-grid image features (from train_vrsbench_vqa.py) + bag-of-words of the
referring expression. Output: one box (x1, y1, x2, y2) in 0-1 image coordinates. Loss: L1 + GIoU.
Scored with the standard Acc@0.5 (IoU >= 0.5) and mean IoU on VRSBench_EVAL_referring.json.

  python training/train_vrsbench_vqa.py --features-only        # caches image features first
  python training/train_vrsbench_grounding.py --out checkpoints/vrsbench_ground.pt
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.models.hr_features import IMG_DIM  # noqa: E402
from satquery.models.rsvqa_features import encode_question, tokens  # noqa: E402


def parse_box(s: str):
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", s)]
    return [min(max(v / 100, 0.0), 1.0) for v in n[:4]] if len(n) >= 4 else None


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/vrsbench"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vrsbench_ground.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    t0 = time.time()

    train = []
    for x in json.loads((args.root / "VRSBench_train.json").read_text()):
        q = x["conversations"][0]["value"]
        if "[refer]" not in q:
            continue
        m = re.search(r"<p>(.*?)</p>", q)
        box = parse_box(x["conversations"][1]["value"])
        if m and box:
            train.append({"img": x["image"], "q": m.group(1), "box": box})
    test = [{"img": x["image_id"], "q": x["question"], "box": parse_box(x["ground_truth"]), "cls": x.get("obj_cls")}
            for x in json.loads((args.root / "VRSBench_EVAL_referring.json").read_text())]
    test = [r for r in test if r["box"]]
    cache = args.root / "features"
    tr_names = json.loads((cache / "train_r18grid.json").read_text())
    te_names = json.loads((cache / "val_r18grid.json").read_text())
    tr_idx, te_idx = {n: i for i, n in enumerate(tr_names)}, {n: i for i, n in enumerate(te_names)}
    n_tr, n_te = len(train), len(test)
    train = [r for r in train if r["img"] in tr_idx]   # images with cached features (all images that have VQA)
    test = [r for r in test if r["img"] in te_idx]
    print(f"kept {len(train)}/{n_tr} train and {len(test)}/{n_te} test referring expressions", flush=True)
    F_tr = np.load(cache / "train_r18grid.npy").astype(np.float32)
    F_te = np.load(cache / "val_r18grid.npy").astype(np.float32)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    F_tr, F_te = (F_tr - mu) / sd, (F_te - mu) / sd
    imgs = sorted({r["img"] for r in train})
    random.shuffle(imgs)
    held = set(imgs[: len(imgs) // 10])
    splits = {"train": [r for r in train if r["img"] not in held], "val": [r for r in train if r["img"] in held], "test": test}
    cnt = Counter(t for r in splits["train"] for t in tokens(r["q"]))
    vocab = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= 3))}
    print({k: len(v) for k, v in splits.items()}, "vocab", len(vocab), flush=True)

    def tensors(s):
        F, idx = (F_te, te_idx) if s == "test" else (F_tr, tr_idx)
        rows = splits[s]
        return (torch.from_numpy(F), torch.tensor([idx[r["img"]] for r in rows]),
                torch.from_numpy(np.stack([encode_question(r["q"], vocab) for r in rows])),
                torch.tensor([r["box"] for r in rows], dtype=torch.float32))

    data = {s: tensors(s) for s in splits}

    class Grounder(nn.Module):
        def __init__(self):
            super().__init__()
            self.img = nn.Sequential(nn.Linear(IMG_DIM, 768), nn.Tanh(), nn.Dropout(0.3))
            self.q = nn.Sequential(nn.Linear(len(vocab), 768), nn.Tanh(), nn.Dropout(0.3))
            self.head = nn.Sequential(nn.Linear(768 * 2, 512), nn.ReLU(), nn.Linear(512, 4))

        def forward(self, f, q):
            i, w = self.img(f), self.q(q)
            o = torch.sigmoid(self.head(torch.cat([i * w, w], 1)))
            cx, cy, bw, bh = o.unbind(1)
            return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).clamp(0, 1)

    def iou(a, b):
        lt, rb = torch.max(a[:, :2], b[:, :2]), torch.min(a[:, 2:], b[:, 2:])
        inter = (rb - lt).clamp(min=0).prod(1)
        ua = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1) + (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1) - inter
        enc = (torch.max(a[:, 2:], b[:, 2:]) - torch.min(a[:, :2], b[:, :2])).prod(1)
        i = inter / ua.clamp(min=1e-6)
        return i, i - (enc - ua) / enc.clamp(min=1e-6)

    model = Grounder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    def evaluate(s):
        model.eval()
        F, im, q, y = data[s]
        with torch.inference_mode():
            p = torch.cat([model(F[im[k:k + 4096]], q[k:k + 4096]) for k in range(0, len(y), 4096)])
        model.train()
        i, _ = iou(p, y)
        return float((i >= 0.5).float().mean()), float(i.mean()), i.numpy()

    F, im, q, y = data["train"]
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = torch.randperm(len(y))
        for k in range(0, len(y), 256):
            b = perm[k:k + 256]
            p = model(F[im[b]], q[b])
            _, g = iou(p, y[b])
            loss = (p - y[b]).abs().mean() * 5 + (1 - g).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, miou, _ = evaluate("val")
        print(f"epoch {ep + 1}: held-out Acc@0.5 {acc:.4f}  mIoU {miou:.4f}", flush=True)
        if miou > best:
            best, best_state = miou, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    acc, miou, ious = evaluate("test")
    by = defaultdict(list)
    for r, v in zip(test, ious):
        by[r["cls"]].append(v >= 0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "vocab": vocab, "feat_mean": mu, "feat_std": sd, "img_dim": IMG_DIM}, args.out)
    metrics = {"dataset": "VRSBench referring eval set", "test_acc_at_0.5": round(acc, 4), "test_mean_iou": round(miou, 4),
               "n_test": len(test), "acc_by_class": {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:10]},
               "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
