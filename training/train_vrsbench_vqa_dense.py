"""Train the dense VRSBench VQA specialist (CPU, or a GPU when one is present) on the 14x14 feature maps built for the grounder.

Reuses the grounder's feature cache (train_dense.npy / val_dense.npy, written by
train_vrsbench_grounding.py), so no image is encoded again. Model: satquery.models.hr_ground_net.DenseVQA.
Same data protocol as train_vrsbench_vqa.py (top 300 training answers, 10% of training images held out
for epoch selection, the VRSBench VQA eval set as the test score), so the two test scores compare directly.
Eval questions whose image is not in the cache are counted as wrong.

  python training/train_vrsbench_grounding.py --features-only        # once, if not done yet
  python training/train_vrsbench_vqa_dense.py --cache <feature folder> --out checkpoints/vrsbench_vqa_dense.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.models.hr_ground_net import DenseVQA, flip_words, token_ids  # noqa: E402
from satquery.models.rsvqa_features import tokens  # noqa: E402

VARIANTS = [(False, False), (True, False), (False, True), (True, True)]   # (horizontal, vertical) flips


def norm(a: str) -> str:
    return re.sub(r"\s+", " ", a.strip().rstrip(".").lower())


def load_cache(cache: Path, stem: str) -> tuple[np.ndarray, dict[str, int]]:
    names = json.loads((cache / f"{stem}.json").read_text())
    return np.load(cache / f"{stem}.npy", mmap_mode="r"), {n: i for i, n in enumerate(names)}


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/vrsbench"))
    ap.add_argument("--cache", type=Path, default=None, help="dense feature folder (default: <root>/features)")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vrsbench_vqa_dense.pt"))
    ap.add_argument("--answers", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    random.seed(0)
    t0 = time.time()
    cache = args.cache or args.root / "features"

    train = []
    for x in json.loads((args.root / "VRSBench_train.json").read_text(encoding="utf-8")):
        q = x["conversations"][0]["value"]
        if "[vqa]" in q:
            train.append({"img": x["image"], "q": re.sub(r"^<image>\s*\[vqa\]\s*", "", q).strip(),
                          "a": norm(x["conversations"][1]["value"])})
    test = [{"img": x["image_id"], "q": x["question"], "a": norm(x["ground_truth"]), "type": x["type"]}
            for x in json.loads((args.root / "VRSBench_EVAL_vqa.json").read_text(encoding="utf-8"))]
    F_tr, tr_idx = load_cache(cache, "train_dense")
    F_te, te_idx = load_cache(cache, "val_dense")
    pca = dict(np.load(cache / "dense_pca.npz"))
    train = [r for r in train if r["img"] in tr_idx]
    imgs = sorted({r["img"] for r in train})
    random.shuffle(imgs)
    held = set(imgs[: len(imgs) // 10])
    splits = {"train": [r for r in train if r["img"] not in held], "val": [r for r in train if r["img"] in held],
              "test": [r for r in test if r["img"] in te_idx]}
    print({k: len(v) for k, v in splits.items()}, f"(test: {len(test) - len(splits['test'])} questions without "
          "cached features count as wrong)", flush=True)

    answers = [a for a, _ in Counter(r["a"] for r in splits["train"]).most_common(args.answers)]
    aidx = {a: i for i, a in enumerate(answers)}
    cnt = Counter(t for r in splits["train"] for t in set(tokens(r["q"])) | set(tokens(flip_words(r["q"], True, True))))
    vocab = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= 3))}
    print("answers", len(answers), "vocab", len(vocab), flush=True)

    def tensors(s: str, flips: bool):
        rows, idx = splits[s], (te_idx if s == "test" else tr_idx)
        vs = VARIANTS if flips else VARIANTS[:1]
        ids = torch.from_numpy(np.stack([np.stack([token_ids(flip_words(r["q"], h, v), vocab) for h, v in vs])
                                         for r in rows]))                      # (N, V, T)
        y = torch.tensor([[aidx.get(norm(flip_words(r["a"], h, v)), -1) for h, v in vs] for r in rows])
        return torch.tensor([idx[r["img"]] for r in rows]), ids, y

    data = {"train": tensors("train", True), "val": tensors("val", False), "test": tensors("test", False)}
    feats = {"train": F_tr, "val": F_tr, "test": F_te}

    def gather(s: str, rows: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.asarray(feats[s][rows.numpy()], dtype=np.float32)).to(dev)

    model = DenseVQA(len(vocab), len(answers)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    im, ids, y = data["train"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, pct_start=0.1,
                                                total_steps=args.epochs * math.ceil(len(y) / args.bs))
    lossf = nn.CrossEntropyLoss(ignore_index=-1)

    def evaluate(s: str) -> tuple[float, dict[str, float]]:
        model.eval()
        im_, ids_, _ = data[s]
        preds = []
        with torch.inference_mode():
            for k in range(0, len(im_), 512):
                preds.append(model(gather(s, im_[k:k + 512]), ids_[k:k + 512, 0].to(dev)).argmax(1).cpu())
        pred = torch.cat(preds).numpy()
        model.train()
        per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r, p in zip(splits[s], pred):
            per[r.get("type", "all")][0] += int(answers[p] == r["a"])
        rows = test if s == "test" else splits[s]      # questions without features count as wrong
        for r in rows:
            per[r.get("type", "all")][1] += 1
        return (sum(c for c, _ in per.values()) / sum(n for _, n in per.values()),
                {k: c / n for k, (c, n) in sorted(per.items())})

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        te = time.time()
        perm = torch.randperm(len(y))
        tot = 0.0
        for k in range(0, len(y), args.bs):
            b = perm[k:k + args.bs]
            v = torch.randint(0, len(VARIANTS), (len(b),))
            f = gather("train", im[b])
            vd = v.to(dev)
            hflip, vflip = (vd == 1) | (vd == 3), (vd == 2) | (vd == 3)
            f[hflip] = f[hflip].flip(-1)
            f[vflip] = f[vflip].flip(-2)
            loss = lossf(model(f, ids[b, v].to(dev)), y[b, v].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        acc, _ = evaluate("val")
        print(f"epoch {ep + 1}/{args.epochs}: loss {tot / len(y):.3f}  held-out accuracy {acc:.4f}  "
              f"({time.time() - te:.0f}s)", flush=True)
        if acc > best:
            best, best_state = acc, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    acc, per = evaluate("test")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "dense", "model": best_state, "vocab": vocab, "answers": answers,
                "pca_mean": pca["mean"], "pca_comp": pca["comp"], "pca_scale": pca["scale"]}, args.out)
    metrics = {"dataset": "VRSBench VQA eval set (exact match after normalisation)", "test_accuracy": round(acc, 4),
               "test_per_type": {k: round(v, 4) for k, v in per.items()}, "n_test": len(test),
               "best_val_accuracy": round(best, 4), "answer_vocabulary": len(answers),
               "model": "DenseVQA: 14x14 ResNet-18 maps (PCA 256) + FiLM + attention pooling",
               "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
