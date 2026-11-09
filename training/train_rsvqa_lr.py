"""Train SatQuery's RSVQA-LR VQA specialist (CPU, minutes).

Image encoder: the EuroSAT-adapted ResNet-18 (frozen) + SatQuery land-cover fractions.
Question encoder: bag of unigrams/bigrams. Fusion: elementwise product (Lobry et al. 2020).
Data: RSVQA-LR official splits. Train on train (572 images, 57k questions), select the epoch
on val, and score test once.

  python training/train_rsvqa_lr.py --root data/rsvqa_lr --scene checkpoints/scene_resnet18.pt \
      --out checkpoints/rsvqa_lr.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.io import load_image  # noqa: E402
from satquery.models.rsvqa_features import (ANSWERS, answer_mask, build_model, encode_question,  # noqa: E402
                                            image_features, load_backbone, quantize, question_type, tokens)

TYPE_MAP = {"comp": "comparison", "presence": "presence", "count": "count", "rural_urban": "rural_urban"}


def load_split(root: Path, split: str):
    q = json.load(open(root / f"LR_split_{split}_questions.json"))["questions"]
    a = {x["id"]: x for x in json.load(open(root / f"LR_split_{split}_answers.json"))["answers"]}
    rows = []
    for x in q:
        if not x.get("active") or not x.get("answers_ids"):
            continue
        ans = a.get(x["answers_ids"][0])
        if not ans or not ans.get("active", True):
            continue
        t = x["type"]
        label = quantize(float(ans["answer"])) if t == "count" else ans["answer"].strip().lower()
        if label not in ANSWERS:
            continue
        rows.append({"img": x["img_id"], "q": x["question"], "type": TYPE_MAP.get(t, t), "label": ANSWERS.index(label)})
    return rows


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/rsvqa_lr"))
    ap.add_argument("--scene", default="checkpoints/scene_resnet18.pt")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/rsvqa_lr.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()

    splits = {s: load_split(args.root, s) for s in ("train", "val", "test")}
    print({s: len(r) for s, r in splits.items()})

    # image features once per image (all splits share Images_LR)
    backbone, mean, std = load_backbone(args.scene)
    feats: dict[int, np.ndarray] = {}
    ids = sorted({r["img"] for rs in splits.values() for r in rs})
    for k, i in enumerate(ids):
        feats[i] = image_features(load_image(args.root / "Images_LR" / f"{i}.tif", "optical_rgb"), backbone, mean, std)
        if k % 100 == 0:
            print(f"features {k}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    fmat = np.stack([feats[i] for i in ids])
    fmu, fsd = fmat.mean(0), fmat.std(0) + 1e-6           # standardise image features (train stats below)
    tr_ids = sorted({r["img"] for r in splits["train"]})
    fmu = np.stack([feats[i] for i in tr_ids]).mean(0)
    fsd = np.stack([feats[i] for i in tr_ids]).std(0) + 1e-6

    cnt = Counter(t for r in splits["train"] for t in tokens(r["q"]))
    vocab = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= args.min_count))}
    print("vocab", len(vocab))

    def tensors(rows):
        img = torch.from_numpy(np.stack([(feats[r["img"]] - fmu) / fsd for r in rows]).astype(np.float32))
        q = torch.from_numpy(np.stack([encode_question(r["q"], vocab) for r in rows]))
        y = torch.tensor([r["label"] for r in rows])
        mask = torch.from_numpy(np.stack([answer_mask(question_type(r["q"])) for r in rows]))
        return img, q, y, mask

    data = {s: tensors(r) for s, r in splits.items()}
    model = build_model(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def evaluate(split):
        model.eval()
        img, q, y, mask = data[split]
        with torch.inference_mode():
            logits = model(img, q).masked_fill(~mask, -1e9)
        pred = logits.argmax(1).numpy()
        model.train()
        yy = y.numpy()
        per = defaultdict(lambda: [0, 0])
        for r, p, t in zip(splits[split], pred, yy):
            per[r["type"]][0] += int(p == t)
            per[r["type"]][1] += 1
        accs = {k: c / n for k, (c, n) in per.items()}
        return float((pred == yy).mean()), accs, float(np.mean(list(accs.values())))

    best, best_state = -1.0, None
    img, q, y, mask = data["train"]
    n = len(y)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            logits = model(img[idx], q[idx]).masked_fill(~mask[idx], -1e9)
            loss = lossf(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        oa, per, aa = evaluate("val")
        print(f"epoch {ep + 1}: loss {tot / n:.4f}  val OA {oa:.4f}  AA {aa:.4f}  {json.dumps({k: round(v, 3) for k, v in per.items()})}", flush=True)
        if aa > best:
            best, best_state = aa, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    oa, per, aa = evaluate("test")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "vocab": vocab, "answers": ANSWERS, "feat_mean": fmu, "feat_std": fsd,
                "scene_ckpt": str(args.scene)}, args.out)
    metrics = {"dataset": "RSVQA-LR (official splits, active questions)", "test_overall_accuracy": round(oa, 4),
               "test_average_accuracy": round(aa, 4), "test_per_type": {k: round(v, 4) for k, v in per.items()},
               "best_val_average_accuracy": round(best, 4), "n_test": len(splits["test"]),
               "count_protocol": "bins 0 / 1-10 / 11-100 / 101-1000 / >1000", "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
