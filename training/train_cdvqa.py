"""Train SatQuery's CDVQA change-question specialist (CPU, minutes).

Pair encoder: frozen EuroSAT-adapted ResNet-18 on both dates + feature differences + pixel change statistics
+ SatQuery land-cover fractions. Question encoder: bag of unigrams/bigrams. Fusion: elementwise product.
Data: the CDVQA subset written by eval/fetch_cdvqa.py. The train split is divided by image pair into
train / validation (no pair appears in both); the test split is scored once with the selected epoch.

  python eval/fetch_cdvqa.py --split train --shards 150
  python eval/fetch_cdvqa.py --split test  --shards 60
  python training/train_cdvqa.py --out checkpoints/cdvqa.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.io import load_image  # noqa: E402
from satquery.models.cdvqa_features import IMG_DIM, pair_features  # noqa: E402
from satquery.models.rsvqa_features import build_model, encode_question, load_backbone, tokens  # noqa: E402


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("eval/data/cdvqa_train.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("eval/data/cdvqa_test.jsonl"))
    ap.add_argument("--scene", default="checkpoints/scene_resnet18.pt")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/cdvqa.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    t0 = time.time()

    train_all, test = read(args.train), read(args.test)
    pairs = sorted({r["pair"] for r in train_all})
    random.shuffle(pairs)
    val_pairs = set(pairs[: int(len(pairs) * args.val_frac)])
    splits = {"train": [r for r in train_all if r["pair"] not in val_pairs],
              "val": [r for r in train_all if r["pair"] in val_pairs], "test": test}
    print({k: (len(v), len({r['pair'] for r in v})) for k, v in splits.items()}, "(questions, pairs)", flush=True)

    def fkey(s: str, r: dict) -> str:
        return f"{'test' if s == 'test' else 'train'}/{r['pair']}"

    backbone, mean, std = load_backbone(args.scene)
    feats: dict[str, np.ndarray] = {}
    todo = {fkey(s, r): r["images"] for s, rs in splits.items() for r in rs}
    for k, (key, ims) in enumerate(todo.items()):
        feats[key] = pair_features(load_image(ims[0]), load_image(ims[1]), backbone, mean, std)
        if k % 50 == 0:
            print(f"features {k}/{len(todo)} pairs ({time.time() - t0:.0f}s)", flush=True)

    answers = sorted({r["answer"] for r in splits["train"]})
    aidx = {a: i for i, a in enumerate(answers)}
    cnt = Counter(t for r in splits["train"] for t in tokens(r["query"]))
    vocab = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= 2))}
    tr = np.stack([feats[fkey("train", r)] for r in splits["train"]])
    fmu, fsd = tr.mean(0), tr.std(0) + 1e-6
    print("answers", len(answers), "vocab", len(vocab), flush=True)

    def tensors(s):
        rows = splits[s]
        img = torch.from_numpy(np.stack([(feats[fkey(s, r)] - fmu) / fsd for r in rows]).astype(np.float32))
        q = torch.from_numpy(np.stack([encode_question(r["query"], vocab) for r in rows]))
        y = torch.tensor([aidx.get(r["answer"], -1) for r in rows])
        return img, q, y

    data = {s: tensors(s) for s in splits}
    model = build_model(len(vocab), img_dim=IMG_DIM, n_answers=len(answers))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    def evaluate(s):
        model.eval()
        img, q, y = data[s]
        with torch.inference_mode():
            pred = model(img, q).argmax(1).numpy()
        model.train()
        per = defaultdict(lambda: [0, 0])
        for r, p, t in zip(splits[s], pred, y.numpy()):
            per[r["qtype"]][0] += int(p == t)
            per[r["qtype"]][1] += 1
        accs = {k: c / n for k, (c, n) in sorted(per.items())}
        return float((pred == y.numpy()).mean()), accs, float(np.mean(list(accs.values())))

    img, q, y = data["train"]
    best, best_state, n = -1.0, None, len(y)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for k in range(0, n, args.bs):
            idx = perm[k:k + args.bs]
            loss = lossf(model(img[idx], q[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        oa, per, aa = evaluate("val")
        print(f"epoch {ep + 1}: val OA {oa:.4f} AA {aa:.4f}", flush=True)
        if oa > best:
            best, best_state = oa, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    oa, per, aa = evaluate("test")
    # majority answer per question type, as a floor for context
    maj = {t: Counter(r["answer"] for r in splits["train"] if r["qtype"] == t).most_common(1)[0][0]
           for t in {r["qtype"] for r in splits["train"]}}
    base = float(np.mean([maj.get(r["qtype"]) == r["answer"] for r in test]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "vocab": vocab, "answers": answers, "feat_mean": fmu, "feat_std": fsd,
                "img_dim": IMG_DIM}, args.out)
    metrics = {"dataset": "CDVQA test (subset via eval/fetch_cdvqa.py)", "test_overall_accuracy": round(oa, 4),
               "test_average_accuracy": round(aa, 4), "test_per_type": {k: round(v, 4) for k, v in per.items()},
               "majority_per_type_baseline": round(base, 4), "best_val_overall_accuracy": round(best, 4),
               "n_test_questions": len(test), "n_test_pairs": len({r["pair"] for r in test}),
               "n_train_pairs": len({r["pair"] for r in splits["train"]}), "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
