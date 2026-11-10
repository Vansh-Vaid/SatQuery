"""Train SatQuery's VRSBench VQA specialist for high-resolution aerial images (CPU).

Image encoder: frozen ImageNet ResNet-18 with a 3x3 spatial grid (satquery/models/hr_features.py).
Question encoder: bag of unigrams/bigrams. Fusion: elementwise product. Answers: the most frequent training
answers (top 300 cover ~90% of the evaluation answers). Data: VRSBench_train.json + Images_train.zip for
training (10% of training images held out for epoch selection), VRSBench_EVAL_vqa.json + Images_val.zip
for the test score. Image features are cached as float16 .npy files, so re-runs skip the encoder.

  python training/train_vrsbench_vqa.py --root data/vrsbench --out checkpoints/vrsbench_vqa.pt
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.models.hr_features import IMG_DIM, encode_batch, load_encoder  # noqa: E402
from satquery.models.rsvqa_features import build_model, encode_question, tokens  # noqa: E402


def norm(a: str) -> str:
    return re.sub(r"\s+", " ", a.strip().rstrip(".").lower())


def features(zip_path: Path, names: list[str], cache: Path, body, bs: int = 64) -> dict[str, int]:
    """Encode the images in `names` once (cached; resumable in chunks). Returns name -> row index."""
    idx = {n: i for i, n in enumerate(names)}
    done = cache.with_suffix(".json")
    if cache.exists() and done.exists() and json.loads(done.read_text()) == names:
        return idx
    from PIL import Image
    z = zipfile.ZipFile(zip_path)
    members = {Path(m).name: m for m in z.namelist() if m.endswith(".png")}
    part = cache.with_suffix(".part.npy")
    out = np.load(part) if part.exists() else np.zeros((len(names), IMG_DIM), np.float16)
    filled = int((np.abs(out).sum(1) > 0).sum()) if part.exists() else 0
    t0 = time.time()
    for k in range(filled - filled % bs, len(names), bs):
        batch = [np.asarray(Image.open(io.BytesIO(z.read(members[n]))).convert("RGB")) for n in names[k:k + bs]]
        out[k:k + len(batch)] = encode_batch(body, batch)
        if (k // bs) % 20 == 0:
            print(f"  {cache.stem}: {k + len(batch)}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
            np.save(part, out)
    np.save(cache, out)
    done.write_text(json.dumps(names))
    part.unlink(missing_ok=True)
    return idx


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/vrsbench"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vrsbench_vqa.pt"))
    ap.add_argument("--answers", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--features-only", action="store_true", help="encode images and stop (lets a slow machine run in chunks)")
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    t0 = time.time()

    train = []
    for x in json.loads((args.root / "VRSBench_train.json").read_text()):
        q = x["conversations"][0]["value"]
        if "[vqa]" not in q:
            continue
        train.append({"img": x["image"], "q": re.sub(r"^<image>\s*\[vqa\]\s*", "", q).strip(),
                      "a": norm(x["conversations"][1]["value"]), "type": "train"})
    test = [{"img": x["image_id"], "q": x["question"], "a": norm(x["ground_truth"]), "type": x["type"]}
            for x in json.loads((args.root / "VRSBench_EVAL_vqa.json").read_text())]
    imgs = sorted({r["img"] for r in train})
    random.shuffle(imgs)
    held = set(imgs[: len(imgs) // 10])
    splits = {"train": [r for r in train if r["img"] not in held], "val": [r for r in train if r["img"] in held],
              "test": test}
    print({k: len(v) for k, v in splits.items()}, flush=True)

    body = load_encoder()
    cache = args.root / "features"
    cache.mkdir(exist_ok=True)
    tr_names = sorted({r["img"] for r in train})
    te_names = sorted({r["img"] for r in test})
    tr_idx = features(args.root / "Images_train.zip", tr_names, cache / "train_r18grid.npy", body)
    te_idx = features(args.root / "Images_val.zip", te_names, cache / "val_r18grid.npy", body)
    if args.features_only:
        print("features cached")
        return
    F_tr = np.load(cache / "train_r18grid.npy").astype(np.float32)
    F_te = np.load(cache / "val_r18grid.npy").astype(np.float32)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    F_tr = (F_tr - mu) / sd
    F_te = (F_te - mu) / sd
    print(f"features ready ({time.time() - t0:.0f}s)", flush=True)

    answers = [a for a, _ in Counter(r["a"] for r in splits["train"]).most_common(args.answers)]
    aidx = {a: i for i, a in enumerate(answers)}
    cnt = Counter(t for r in splits["train"] for t in tokens(r["q"]))
    vocab = {t: k for k, t in enumerate(sorted(t for t, c in cnt.items() if c >= 3))}
    print("answers", len(answers), "vocab", len(vocab), flush=True)

    def tensors(s):
        rows = splits[s]
        F, idx = (F_te, te_idx) if s == "test" else (F_tr, tr_idx)
        im = torch.tensor([idx[r["img"]] for r in rows])
        q = torch.from_numpy(np.stack([encode_question(r["q"], vocab) for r in rows]))
        y = torch.tensor([aidx.get(r["a"], -1) for r in rows])
        return torch.from_numpy(F), im, q, y

    data = {s: tensors(s) for s in splits}
    model = build_model(len(vocab), img_dim=IMG_DIM, n_answers=len(answers), hidden=768)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)

    def evaluate(s):
        model.eval()
        F, im, q, y = data[s]
        preds = []
        with torch.inference_mode():
            for k in range(0, len(y), 4096):
                preds.append(model(F[im[k:k + 4096]], q[k:k + 4096]).argmax(1))
        pred = torch.cat(preds).numpy()
        model.train()
        per = defaultdict(lambda: [0, 0])
        for r, p in zip(splits[s], pred):
            per[r["type"]][0] += int(answers[p] == r["a"])
            per[r["type"]][1] += 1
        accs = {k: c / n for k, (c, n) in sorted(per.items())}
        return sum(c for c, _ in per.values()) / sum(n for _, n in per.values()), accs

    F, im, q, y = data["train"]
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = torch.randperm(len(y))
        for k in range(0, len(y), args.bs):
            b = perm[k:k + args.bs]
            loss = lossf(model(F[im[b]], q[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        oa, _ = evaluate("val")
        print(f"epoch {ep + 1}: held-out train-image accuracy {oa:.4f}", flush=True)
        if oa > best:
            best, best_state = oa, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    oa, per = evaluate("test")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "vocab": vocab, "answers": answers, "feat_mean": mu, "feat_std": sd,
                "img_dim": IMG_DIM, "hidden": 768}, args.out)
    metrics = {"dataset": "VRSBench VQA eval set (exact match after normalisation)", "test_accuracy": round(oa, 4),
               "test_per_type": {k: round(v, 4) for k, v in per.items()}, "n_test": len(test),
               "best_val_accuracy": round(best, 4), "answer_vocabulary": len(answers),
               "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
