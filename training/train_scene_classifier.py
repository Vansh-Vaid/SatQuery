"""Remote-sensing adaptation that runs on a CPU: fine-tune ResNet-18 (ImageNet) on EuroSAT.

EuroSAT (Helber et al. 2019): 27,000 Sentinel-2 RGB patches (64x64 px, 10 m), 10 land-use classes,
CC-BY 4.0. Official torchgeo splits (train 16,199 / val 5,399 / test 5,399) are used, and test is
touched exactly once at the end.

  python training/train_scene_classifier.py --data data/eurosat --out checkpoints/scene_resnet18.pt

Writes the checkpoint (weights + class names + normalisation) and <out>.metrics.json with per-class
test precision/recall and the confusion matrix. Augmentations include brightness/contrast/saturation
jitter so the model tolerates SatQuery's percentile-stretched renderings of Sentinel-2 GeoTIFFs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def split_files(root: Path, split: str) -> list[tuple[Path, int]]:
    names = [n.strip() for n in (root / f"eurosat-{split}.txt").read_text().splitlines() if n.strip()]
    base = next(root.glob("rgb/*"))  # rgb/2750
    out = []
    for n in names:
        stem = Path(n).stem
        cls = stem.rsplit("_", 1)[0]
        p = base / cls / f"{stem}.jpg"
        if p.exists():
            out.append((p, CLASSES.index(cls)))
    return out


def main() -> None:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/eurosat"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/scene_resnet18.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    train_tf = T.Compose([
        T.RandomResizedCrop(args.size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.02),
        T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.Resize(args.size), T.ToTensor(), T.Normalize(MEAN, STD)])

    class DS(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y = self.items[i]
            return self.tf(Image.open(p).convert("RGB")), y

    tr, va, te = (split_files(args.data, s) for s in ("train", "val", "test"))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    dl = lambda items, tf, shuffle: DataLoader(DS(items, tf), batch_size=args.bs, shuffle=shuffle, num_workers=0)
    train_dl, val_dl, test_dl = dl(tr, train_tf, True), dl(va, eval_tf, False), dl(te, eval_tf, False)

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    params = [{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 4e-4},
              {"params": model.fc.parameters(), "lr": 2e-3}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[4e-4, 2e-3], total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def evaluate(loader):
        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for x, y in loader:
                ps.append(model(x).argmax(1).numpy())
                ys.append(y.numpy())
        model.train()
        y, p = np.concatenate(ys), np.concatenate(ps)
        return float((y == p).mean()), y, p

    best, t0 = -1.0, time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(y)
            n += len(y)
            if i % 40 == 0:
                print(f"  epoch {ep + 1} step {i}/{len(train_dl)} loss {tot / n:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        acc, _, _ = evaluate(val_dl)
        print(f"epoch {ep + 1}: train loss {tot / n:.4f}  val acc {acc:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": CLASSES, "mean": MEAN, "std": STD,
                        "size": args.size, "arch": "resnet18", "val_acc": acc, "epoch": ep + 1}, args.out)

    state = torch.load(args.out, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    acc, y, p = evaluate(test_dl)
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    per = {c: {"precision": round(float(cm[i, i] / max(cm[:, i].sum(), 1)), 4),
               "recall": round(float(cm[i, i] / max(cm[i].sum(), 1)), 4), "n": int(cm[i].sum())}
           for i, c in enumerate(CLASSES)}
    metrics = {"dataset": "EuroSAT RGB (torchgeo official split)", "test_accuracy": round(acc, 4),
               "best_val_accuracy": round(best, 4), "epochs": args.epochs, "per_class": per,
               "confusion_matrix": cm.tolist(), "classes": CLASSES,
               "train_minutes": round((time.time() - t0) / 60, 1)}
    Path(str(args.out) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k in ("test_accuracy", "best_val_accuracy", "train_minutes")}))


if __name__ == "__main__":
    main()
