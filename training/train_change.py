"""Train the optional learned change detector (Siamese U-Net) on LEVIR-CD.

LEVIR-CD layout (https://justchenhao.github.io/LEVIR/):
    <root>/{train,val,test}/{A,B,label}/<name>.png     (1024x1024 RGB pairs, binary masks)

  python training/train_change.py --root /data/LEVIR-CD --out checkpoints/change_unet.pt --epochs 40

Enable in SatQuery: SATQUERY_CHANGE_CKPT=checkpoints/change_unet.pt (change_detector method=learned/auto).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.models.change_net import SiameseUNet  # noqa: E402


class LevirCD:
    def __init__(self, root: Path, split: str, crop: int = 256, train: bool = True):
        self.files = sorted((root / split / "A").glob("*.png"))
        self.root, self.split, self.crop, self.train = root, split, crop, train

    def __len__(self) -> int:
        return len(self.files) * (16 if self.train else 1)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        f = self.files[i % len(self.files)]
        a = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255
        b = np.asarray(Image.open(self.root / self.split / "B" / f.name).convert("RGB"), np.float32) / 255
        m = (np.asarray(Image.open(self.root / self.split / "label" / f.name).convert("L")) > 127).astype(np.float32)
        if self.train:
            h, w = m.shape
            y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
            a, b, m = a[y:y + self.crop, x:x + self.crop], b[y:y + self.crop, x:x + self.crop], m[y:y + self.crop, x:x + self.crop]
            if random.random() < 0.5:
                a, b, m = a[:, ::-1], b[:, ::-1], m[:, ::-1]
            if random.random() < 0.5:
                a, b, m = a[::-1], b[::-1], m[::-1]
            if random.random() < 0.5:
                a, b = b, a  # temporal symmetry
            gain = random.uniform(0.85, 1.15)
            b = np.clip(b * gain, 0, 1)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return t(a), t(b), torch.from_numpy(np.ascontiguousarray(m))[None]


def dice_loss(logits, target, eps=1.0):
    import torch
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return 1 - (num / den).mean()


def evaluate(model, ds, dev) -> dict[str, float]:
    import torch
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            a, b, m = ds[i]
            logits = model(a[None].to(dev), b[None].to(dev))
            p = (torch.sigmoid(logits) > 0.5).cpu()
            tp += int((p & (m[None] > 0)).sum())
            fp += int((p & (m[None] == 0)).sum())
            fn += int((~p & (m[None] > 0)).sum())
    model.train()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "iou": tp / max(tp + fp + fn, 1)}


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseUNet().to(dev)
    tr = DataLoader(LevirCD(args.root, "train"), batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    val = LevirCD(args.root, "val", train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=dev))
    best = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        tot = 0.0
        for a, b, m in tr:
            a, b, m = a.to(dev), b.to(dev), m.to(dev)
            logits = model(a, b)
            loss = bce(logits, m) + dice_loss(logits, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        metrics = evaluate(model, val, dev)
        print(f"epoch {ep + 1}: loss {tot / len(tr):.4f} val {metrics}", flush=True)
        if metrics["f1"] > best:
            best = metrics["f1"]
            torch.save({"model": model.state_dict(), "val": metrics, "epoch": ep + 1}, args.out)
    print(f"best val F1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
