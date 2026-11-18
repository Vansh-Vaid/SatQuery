"""Turn VRSBench into chat rows for fine-tuning a small VLM (training/finetune_vlm.py) on open-ended questions.

Rows use the exact prompt the app sends (satquery.models.vlm_prompts.build_prompt), so a model trained here sees
the same text at inference. Writes, under --out:
  train.jsonl, validation.jsonl      open VQA + captions from VRSBench_train.json (validation = held-out images)
  eval_vqa.jsonl, eval_caption.jsonl samples of the official evaluation sets, for the before/after comparison
  images/                            the images those rows refer to
Each row: {"id", "images": ["images/<file>"], "prompt", "answer", "task"}.

  python training/prepare_vrsbench_vlm.py --root data/vrsbench --out data/vrsbench_vlm
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.raster import Modality  # noqa: E402
from satquery.models.vlm_prompts import build_prompt  # noqa: E402

RGB = [Modality.OPTICAL_RGB]


def train_rows(root: Path) -> list[dict]:
    rows = []
    for k, x in enumerate(json.loads((root / "VRSBench_train.json").read_text(encoding="utf-8"))):
        q, a = x["conversations"][0]["value"], x["conversations"][1]["value"].strip()
        m = re.match(r"^\s*(?:<image>\s*)?\[(vqa|caption)\]\s*(.*)$", q, re.S)
        if not m or not a:
            continue
        task, text = m.group(1), m.group(2).strip()
        prompt = build_prompt("vqa" if task == "vqa" else "caption", text, RGB)
        rows.append({"id": f"vrs-train-{k}", "image": x["image"], "prompt": prompt, "answer": a, "task": task})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/vrsbench"))
    ap.add_argument("--out", type=Path, default=Path("data/vrsbench_vlm"))
    ap.add_argument("--n-vqa", type=int, default=9000)
    ap.add_argument("--n-caption", type=int, default=3000)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--n-eval-vqa", type=int, default=300)
    ap.add_argument("--n-eval-caption", type=int, default=100)
    args = ap.parse_args()
    rng = random.Random(0)
    rows = train_rows(args.root)
    imgs = sorted({r["image"] for r in rows})
    rng.shuffle(imgs)
    val_imgs = set(imgs[: max(1, int(len(imgs) * args.val_frac))])
    by = {"vqa": [r for r in rows if r["task"] == "vqa"], "caption": [r for r in rows if r["task"] == "caption"]}
    for v in by.values():
        rng.shuffle(v)
    train = [r for r in by["vqa"] if r["image"] not in val_imgs][: args.n_vqa] + \
            [r for r in by["caption"] if r["image"] not in val_imgs][: args.n_caption]
    val = [r for r in rows if r["image"] in val_imgs][:300]
    rng.shuffle(train)

    ev_vqa = json.loads((args.root / "VRSBench_EVAL_vqa.json").read_text(encoding="utf-8"))
    ev_cap = json.loads((args.root / "VRSBench_EVAL_Cap.json").read_text(encoding="utf-8"))
    rng.shuffle(ev_vqa)
    rng.shuffle(ev_cap)
    eval_vqa = [{"id": f"vrs-vqa-{x['question_id']}", "image": x["image_id"], "task": "vqa", "type": x.get("type"),
                 "prompt": build_prompt("vqa", x["question"], RGB), "question": x["question"],
                 "answer": x["ground_truth"]} for x in ev_vqa[: args.n_eval_vqa]]
    eval_cap = [{"id": f"vrs-cap-{k}", "image": x["image_id"], "task": "caption",
                 "prompt": build_prompt("caption", "Describe this image in detail.", RGB),
                 "answer": x["ground_truth"]} for k, x in enumerate(ev_cap[: args.n_eval_caption])]

    out_img = args.out / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    for zname, need in (("Images_train.zip", {r["image"] for r in train + val}),
                        ("Images_val.zip", {r["image"] for r in eval_vqa + eval_cap})):
        with zipfile.ZipFile(args.root / zname) as z:
            members = {Path(m).name: m for m in z.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg"))}
            for name in need:
                if name in members and not (out_img / name).exists():
                    (out_img / name).write_bytes(z.read(members[name]))
    for fname, rs in (("train.jsonl", train), ("validation.jsonl", val), ("eval_vqa.jsonl", eval_vqa),
                      ("eval_caption.jsonl", eval_cap)):
        with (args.out / fname).open("w", encoding="utf-8") as f:
            for r in rs:
                if (out_img / r["image"]).exists():
                    f.write(json.dumps({**{k: v for k, v in r.items() if k != "image"},
                                        "images": [f"images/{r['image']}"]}) + "\n")
    print({k: len(v) for k, v in (("train", train), ("validation", val), ("eval_vqa", eval_vqa),
                                   ("eval_caption", eval_cap))}, "images:", len(list(out_img.iterdir())))


if __name__ == "__main__":
    main()
