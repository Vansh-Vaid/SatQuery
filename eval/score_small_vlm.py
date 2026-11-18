"""Score a VLM (optionally with a LoRA adapter) on the rows written by training/prepare_vrsbench_vlm.py.

Uses the app's own backend (satquery.models.vlm_backend.HFBackend), so the score reflects what users would get.
Reports open-VQA exact-match accuracy (metrics.vqa_correct), caption BLEU-4 / ROUGE-L, and seconds per answer on
the CPU (the free deployment has no GPU), measured on a few rows.

  python eval/score_small_vlm.py --data data/vrsbench_vlm --base HuggingFaceTB/SmolVLM-256M-Instruct \
      [--adapter checkpoints/smolvlm-rs-lora] --out results/smolvlm_lora.json [--cpu-rows 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import bleu, rouge_l, vqa_correct  # noqa: E402


def rows(path: Path, limit: int | None) -> list[dict]:
    out = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return out[:limit] if limit else out


def answer(backend, root: Path, row: dict, max_new: int) -> str:
    from PIL import Image
    images = [Image.open(root / p).convert("RGB") for p in row["images"]]
    return backend.generate(images, row["prompt"], max_new_tokens=max_new)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-vqa", type=int, default=300)
    ap.add_argument("--n-caption", type=int, default=100)
    ap.add_argument("--cpu-rows", type=int, default=5, help="rows timed on the CPU (0 = skip)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    from satquery.models.vlm_backend import HFBackend

    backend = HFBackend(args.base, args.adapter, device=args.device)
    vqa = rows(args.data / "eval_vqa.jsonl", args.n_vqa)
    cap = rows(args.data / "eval_caption.jsonl", args.n_caption)
    t0 = time.time()
    hits, per_type = 0, {}
    for r in vqa:
        ok = vqa_correct(answer(backend, args.data, r, 16), r["answer"], r.get("type"))
        hits += ok
        t = per_type.setdefault(r.get("type") or "all", [0, 0])
        t[0] += ok
        t[1] += 1
    b4, rl = [], []
    for r in cap:
        pred = answer(backend, args.data, r, 96)
        b4.append(bleu(pred, [r["answer"]]).get("BLEU-4", 0.0))
        rl.append(rouge_l(pred, [r["answer"]]))
    result = {"base": args.base, "adapter": args.adapter, "n_vqa": len(vqa), "n_caption": len(cap),
              "vqa_accuracy": round(hits / max(len(vqa), 1), 4),
              "vqa_by_type": {k: round(c / n, 3) for k, (c, n) in sorted(per_type.items())},
              "caption_bleu4": round(sum(b4) / max(len(b4), 1), 4),
              "caption_rougeL": round(sum(rl) / max(len(rl), 1), 4),
              "minutes": round((time.time() - t0) / 60, 1)}
    if args.cpu_rows:
        cpu = HFBackend(args.base, args.adapter, device="cpu")
        sample = (vqa[: max(1, args.cpu_rows // 2)] + cap[: max(1, args.cpu_rows - args.cpu_rows // 2)])
        answer(cpu, args.data, sample[0], 8)                       # load once, outside the timing
        t = time.time()
        for r in sample:
            answer(cpu, args.data, r, 16 if r["task"] == "vqa" else 96)
        result["cpu_seconds_per_answer"] = round((time.time() - t) / len(sample), 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
