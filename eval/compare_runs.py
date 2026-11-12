"""Compare evaluation runs on the examples they all scored (runs cut short by a time limit score
different numbers of examples, so their own summaries are not comparable).

  python eval/compare_runs.py results/bench_base results/bench_lora --names base LoRA

VQA answers are re-scored with the current `metrics.vqa_correct`; grounding uses each run's stored IoU.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import bleu, rouge_l, vqa_correct  # noqa: E402


def load(run: Path) -> dict[str, dict]:
    with (run / "predictions.jsonl").open(encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, fh) if r.get("id")}


def score(rows: list[dict]) -> dict[str, str]:
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    ious: dict[str, list[float]] = defaultdict(list)
    caps: list[tuple[str, list[str]]] = []
    for r in rows:
        if r["task"] in ("vqa", "change_vqa"):
            ok = vqa_correct(r.get("pred") or "", r["answer"], r.get("qtype"))
            for k in ("VQA overall", f"VQA {r.get('qtype')}"):
                acc[k][0] += int(ok)
                acc[k][1] += 1
        elif r["task"] == "grounding" and r.get("iou") is not None:
            for k in ("grounding overall", f"grounding {r.get('qtype')}"):
                ious[k].append(float(r["iou"]))
        elif r["task"] == "caption":
            refs = r["answer"] if isinstance(r["answer"], list) else [r["answer"] or ""]
            caps.append((r.get("full") or "", refs))
    out = {k: f"{c / n:.1%} (n={n})" for k, (c, n) in sorted(acc.items())}
    for k, v in sorted(ious.items()):
        out[k] = f"Acc@0.5 {sum(x >= 0.5 for x in v) / len(v):.1%}, mIoU {sum(v) / len(v):.3f} (n={len(v)})"
    if caps:
        out["caption BLEU-4"] = f"{sum(bleu(p, r)['BLEU-4'] for p, r in caps) / len(caps):.3f} (n={len(caps)})"
        out["caption ROUGE-L"] = f"{sum(rouge_l(p, r) for p, r in caps) / len(caps):.3f} (n={len(caps)})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run folders with predictions.jsonl")
    ap.add_argument("--names", nargs="*", help="column names (default: folder names)")
    args = ap.parse_args()
    names = args.names or [p.name for p in args.runs]
    preds = [load(p) for p in args.runs]
    shared = sorted(set.intersection(*(set(p) for p in preds)))
    if not shared:
        sys.exit("the runs have no examples in common")
    tables = [score([p[i] for i in shared]) for p in preds]
    keys = sorted({k for t in tables for k in t}, key=lambda k: (k.split()[0], "overall" not in k, k))
    print(f"{len(shared)} examples scored by every run\n")
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(t.get(k, "-") for t in tables) + " |")


if __name__ == "__main__":
    main()
