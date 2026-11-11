"""Confidence calibration of saved evaluation runs: ECE, accuracy per label, reliability diagram.

  python eval/calibration_report.py eval/results/rsvqa_lr_agent_v3 eval/results/cdvqa_agent \
      eval/results/vrs_agent_vqa_v2 --names RSVQA-LR CDVQA VRSBench --plot docs/images/calibration.png

Reads predictions.jsonl (written by run_eval.py) from each directory; only answers with a confidence and
a right/wrong score are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import calibration  # noqa: E402


def load(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "predictions.jsonl").open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("confidence") is not None and "correct" in r]
    return calibration([r["confidence"] for r in rows], [r["correct"] for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--names", nargs="*", help="display name per run (default: directory name)")
    ap.add_argument("--plot", type=Path, help="write a reliability diagram PNG here")
    args = ap.parse_args()
    names = args.names or [r.name for r in args.runs]
    if len(names) != len(args.runs):
        ap.error("--names needs one name per run")

    results = {n: load(r) for n, r in zip(names, args.runs)}
    print("| benchmark | n | accuracy | mean confidence | ECE | high | medium | low |")
    print("|---|---|---|---|---|---|---|---|")
    for n, m in results.items():
        lab = m["by_label"]
        cell = lambda k: f"{lab[k]['accuracy']:.1%} (n={lab[k]['n']})" if k in lab else "-"  # noqa: E731
        print(f"| {n} | {m['n']} | {m['accuracy']:.1%} | {m['mean_confidence']:.1%} | {m['ECE']:.3f} | "
              f"{cell('high')} | {cell('medium')} | {cell('low')} |")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
        ax.plot([0, 1], [0, 1], ls="--", c="#888", lw=1, label="perfect calibration")
        for (n, m), mk in zip(results.items(), "osd^v"):
            xs = [b["confidence"] for b in m["reliability"] if b["n"] >= 10]
            ys = [b["accuracy"] for b in m["reliability"] if b["n"] >= 10]
            ax.plot(xs, ys, marker=mk, lw=1.5, label=f"{n} (ECE {m['ECE']:.3f})")
        ax.set_xlabel("stated confidence (bin mean)")
        ax.set_ylabel("observed accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("SatQuery confidence vs accuracy")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
