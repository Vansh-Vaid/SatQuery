"""Run SatQuery on an evaluation JSONL (see eval/convert.py) and report metrics.

  # full agentic pipeline (planner + tools + integration), as a user would get it:
  python eval/run_eval.py --data eval/data/rsvqa_lr_test.jsonl --mode agent --limit 2000 --out eval/results/rsvqa
  # VLM tool alone (ablation / model comparison: base vs LoRA):
  python eval/run_eval.py --data eval/data/bentxt_bench.jsonl --mode vlm --out eval/results/bench_lora

Outputs: predictions.jsonl, metrics.json, summary.md in --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import bleu, box_iou, cider, normalize_answer, rouge_l, vqa_correct  # noqa: E402


def load(row: dict):
    from satquery.imagery.io import load_image
    mods = row.get("modalities") or [None] * len(row["images"])
    return [load_image(p, m) for p, m in zip(row["images"], mods)]


def predict_agent(agent, row):
    resp = agent.run(row["query"], load(row), "bitemporal" if row["task"] == "change_vqa" else None)
    boxes = [[b["x0"], b["y0"], b["x1"], b["y1"]] for b in resp.boxes]
    return {"pred": resp.short_answer or resp.answer, "full": resp.answer, "boxes": boxes,
            "confidence": resp.confidence, "tools": [s["tool"] for s in resp.trace.get("steps", [])]}


def predict_vlm(vlm, row):
    from satquery.imagery.validate import validate_inputs
    from satquery.tools.base import ToolContext
    rasters = load(row)
    rep = validate_inputs(rasters, "bitemporal" if row["task"] == "change_vqa" else None)
    mode = {"vqa": "vqa", "caption": "caption", "grounding": "grounding"}.get(row["task"], "vqa")
    res = vlm.run(ToolContext(row["query"], rep.config, rep.rasters), {**vlm.defaults(), "mode": mode})
    return {"pred": res.facts.get("short_answer") or res.answer or "", "full": res.answer,
            "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in res.boxes], "confidence": res.confidence,
            "tools": ["rs_vlm"], "status": res.status}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mode", choices=["agent", "vlm"], default="agent")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.data.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "agent":
        from satquery.agent.controller import SatQueryAgent
        runner, fn = SatQueryAgent(), predict_agent
    else:
        from satquery.tools.vlm import RSVLM
        runner, fn = RSVLM(), predict_vlm
        ok, why = runner.available()
        if not ok:
            sys.exit(f"VLM unavailable: {why}")

    acc = defaultdict(lambda: [0, 0])
    caps_p, caps_r, ious = [], [], defaultdict(list)
    t0 = time.time()
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            try:
                p = fn(runner, row)
            except Exception as e:  # keep going; count as wrong
                p = {"pred": "", "full": f"ERROR {type(e).__name__}: {e}", "boxes": [], "confidence": None, "tools": []}
            rec = {"id": row["id"], "task": row["task"], "qtype": row.get("qtype"), "query": row["query"],
                   "answer": row.get("answer"), **p}
            if row["task"] in ("vqa", "change_vqa"):
                ok = vqa_correct(p["pred"], row["answer"], row.get("qtype"))
                rec["correct"] = ok
                for k in ("overall", f"{row['task']}/{row.get('qtype')}"):
                    acc[k][0] += int(ok)
                    acc[k][1] += 1
            elif row["task"] == "caption":
                refs = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
                caps_p.append(p["full"] or "")
                caps_r.append(refs)
            elif row["task"] == "grounding" and row.get("bbox"):
                best = max((box_iou(b, row["bbox"]) for b in p["boxes"]), default=0.0)
                rec["iou"] = best
                ious[row.get("qtype") or "grounding"].append(best)
                ious["overall"].append(best)
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)}  {(time.time() - t0) / (i + 1):.2f}s/sample", flush=True)

    metrics: dict = {"n": len(rows), "mode": args.mode, "data": str(args.data)}
    if acc:
        metrics["vqa_accuracy"] = {k: {"acc": round(c / n, 4), "n": n} for k, (c, n) in sorted(acc.items())}
    if caps_p:
        b = defaultdict(float)
        for p, r in zip(caps_p, caps_r):
            for k, v in bleu(p, r).items():
                b[k] += v
        metrics["caption"] = {k: round(v / len(caps_p), 4) for k, v in b.items()}
        metrics["caption"]["ROUGE-L"] = round(sum(rouge_l(p, r) for p, r in zip(caps_p, caps_r)) / len(caps_p), 4)
        c = cider(caps_p, caps_r)
        if c is not None:
            metrics["caption"]["CIDEr"] = round(c, 4)
    if ious:
        metrics["grounding"] = {k: {"mIoU": round(sum(v) / len(v), 4), "Acc@0.5": round(sum(x >= 0.5 for x in v) / len(v), 4),
                                    "n": len(v)} for k, v in ious.items()}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = [f"# SatQuery evaluation ({args.mode})", "", f"data: `{args.data}` · n = {len(rows)}", ""]
    for sect, vals in metrics.items():
        if isinstance(vals, dict):
            lines += [f"## {sect}", "", "| key | value |", "|---|---|"]
            lines += [f"| {k} | {json.dumps(v)} |" for k, v in vals.items()]
            lines.append("")
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
