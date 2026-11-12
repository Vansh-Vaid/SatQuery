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

from metrics import bleu, box_iou, calibration, cider, normalize_answer, rouge_l, vqa_correct  # noqa: E402


def load(row: dict):
    from satquery.imagery.io import load_image
    mods = row.get("modalities") or [None] * len(row["images"])
    return [load_image(p, m) for p, m in zip(row["images"], mods)]


def predict_agent(agent, row):
    resp = agent.run(row["query"], load(row), "bitemporal" if row["task"] == "change_vqa" else None)
    boxes = [[b["x0"], b["y0"], b["x1"], b["y1"]] for b in resp.boxes]
    # a caption is the whole answer; for questions the short answer is what gets scored
    pred = resp.answer if row["task"] == "caption" else (resp.short_answer or resp.answer)
    return {"pred": pred, "full": resp.answer, "boxes": boxes,
            "confidence": resp.confidence, "tools": [s["tool"] for s in resp.trace.get("steps", [])]}


def predict_vlm(vlm, row):
    from satquery.imagery.validate import validate_inputs
    from satquery.tools.base import ToolContext
    rasters = load(row)
    rep = validate_inputs(rasters, "bitemporal" if row["task"] == "change_vqa" else None)
    mode = {"vqa": "vqa", "caption": "caption", "grounding": "grounding"}.get(row["task"], "vqa")
    # rows converted from a training-format file carry the exact prompt the model was trained with
    params = {**vlm.defaults(), "mode": mode, "prompt": row.get("prebuilt_prompt")}
    res = vlm.run(ToolContext(row["query"], rep.config, rep.rasters), params)
    return {"pred": res.facts.get("short_answer") or res.answer or "", "full": res.answer,
            "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in res.boxes], "confidence": res.confidence,
            "tools": ["rs_vlm"], "status": res.status,
            **({"error": "; ".join(res.notes) or "tool returned an error"} if res.status == "error" else {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mode", choices=["agent", "vlm"], default="agent")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit-min", type=float, default=0,
                    help="stop after this many minutes and score the examples done so far (0 = no limit)")
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
    cal_c, cal_y = [], []
    errors: list[str] = []
    t0 = time.time()
    done = 0
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            if args.time_limit_min and (time.time() - t0) / 60 >= args.time_limit_min:
                print(f"time limit of {args.time_limit_min:g} min reached after {i}/{len(rows)} examples; "
                      "scoring those", flush=True)
                break
            done = i + 1
            try:
                p = fn(runner, row)
            except Exception as e:  # keep going; count as wrong, but record why
                p = {"pred": "", "full": None, "boxes": [], "confidence": None, "tools": [], "status": "error",
                     "error": f"{type(e).__name__}: {e}"}
            if p.get("status") == "error":
                errors.append(p.get("error", ""))
            rec = {"id": row["id"], "task": row["task"], "qtype": row.get("qtype"), "query": row["query"],
                   "answer": row.get("answer"), **p}
            if row["task"] in ("vqa", "change_vqa"):
                ok = vqa_correct(p["pred"], row["answer"], row.get("qtype"))
                rec["correct"] = ok
                if p.get("confidence") is not None:
                    cal_c.append(p["confidence"])
                    cal_y.append(ok)
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

    metrics: dict = {"n": done, "mode": args.mode, "data": str(args.data), "errors": len(errors)}
    if errors:
        metrics["first_error"] = errors[0][:500]
    if done < len(rows):
        metrics["stopped_early"] = f"time limit: scored {done} of {len(rows)} examples"
    if acc:
        metrics["vqa_accuracy"] = {k: {"acc": round(c / n, 4), "n": n} for k, (c, n) in sorted(acc.items())}
    if cal_c:
        metrics["calibration"] = calibration(cal_c, cal_y)
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
    lines = [f"# SatQuery evaluation ({args.mode})", "", f"data: `{args.data}` · n = {done}", ""]
    for sect, vals in metrics.items():
        if isinstance(vals, dict):
            lines += [f"## {sect}", "", "| key | value |", "|---|---|"]
            lines += [f"| {k} | {json.dumps(v)} |" for k, v in vals.items()]
            lines.append("")
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if errors:
        print(f"WARNING: {len(errors)} of {done} predictions failed and were scored as wrong. First error: "
              f"{errors[0][:500]}", file=sys.stderr)
    if done and len(errors) == done:
        sys.exit("every prediction failed, so these scores do not measure the model (see the error above)")


if __name__ == "__main__":
    main()
