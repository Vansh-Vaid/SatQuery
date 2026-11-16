"""Should "where is the X?" questions be answered from the grounder's box instead of the VQA model?

On VRSBench's eval set, takes the object-position questions of the form "Where is the <object> located?"
whose answer is a place in the image (top-left ... center ... bottom-right), and scores on the same questions:
  * hr_vqa      - the VQA specialist's answer (what the app gives today)
  * grounder    - the 3x3 cell holding the centre of hr_grounder's box for "the <object>"
  * gated       - the grounder when its confidence is at least --gate, otherwise hr_vqa
with the same comparator as the VQA benchmark (metrics.vqa_correct).

  python eval/position_from_grounder.py --root data/vrsbench --n 600
  python eval/position_from_grounder.py --ground-ckpt checkpoints/vrsbench_ground_dense.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WHERE = re.compile(r"^(?:where is|where are|what is the (?:position|location) of) the (?P<obj>[a-z\- ]+?)"
                   r"(?: located| positioned| situated| placed)?(?: in the image| in this image)?\??$", re.I)
PLACE = re.compile(r"^(?:(?:top|bottom|middle)(?:[- ](?:left|right|middle))?|left|right|center|centre)"
                   r"(?: side)?$")


def grid_label(cx: float, cy: float) -> str:
    """Box centre (0-1) -> the dataset's 3x3 place words."""
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "middle"
    return "center" if row == col == "middle" else f"{row}-{col}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "data/vrsbench")
    ap.add_argument("--n", type=int, default=600, help="questions to score (random sample, fixed seed)")
    ap.add_argument("--ground-ckpt", default=None, help="grounder checkpoint (default: the app's)")
    ap.add_argument("--gate", type=float, default=0.5, help="grounder confidence needed to override hr_vqa")
    args = ap.parse_args()
    if args.ground_ckpt:
        os.environ["SATQUERY_VRSBENCH_GROUND_CKPT"] = str(args.ground_ckpt)

    from metrics import normalize_answer, vqa_correct

    from satquery.imagery.io import load_image
    from satquery.imagery.validate import validate_inputs
    from satquery.tools.base import ToolContext
    from satquery.tools.hrground import HighResGrounder
    from satquery.tools.hrvqa import HighResVQA

    rows = []
    for r in json.loads((args.root / "VRSBench_EVAL_vqa.json").read_text(encoding="utf-8")):
        m = WHERE.match(r["question"].strip())
        if r.get("type") == "object position" and m and PLACE.match(normalize_answer(r["ground_truth"])):
            rows.append({**r, "obj": m.group("obj").strip()})
    print(f"{len(rows)} 'where is the X' questions with a place answer", flush=True)
    rows = random.Random(0).sample(rows, min(args.n, len(rows)))

    vqa, grd = HighResVQA(), HighResGrounder()
    for tool in (vqa, grd):
        ok, why = tool.available()
        if not ok:
            sys.exit(f"{tool.spec.name} unavailable: {why}")
    hits = {"hr_vqa": 0, "grounder": 0, "gated": 0}
    n = used = 0
    for i, r in enumerate(rows):
        img = args.root / "Images_val" / r["image_id"]
        if not img.exists():
            continue
        rep = validate_inputs([load_image(img)], None)
        v = vqa.run(ToolContext(r["question"], rep.config, rep.rasters), vqa.defaults())
        g = grd.run(ToolContext(f"the {r['obj']}", rep.config, rep.rasters), grd.defaults())
        if v.status != "ok" or g.status != "ok":
            continue
        b = g.facts["box_norm"]
        place = grid_label((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        gt = r["ground_truth"]
        a_v, a_g = vqa_correct(v.facts["short_answer"], gt), vqa_correct(place, gt)
        use_g = (g.confidence or 0) >= args.gate
        hits["hr_vqa"] += a_v
        hits["grounder"] += a_g
        hits["gated"] += a_g if use_g else a_v
        used += use_g
        n += 1
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}: " + ", ".join(f"{k} {h / n:.1%}" for k, h in hits.items()), flush=True)
    if not n:
        sys.exit("no question could be scored")
    print(f"\n{n} questions scored (grounder used for {used} of them by the gate >= {args.gate})")
    print("| answer from | accuracy |\n|---|---|")
    for k, h in hits.items():
        print(f"| {k} | {h / n:.1%} |")


if __name__ == "__main__":
    main()
