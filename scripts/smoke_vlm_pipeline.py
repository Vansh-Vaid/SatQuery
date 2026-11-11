"""End-to-end smoke test of the VLM adaptation pipeline on CPU in ~1 minute.

Uses a tiny random-weight Qwen3-VL (same architecture/processor/chat template as
Qwen3-VL-2B) so it validates the *plumbing* - rendering, prompt templates, answer-only
loss masking, LoRA targeting, adapter save/load, generation, box parsing and the
rs_vlm tool inside the agent - without a GPU or a multi-GB download. It does not
measure accuracy (weights are random).

  python scripts/smoke_vlm_pipeline.py [--base trl-internal-testing/tiny-Qwen3VLForConditionalGeneration]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satquery.demo import synth as S  # noqa: E402
from satquery.imagery.raster import Modality  # noqa: E402
from satquery.models.vlm_prompts import build_prompt, render_for_vlm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="trl-internal-testing/tiny-Qwen3VLForConditionalGeneration")
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="satquery_smoke_"))
    (work / "images").mkdir()
    lab = S.base_layout(120, 120)
    render_for_vlm(S.render_s2(lab, seed=1)).save(work / "images/p_s2.png")
    render_for_vlm(S.render_s1(lab, seed=2)).save(work / "images/p_s1.png")
    samples = [
        ("vqa", "Is there any water body in the image?", "Yes", 2),
        ("caption", "", "The scene is dominated by agricultural land with a lake and a river.", 2),
        ("grounding", "the lake", json.dumps([{"bbox_2d": [90, 100, 410, 340], "label": "water"}]), 1),
    ]
    for split in ("train", "validation"):
        with (work / f"{split}.jsonl").open("w") as fh:
            for i, (task, q, a, n) in enumerate(samples * 2):
                mods = [Modality.S2_MS, Modality.SAR][:n]
                imgs = ["images/p_s2.png", "images/p_s1.png"][:n]
                fh.write(json.dumps({"id": f"{split}{i}", "images": imgs, "modalities": [m.value for m in mods],
                                     "prompt": build_prompt(task, q, mods, expression=q or None), "answer": a,
                                     "type": task, "category": task, "split": split}) + "\n")
    out = work / "adapter"
    cmd = [sys.executable, str(ROOT / "training/finetune_vlm.py"), "--data", str(work), "--out", str(out),
           "--base", args.base, "--max-steps", "3", "--grad-accum", "2", "--eval-every", "3", "--rank", "4"]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    assert (out / "adapter_config.json").exists(), "adapter not saved"

    from satquery.agent.controller import SatQueryAgent
    from satquery.agent.planner import RulePlanner
    from satquery.models.vlm_backend import HFBackend
    from satquery.tools.registry import default_registry
    from satquery.tools.vlm import RSVLM

    backend = HFBackend(args.base, str(out))
    text, conf = backend.generate([render_for_vlm(S.render_s2(lab))], build_prompt("vqa", "Is there water?",
                                                                                    [Modality.S2_MS]), 12)
    print(f"generation OK: {ascii(text[:60])} (confidence {conf})")   # ascii: random tokens, any console
    agent = SatQueryAgent(default_registry(RSVLM(backend=backend)), planner=RulePlanner())
    r = agent.run("Use the optical and SAR images together to identify built-up and water-covered regions.",
                  [S.render_s2(lab, acquired=date(2024, 6, 1)), S.render_s1(lab, acquired=date(2024, 6, 2))])
    step = next(s for s in r.trace["steps"] if s["tool"] == "rs_vlm")
    print("agent rs_vlm step:", step["status"], step["facts"].get("backend"))
    assert step["status"] == "ok"
    print(f"SMOKE TEST PASSED (artifacts in {work})")


if __name__ == "__main__":
    main()
