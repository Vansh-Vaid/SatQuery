"""Small VLM path (SmolVLM): processor options and the VRSBench chat-row preparation, without downloading a model."""
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_processor_kwargs_cap_image_size_only_for_small_vlms():
    from satquery.models.vlm_backend import processor_kwargs
    assert processor_kwargs("HuggingFaceTB/SmolVLM-256M-Instruct", 448) == {"size": {"longest_edge": 448}}
    assert processor_kwargs("Qwen/Qwen3-VL-2B-Instruct") == {}


def _fake_vrsbench(root: Path, n=40):
    from PIL import Image
    root.mkdir(parents=True)
    train, ev_vqa, ev_cap = [], [], []
    for zname, prefix in (("Images_train.zip", "tr"), ("Images_val.zip", "va")):
        with zipfile.ZipFile(root / zname, "w") as z:
            for k in range(n // 4):
                buf = io.BytesIO()
                Image.new("RGB", (32, 32), (k * 6 % 255, 90, 40)).save(buf, "PNG")
                z.writestr(f"Images/{prefix}_{k:03d}.png", buf.getvalue())
    for k in range(n):
        img = f"tr_{k % (n // 4):03d}.png"
        tag = "vqa" if k % 3 else "caption"
        q = "How many ships are there?" if tag == "vqa" else "Describe this image."
        train.append({"image": img, "conversations": [{"from": "human", "value": f"<image>\n[{tag}] {q}"},
                                                      {"from": "gpt", "value": "2" if tag == "vqa" else "A harbour."}]})
    train.append({"image": "tr_000.png", "conversations": [{"from": "human", "value": "<image>\n[refer] the ship"},
                                                           {"from": "gpt", "value": "{<1><2><3><4>}"}]})
    for k in range(n // 4):
        ev_vqa.append({"image_id": f"va_{k:03d}.png", "question": "Is there a ship?", "ground_truth": "yes",
                       "question_id": k, "type": "object existence"})
        ev_cap.append({"image_id": f"va_{k:03d}.png", "ground_truth": "A harbour with ships."})
    (root / "VRSBench_train.json").write_text(json.dumps(train))
    (root / "VRSBench_EVAL_vqa.json").write_text(json.dumps(ev_vqa))
    (root / "VRSBench_EVAL_Cap.json").write_text(json.dumps(ev_cap))


def test_prepare_writes_app_prompts_and_keeps_validation_images_separate(tmp_path, monkeypatch):
    from satquery.imagery.raster import Modality
    from satquery.models.vlm_prompts import build_prompt
    _fake_vrsbench(tmp_path / "vrs")
    out = tmp_path / "rows"
    monkeypatch.setattr(sys, "argv", ["prepare_vrsbench_vlm.py", "--root", str(tmp_path / "vrs"), "--out", str(out),
                                      "--val-frac", "0.2", "--n-eval-vqa", "5", "--n-eval-caption", "3"])
    sys.path.insert(0, str(ROOT / "training"))
    import prepare_vrsbench_vlm
    prepare_vrsbench_vlm.main()
    read = lambda f: [json.loads(x) for x in (out / f).read_text().splitlines() if x]  # noqa: E731
    train, val, ev = read("train.jsonl"), read("validation.jsonl"), read("eval_vqa.jsonl")
    assert train and val and len(ev) == 5 and len(read("eval_caption.jsonl")) == 3
    assert all(r["task"] in ("vqa", "caption") for r in train)                      # [refer] rows are left out
    assert {r["images"][0] for r in train}.isdisjoint({r["images"][0] for r in val})
    vqa = next(r for r in train if r["task"] == "vqa")
    assert vqa["prompt"] == build_prompt("vqa", "How many ships are there?", [Modality.OPTICAL_RGB])
    assert all((out / r["images"][0]).exists() for r in train + val + ev)
