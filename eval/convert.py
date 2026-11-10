"""Convert benchmark datasets into SatQuery's common evaluation JSONL.

Row schema: {"id", "images": [path, ...], "modalities": [...] | null, "query", "answer",
             "task": vqa|caption|grounding|change_vqa, "qtype", "bbox": [x0,y0,x1,y1] px | null}

  python eval/convert.py rsvqa    --root /data/RSVQA_LR --split test          --out eval/data/rsvqa_lr_test.jsonl
  python eval/convert.py vrsbench --root /data/VRSBench                         --out eval/data/vrsbench.jsonl
  python eval/convert.py cdvqa    --questions /data/CDVQA/Test.json --im1 /data/CDVQA/im1 --im2 /data/CDVQA/im2 \
                                  --out eval/data/cdvqa_test.jsonl
  python eval/convert.py bentxt   --root data/bentxt --split bench             --out eval/data/bentxt_bench.jsonl

Field names follow the public releases (RSVQA: Lobry et al. 2020; VRSBench: Li et al. 2024; CDVQA:
Yuan et al. 2022). CDVQA ships in more than one JSON layout, so its keys are configurable; run with
``--peek`` first to print the first record and confirm the mapping.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def rsvqa(root: Path, split: str, res: str = "LR") -> list[dict]:
    q = json.loads((root / f"{res}_split_{split}_questions.json").read_text())["questions"]
    a = {x["id"]: x for x in json.loads((root / f"{res}_split_{split}_answers.json").read_text())["answers"]}
    img_dir = next((root / d for d in (f"Images_{res}", "Images", "images") if (root / d).exists()), root)
    rows = []
    for x in q:
        if not x.get("active", True) or not x.get("answers_ids"):
            continue
        ans = a.get(x["answers_ids"][0])
        if not ans or not ans.get("active", True):
            continue
        img = next((img_dir / f"{x['img_id']}{e}" for e in (".tif", ".png", ".jpg") if (img_dir / f"{x['img_id']}{e}").exists()),
                   img_dir / f"{x['img_id']}.tif")
        rows.append({"id": f"rsvqa-{x['id']}", "images": [str(img)], "modalities": ["optical_rgb"],
                     "query": x["question"], "answer": ans["answer"], "task": "vqa", "qtype": x.get("type", "vqa"),
                     "bbox": None})
    return rows


def _scale_box(b: list[float], w: int, h: int) -> list[float]:
    mx = max(b)
    if mx <= 1.0:
        return [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    if mx <= 100.0 and max(w, h) > 100:
        return [b[0] / 100 * w, b[1] / 100 * h, b[2] / 100 * w, b[3] / 100 * h]
    return list(b)


def vrsbench(root: Path) -> list[dict]:
    """VRSBench eval files. Images are extracted once from Images_val.zip if needed.

    Referring answers are written ``{<x1><y1><x2><y2>}`` on a 0-100 scale of the image (verified against
    VRSBench_EVAL_referring.json, 2026-11-09); they are converted to pixel boxes.
    """
    import zipfile

    from PIL import Image
    img_dir = next((root / d for d in ("Images_val", "images_val", "Images", "images") if (root / d).exists()), None)
    if img_dir is None and (root / "Images_val.zip").exists():
        print("extracting Images_val.zip ...")
        zipfile.ZipFile(root / "Images_val.zip").extractall(root)
        img_dir = root / "Images_val"
    img_dir = img_dir or root
    rows = []
    for name, task in (("VRSBench_EVAL_vqa.json", "vqa"), ("VRSBench_EVAL_Cap.json", "caption"),
                       ("VRSBench_EVAL_referring.json", "grounding")):
        f = root / name
        if not f.exists():
            print(f"missing {f}, skipped")
            continue
        for i, x in enumerate(json.loads(f.read_text())):
            img = img_dir / x["image_id"]
            gt = x.get("ground_truth")
            row = {"id": f"vrs-{task}-{x.get('question_id', i)}", "images": [str(img)], "modalities": ["optical_rgb"],
                   "query": x.get("question") or "Describe the image in detail.", "answer": gt, "task": task,
                   "qtype": x.get("type", task), "bbox": None}
            if task == "grounding":
                box = gt if isinstance(gt, list) else [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", str(gt))]
                if len(box) < 4:
                    continue
                with Image.open(img) as im:
                    w, h = im.size
                row["bbox"] = [box[0] / 100 * w, box[1] / 100 * h, box[2] / 100 * w, box[3] / 100 * h]
                row["answer"] = None
                row["obj_cls"] = x.get("obj_cls")
                row["query"] = "Locate " + row["query"][0].lower() + row["query"][1:]   # VRSBench's [refer] task tag
            rows.append(row)
    return rows


def cdvqa(questions: Path, im1: Path, im2: Path, k_img: str, k_q: str, k_a: str, k_type: str,
          list_key: str | None) -> list[dict]:
    data = json.loads(questions.read_text())
    items = data[list_key] if list_key else (data if isinstance(data, list) else next(
        v for v in data.values() if isinstance(v, list)))
    rows = []
    for i, x in enumerate(items):
        name = str(x[k_img])
        if not Path(name).suffix:
            name += ".png"
        rows.append({"id": f"cdvqa-{x.get('id', i)}", "images": [str(im1 / name), str(im2 / name)],
                     "modalities": ["optical_rgb", "optical_rgb"], "query": x[k_q], "answer": x[k_a],
                     "task": "change_vqa", "qtype": x.get(k_type, "change"), "bbox": None})
    return rows


def bentxt(root: Path, split: str) -> list[dict]:
    from satquery.models.vlm_prompts import parse_boxes
    rows = []
    for line in (root / f"{split}.jsonl").open(encoding="utf-8"):
        x = json.loads(line)
        task = {"binary": "vqa", "mcq": "vqa", "captioning": "caption", "bounding box": "grounding"}[x["type"]]
        # keep the raw instruction (strip SatQuery's header lines) so the agent sees a natural query
        query = x["prompt"].split("\n", len(x["images"]))[-1]
        query = query.replace("\nAnswer concisely.", "")
        row = {"id": f"ben-{x['id']}", "images": [str(root / p) for p in x["images"]],
               "modalities": x["modalities"], "query": query, "answer": x["answer"], "task": task,
               "qtype": x["category"], "bbox": None, "prebuilt_prompt": x["prompt"]}
        if task == "grounding":
            b = parse_boxes(x["answer"])
            row["bbox"] = [v / 1000 * 120 for v in b[0]["bbox_2d"]] if b else None
            row["answer"] = None
        rows.append(row)
    return rows


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="ds", required=True)
    p = sub.add_parser("rsvqa"); p.add_argument("--root", type=Path, required=True); p.add_argument("--split", default="test")
    p.add_argument("--res", default="LR", choices=["LR", "HR"]); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("vrsbench"); p.add_argument("--root", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("cdvqa")
    p.add_argument("--questions", type=Path, required=True); p.add_argument("--im1", type=Path, required=True)
    p.add_argument("--im2", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key-image", default="image"); p.add_argument("--key-question", default="question")
    p.add_argument("--key-answer", default="answer"); p.add_argument("--key-type", default="type")
    p.add_argument("--list-key", default=None); p.add_argument("--peek", action="store_true")
    p = sub.add_parser("bentxt"); p.add_argument("--root", type=Path, required=True); p.add_argument("--split", default="bench")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.ds == "rsvqa":
        rows = rsvqa(a.root, a.split, a.res)
    elif a.ds == "vrsbench":
        rows = vrsbench(a.root)
    elif a.ds == "cdvqa":
        if a.peek:
            d = json.loads(a.questions.read_text())
            first = d[0] if isinstance(d, list) else {k: (v[0] if isinstance(v, list) else v) for k, v in d.items()}
            print(json.dumps(first, indent=2)[:2000])
            return
        rows = cdvqa(a.questions, a.im1, a.im2, a.key_image, a.key_question, a.key_answer, a.key_type, a.list_key)
    else:
        rows = bentxt(a.root, a.split)
    _write(rows, a.out)


if __name__ == "__main__":
    main()
