"""Fetch a reproducible subset of CDVQA (Yuan et al., TGRS 2022) from its Hugging Face WebDataset mirror.

The mirror (ljx620/CDVQA) stores every question with its own copy of both images (~80 GB in total), so
this script streams shards without saving them, keeps each image pair once, and writes eval rows.
Shards are taken evenly spaced across the split, so the subset spans the whole set of pairs.

  python eval/fetch_cdvqa.py --split test  --shards 60  --out data/cdvqa      # ~180 pairs, ~6k questions
  python eval/fetch_cdvqa.py --split train --shards 150 --out data/cdvqa

Writes data/cdvqa/<split>/<pair>_t1.png, _t2.png and eval/data/cdvqa_<split>.jsonl
(rows: id, task=change_vqa, qtype, query, answer, images, pair).
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "https://huggingface.co/datasets/ljx620/CDVQA"
PROMPT = re.compile(r"^(Image \d: <image>\s*)+", re.I)


def shard_list(split: str) -> list[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/ljx620/CDVQA/tree/main/{split}") as r:
        files = json.load(r)
    return sorted(f["path"] for f in files if f["path"].endswith(".tar"))


def fetch_shard(path: str, img_dir: Path, tries: int = 3) -> list[dict]:
    """Rows of one shard. Cached per shard, so an interrupted run resumes; a shard that keeps failing is skipped."""
    cache = img_dir / "_rows" / (Path(path).stem + ".jsonl")
    if cache.exists():
        return [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
    for attempt in range(1, tries + 1):
        try:
            rows = _download(path, img_dir)
            cache.parent.mkdir(exist_ok=True)
            cache.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return rows
        except (OSError, tarfile.TarError) as e:   # truncated stream, reset connection, HTTP error
            print(f"  {path}: attempt {attempt} failed ({e})", flush=True)
    print(f"  {path}: skipped after {tries} attempts", flush=True)
    return []


def _download(path: str, img_dir: Path) -> list[dict]:
    rows, pending = [], {}
    with urllib.request.urlopen(f"{REPO}/resolve/main/{path}", timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|") as tar:
            for m in tar:
                key, _, ext = m.name.partition(".")
                pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
    for key, parts in sorted(pending.items()):
        meta = json.loads(parts["json"])
        pair = Path(meta["meta"]["image_id"]).stem
        ims = [img_dir / f"{pair}_t1.png", img_dir / f"{pair}_t2.png"]
        for p, ext in zip(ims, ("0.img", "1.img")):
            if not p.exists():
                p.write_bytes(parts[ext])
        conv = meta["conversations"]
        rows.append({"id": key, "task": "change_vqa", "qtype": meta["meta"]["question_type"],
                     "query": PROMPT.sub("", conv[0]["value"]).strip(), "answer": conv[1]["value"].strip(),
                     "images": [str(p) for p in ims], "pair": pair})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("data/cdvqa"))
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    allp = shard_list(a.split)
    step = max(1, len(allp) // a.shards)
    pick = allp[::step][: a.shards]
    img_dir = a.out / a.split
    img_dir.mkdir(parents=True, exist_ok=True)
    out = a.jsonl or Path(f"eval/data/cdvqa_{a.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{a.split}: {len(pick)} of {len(allp)} shards", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(lambda p: fetch_shard(p, img_dir), pick)):
            rows += r
            if k % 10 == 0:
                print(f"  {k + 1}/{len(pick)} shards, {len(rows)} questions", flush=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} questions over {len({r['pair'] for r in rows})} pairs -> {out}")


if __name__ == "__main__":
    main()
