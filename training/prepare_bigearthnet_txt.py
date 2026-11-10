"""Prepare BigEarthNet.txt for VLM fine-tuning (instruction JSONL + rendered images).

Inputs
  * BigEarthNet.txt.parquet  (HF: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt; columns
    ID, s1_name, patch_id, input, output, type, category, split, ...)
  * BigEarthNet v2.0 imagery, either
      --s2-root/--s1-root : the extracted GeoTIFF folders
                           (<root>/<tile>/<patch_id>/<patch_id>_B02.tif, ..._VV.tif)
      --lmdb              : the rico-hdl encoded LMDB (key -> safetensors band dict)
      --rendered          : PNGs already rendered into <out>/images by
                            training/fetch_bigearthnet_subset.py (streams only the needed
                            patches from Zenodo; the route used on Kaggle)

Output (--out)
  images/<patch_id>_s2.png, images/<patch_id>_s1.png   rendered exactly like inference
  {split}.jsonl  rows: {"id","images":[...],"modalities":[...],"prompt","answer","type","category","split"}

Sampling: the full set has 9.6 M annotations; ``--per-type`` caps each (split, type)
so a LoRA run fits a single GPU budget (default 6k/type -> ~24k train samples).
A fraction of samples (--single-frac) keeps only the S2 image so the model also
learns single-image operation; the rest present the co-registered S2+S1 pair.

Example (Kaggle/Colab):
  python training/prepare_bigearthnet_txt.py --s2-root /data/BigEarthNet-S2 --s1-root /data/BigEarthNet-S1 \
      --out data/bentxt --per-type 6000 --splits train validation bench
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.imagery.raster import Modality, Raster  # noqa: E402
from satquery.models.vlm_prompts import build_prompt, render_for_vlm  # noqa: E402

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
PATCH = 120
TYPE_TO_TASK = {"binary": "vqa", "mcq": "vqa", "captioning": "caption", "bounding box": "grounding"}


# ----------------------------------------------------------------- image access
class FolderReader:
    def __init__(self, s2_root: Path, s1_root: Path | None):
        self.s2 = {p.name: p for p in s2_root.glob("*/*") if p.is_dir()}
        self.s1 = {p.name: p for p in s1_root.glob("*/*") if p.is_dir()} if s1_root else {}
        print(f"indexed {len(self.s2)} S2 and {len(self.s1)} S1 patch folders")

    @staticmethod
    def _read(d: Path | None, stem: str, bands) -> dict[str, np.ndarray] | None:
        import rasterio
        if d is None:
            return None
        out = {}
        for b in bands:
            f = d / f"{stem}_{b}.tif"
            if f.exists():
                with rasterio.open(f) as src:
                    out[b] = src.read(1, out_shape=(PATCH, PATCH)).astype(np.float32)
        return out or None

    def s2_bands(self, patch_id: str):
        return self._read(self.s2.get(patch_id), patch_id, S2_BANDS)

    def s1_bands(self, s1_name: str):
        return self._read(self.s1.get(s1_name), s1_name, ("VV", "VH"))


class LMDBReader:
    def __init__(self, path: Path):
        import lmdb
        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)

    def _get(self, key: str) -> dict[str, np.ndarray] | None:
        from safetensors.numpy import load
        with self.env.begin() as txn:
            raw = txn.get(key.encode())
        if raw is None:
            return None
        out = {}
        for k, v in load(raw).items():
            v = v.astype(np.float32)
            if v.shape != (PATCH, PATCH):
                import cv2
                v = cv2.resize(v, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
            out[k.upper()] = v
        return out

    def s2_bands(self, patch_id: str):
        return self._get(patch_id)

    def s1_bands(self, s1_name: str):
        return self._get(s1_name)


def to_s2_raster(b: dict[str, np.ndarray]) -> Raster:
    names = [n for n in S2_BANDS if n in b]
    data = np.stack([b[n] for n in names]) / 10000.0
    return Raster(data=np.clip(data, 0, 1.5), bands=names, modality=Modality.S2_MS, source_format="geotiff")


def to_s1_raster(b: dict[str, np.ndarray]) -> Raster:
    names = [n for n in ("VV", "VH") if n in b]
    data = np.stack([b[n] for n in names])
    if np.nanmin(data) >= 0 and np.nanmedian(data) < 2:  # linear -> dB
        data = 10 * np.log10(np.maximum(data, 1e-6))
    return Raster(data=data, bands=names, modality=Modality.SAR, source_format="geotiff")


# ----------------------------------------------------------------- targets
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def grounding_target(output: str, label_hint: str = "") -> tuple[str, str]:
    """Convert a BigEarthNet.txt bounding-box answer into SatQuery's JSON 0-1000 convention.

    The coordinate frame is inferred per answer (normalised [0,1], pixel [0,120], or [0,1000]);
    the returned tag is counted and printed so the inference can be audited.
    """
    nums = [float(x) for x in _NUM.findall(output)]
    if len(nums) < 4 or len(nums) % 4:
        return output.strip(), "unparsed"
    boxes = [nums[i:i + 4] for i in range(0, len(nums), 4)]
    mx = max(nums)
    if mx <= 1.0:
        scale, tag = 1000.0, "normalised"
    elif mx <= PATCH:
        scale, tag = 1000.0 / PATCH, "pixel120"
    else:
        scale, tag = 1.0, "per-mille"
    out = [{"bbox_2d": [round(v * scale) for v in b], "label": label_hint} for b in boxes]
    return json.dumps(out), tag


COLUMNS = ["ID", "s1_name", "patch_id", "input", "output", "type", "category", "split"]


def load_annotations(parquet: str | None):
    import pandas as pd
    if parquet is None:
        from huggingface_hub import hf_hub_download
        parquet = hf_hub_download("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", "BigEarthNet.txt.parquet", repo_type="dataset")
    return pd.read_parquet(parquet, columns=COLUMNS)


def select_rows(df, split: str, per_type: int, bench_per_type: int, seed: int):
    """The annotations used for one split. Shared with fetch_bigearthnet_subset.py, so both pick the same rows."""
    part = df[df["split"] == split]
    cap = bench_per_type if split == "bench" else per_type
    if cap and len(part):
        import pandas as pd
        part = pd.concat([g.sample(min(len(g), cap), random_state=seed) for _, g in part.groupby("type")])
    return part


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=None, help="local BigEarthNet.txt.parquet (downloaded from HF if omitted)")
    ap.add_argument("--s2-root", type=Path)
    ap.add_argument("--s1-root", type=Path)
    ap.add_argument("--lmdb", type=Path)
    ap.add_argument("--rendered", action="store_true", help="use PNGs already in <out>/images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "bench"])
    ap.add_argument("--per-type", type=int, default=6000, help="max samples per (split, type), except bench")
    ap.add_argument("--bench-per-type", type=int, default=0, help="max bench samples per type (0 = keep bench whole)")
    ap.add_argument("--single-frac", type=float, default=0.3, help="fraction of samples with the S2 image only")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if not (args.lmdb or args.s2_root or args.rendered):
        ap.error("give --rendered, --lmdb or --s2-root (and optionally --s1-root)")

    df = load_annotations(args.parquet)
    print(df.groupby(["split", "type"]).size().to_string())
    reader = None if args.rendered else LMDBReader(args.lmdb) if args.lmdb else FolderReader(args.s2_root, args.s1_root)
    rng = random.Random(args.seed)
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, tuple[str | None, str | None]] = {}
    frames: Counter = Counter()

    def images_for(patch_id: str, s1_name: str) -> tuple[str | None, str | None]:
        if patch_id in rendered:
            return rendered[patch_id]
        if reader is None:  # pre-rendered by fetch_bigearthnet_subset.py
            got = [img_dir / f"{patch_id}_{k}.png" for k in ("s2", "s1")]
            rendered[patch_id] = tuple(str(q.relative_to(args.out)) if q.exists() else None for q in got)
            return rendered[patch_id]
        s2p = s1p = None
        b2 = reader.s2_bands(patch_id)
        if b2 is not None and {"B02", "B03", "B04"} <= set(b2):
            s2p = img_dir / f"{patch_id}_s2.png"
            render_for_vlm(to_s2_raster(b2)).save(s2p)
        b1 = reader.s1_bands(s1_name) if isinstance(s1_name, str) else None
        if b1 is not None and "VV" in b1:
            s1p = img_dir / f"{patch_id}_s1.png"
            render_for_vlm(to_s1_raster(b1)).save(s1p)
        rendered[patch_id] = (str(s2p.relative_to(args.out)) if s2p else None,
                              str(s1p.relative_to(args.out)) if s1p else None)
        return rendered[patch_id]

    for split in args.splits:
        part = select_rows(df, split, args.per_type, args.bench_per_type, args.seed)
        n_ok = n_skip = 0
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in part.itertuples(index=False):
                s2p, s1p = images_for(row.patch_id, row.s1_name)
                if s2p is None:
                    n_skip += 1
                    continue
                single = s1p is None or (split != "bench" and rng.random() < args.single_frac)
                imgs = [s2p] if single else [s2p, s1p]
                mods = [Modality.S2_MS] if single else [Modality.S2_MS, Modality.SAR]
                task = TYPE_TO_TASK.get(row.type, "vqa")
                answer = str(row.output).strip()
                if task == "grounding":
                    answer, tag = grounding_target(answer, str(row.category))
                    frames[tag] += 1
                prompt = build_prompt(task, str(row.input), mods,
                                      expression=str(row.input) if task == "grounding" else None)
                fh.write(json.dumps({"id": str(row.ID), "images": imgs, "modalities": [m.value for m in mods],
                                     "prompt": prompt, "answer": answer, "type": row.type,
                                     "category": row.category, "split": split}) + "\n")
                n_ok += 1
        print(f"{split}: wrote {n_ok} samples, skipped {n_skip} without imagery")
    if frames:
        print("bounding-box coordinate frames detected:", dict(frames),
              "- verify against a few raw 'output' values before training")


if __name__ == "__main__":
    main()
