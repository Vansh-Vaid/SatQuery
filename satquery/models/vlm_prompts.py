"""Single source of truth for how imagery and tasks are presented to the RS-adapted VLM.

Used by BOTH training (training/prepare_bigearthnet_txt.py, training/finetune_vlm.py)
and inference (satquery/tools/vlm.py) so the fine-tuned model sees at inference
exactly the image renderings and instruction templates it was trained on.

Grounding convention: boxes as JSON ``[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]``
in 0-1000 relative coordinates (Qwen-VL native convention).
"""
from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
from PIL import Image

from satquery.imagery.raster import Modality, Raster
from satquery.imagery.render import to_rgb

SYSTEM_PROMPT = ("You are SatQuery-VLM, a vision-language model adapted to Earth-observation imagery "
                 "(Sentinel-2 multispectral, Sentinel-1 SAR and aerial/optical images). Answer precisely and "
                 "only from what the images show.")

MIN_SIDE = 256  # BigEarthNet patches are 120 px; upsample so the vision encoder sees enough patches
COORD_SCALE = 1000


def describe_image(r_modality: Modality, idx: int, acquired: date | None = None) -> str:
    kind = {
        Modality.S2_MS: "Sentinel-2 true-colour composite (B04, B03, B02)",
        Modality.OPTICAL_RGB: "optical RGB image",
        Modality.SAR: "Sentinel-1 SAR false-colour composite (R=VV, G=VH, B=VV-VH, dB)",
    }[r_modality]
    when = f", acquired {acquired.isoformat()}" if acquired else ""
    return f"Image {idx}: {kind}{when}."


def render_for_vlm(r: Raster) -> Image.Image:
    rgb = to_rgb(r)
    im = Image.fromarray(rgb)
    s = min(im.size)
    if s < MIN_SIDE:
        k = MIN_SIDE / s
        im = im.resize((round(im.width * k), round(im.height * k)), Image.BICUBIC)
    return im


def build_prompt(task: str, query: str, modalities: list[Modality], dates: list[date | None] | None = None,
                 expression: str | None = None) -> str:
    dates = dates or [None] * len(modalities)
    header = "\n".join(describe_image(m, i + 1, d) for i, (m, d) in enumerate(zip(modalities, dates)))
    if task == "caption":
        body = query if query.strip() else "Describe the land cover and major objects visible in this image."
        body += "\nWrite a factual description of the land cover, its spatial layout and notable objects."
    elif task == "grounding":
        expr = expression or query
        body = (f"Locate: {expr}\nReturn a JSON list of objects, each with \"bbox_2d\": [x1, y1, x2, y2] in "
                f"0-{COORD_SCALE} relative coordinates of Image 1, and \"label\".")
    elif task == "change":
        body = f"{query}\nCompare Image 1 (earlier) with Image 2 (later)."
    else:  # vqa / cross-modal vqa
        body = query + "\nAnswer concisely."
    return f"{header}\n{body}"


def chat_messages(images: list[Image.Image], prompt: str, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": im} for im in images] +
                                        [{"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


# ------------------------------------------------------------------ boxes
_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_boxes(text: str) -> list[dict]:
    """Parse model output into [{"bbox_2d": [...], "label": str}] (0-1000 coords)."""
    blob = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", blob, re.S)
    if m:
        blob = m.group(1)
    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            data = [data]
        out = []
        for d in data:
            if isinstance(d, dict) and "bbox_2d" in d and len(d["bbox_2d"]) == 4:
                out.append({"bbox_2d": [float(v) for v in d["bbox_2d"]], "label": str(d.get("label", ""))})
        if out:
            return out
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return [{"bbox_2d": [float(g) for g in mm.groups()], "label": ""} for mm in _BOX_RE.finditer(text)]


def box_to_pixels(b: list[float], h: int, w: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(np.clip(v, 0, COORD_SCALE)) for v in b)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return x0 / COORD_SCALE * w, y0 / COORD_SCALE * h, x1 / COORD_SCALE * w, y1 / COORD_SCALE * h


def pixels_to_box(x0: float, y0: float, x1: float, y1: float, h: int, w: int) -> list[int]:
    return [round(x0 / w * COORD_SCALE), round(y0 / h * COORD_SCALE), round(x1 / w * COORD_SCALE),
            round(y1 / h * COORD_SCALE)]
