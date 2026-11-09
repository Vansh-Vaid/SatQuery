"""Rendering of rasters and analysis layers into displayable 8-bit images."""
from __future__ import annotations

import base64
import io
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from satquery.imagery.raster import Modality, Raster

# Fixed dB display windows for SAR so colours are comparable across images.
_SAR_WINDOWS = {"VV": (-25.0, 0.0), "VH": (-32.0, -5.0), "RATIO": (0.0, 15.0)}


def stretch(arr: np.ndarray, valid: np.ndarray | None = None, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile stretch to [0, 1]; robust to NaNs and constant images."""
    sel = arr[valid & np.isfinite(arr)] if valid is not None else arr[np.isfinite(arr)]
    if sel.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    a, b = np.percentile(sel, [lo, hi])
    if b - a < 1e-9:
        b = a + 1e-9
    return np.clip((np.nan_to_num(arr, nan=a) - a) / (b - a), 0, 1).astype(np.float32)


def _window(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0, 1)


def to_rgb(r: Raster) -> np.ndarray:
    """True-colour (optical) or VV/VH/ratio false-colour (SAR) uint8 HxWx3."""
    if r.modality is Modality.SAR:
        vv = r.band("VV")
        vv = vv if vv is not None else r.data[0]
        vh = r.band("VH")
        if vh is None:
            g = _window(vv, *_SAR_WINDOWS["VV"])
            rgb = np.stack([g, g, g], axis=-1)
        else:
            rgb = np.stack([
                _window(vv, *_SAR_WINDOWS["VV"]),
                _window(vh, *_SAR_WINDOWS["VH"]),
                _window(vv - vh, *_SAR_WINDOWS["RATIO"]),
            ], axis=-1)
    else:
        if r.has("B04", "B03", "B02"):
            chans = [r.band("B04"), r.band("B03"), r.band("B02")]
        else:
            chans = [r.data[min(i, r.data.shape[0] - 1)] for i in range(3)]
        if r.modality is Modality.OPTICAL_RGB and r.source_format in ("png", "jpeg"):
            rgb = np.stack([np.clip(c, 0, 1) for c in chans], axis=-1)
        else:
            stack = np.stack(chans)
            # shared stretch keeps colour balance
            s = stretch(stack, np.broadcast_to(r.valid_mask, stack.shape) if r.valid_mask is not None else None)
            rgb = np.transpose(s, (1, 2, 0))
    out = (rgb * 255).astype(np.uint8)
    if r.valid_mask is not None:
        out[~r.valid_mask] = 0
    return out


def false_color(r: Raster) -> np.ndarray | None:
    """NIR-R-G false colour (vegetation red); None if the image has no NIR."""
    if not r.has("B08", "B04", "B03"):
        return None
    stack = np.stack([r.band("B08"), r.band("B04"), r.band("B03")])
    return (np.transpose(stretch(stack), (1, 2, 0)) * 255).astype(np.uint8)


def colorize(arr: np.ndarray, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    from matplotlib import colormaps

    finite = arr[np.isfinite(arr)]
    vmin = float(np.min(finite)) if vmin is None and finite.size else (vmin or 0.0)
    vmax = float(np.max(finite)) if vmax is None and finite.size else (vmax or 1.0)
    norm = _window(arr, vmin, vmax if vmax > vmin else vmin + 1e-6)
    return (colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, mask: np.ndarray, color: Sequence[int] = (255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return out.astype(np.uint8)


def draw_boxes(rgb: np.ndarray, boxes: Iterable[dict], color: Sequence[int] = (255, 230, 0)) -> np.ndarray:
    """Boxes are dicts with pixel keys x0,y0,x1,y1 and optional label."""
    from PIL import ImageFont

    im = Image.fromarray(rgb)
    scale = max(1, round(max(im.size) / 400))
    try:
        font = ImageFont.load_default(size=max(11, 6 * scale + 6))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    d = ImageDraw.Draw(im)
    for b in boxes:
        xy = [b["x0"], b["y0"], b["x1"], b["y1"]]
        d.rectangle(xy, outline=tuple(color), width=2 * scale)
        if b.get("label"):
            tx, ty = b["x0"] + 2, max(0, b["y0"] - 14 * scale - 4)
            l, t, r, bt = d.textbbox((tx, ty), str(b["label"]), font=font)
            d.rectangle([l - 2, t - 1, r + 2, bt + 1], fill=(0, 0, 0))
            d.text((tx, ty), str(b["label"]), fill=tuple(color), font=font)
    return np.asarray(im)


def upscale_for_display(rgb: np.ndarray, min_side: int = 384) -> np.ndarray:
    """Nearest-neighbour upscale tiny patches (e.g. 120x120 BigEarthNet) for the UI."""
    h, w = rgb.shape[:2]
    k = int(np.ceil(min_side / max(1, min(h, w))))
    if k <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    # fast compression: these are small, transient previews; optimize=True cost ~0.1 s per image
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def png_data_uri(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgb)).decode("ascii")
