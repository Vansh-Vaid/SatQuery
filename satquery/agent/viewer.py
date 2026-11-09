"""Layered viewer payload for the interactive geo-canvas.

Unlike ``visuals`` (pre-composited PNGs for the HTML report), the viewer gets
separable layers on the reference pixel grid so the UI can toggle and split-swipe:
  * base layers per input image: true colour, false-colour NIR (if NIR), SAR VV/VH
  * transparent RGBA overlays: masks (class colour, alpha) and heatmaps (colormap, alpha)
  * boxes as coordinates (drawn as SVG, hover-linkable from the answer text)
  * geo: CRS, affine transform and WGS84 footprint so the UI can show lat/lon and lock the globe
"""
from __future__ import annotations

from typing import Any

import numpy as np

from satquery.imagery.raster import Modality, Raster
from satquery.imagery.render import colorize, false_color, png_data_uri, to_rgb
from satquery.tools.base import Box, ToolResult


def _hex(c: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


def _rgba_mask(mask: np.ndarray, color: tuple[int, int, int], alpha: int = 165) -> np.ndarray:
    out = np.zeros(mask.shape + (4,), np.uint8)
    m = mask.astype(bool)
    out[m, :3] = color
    out[m, 3] = alpha
    return out


def _rgba_heat(arr: np.ndarray, cmap: str, vmin, vmax, alpha: int = 215) -> np.ndarray:
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    rgb = colorize(np.where(finite, a, vmin if vmin is not None else 0.0), cmap, vmin, vmax)
    out = np.zeros(a.shape + (4,), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.where(finite, alpha, 0)
    return out


def image_layers(r: Raster) -> list[dict[str, str]]:
    layers = []
    if r.modality is Modality.SAR:
        layers.append({"id": "sar", "title": "SAR amplitude (VV/VH/ratio)", "png": png_data_uri(to_rgb(r))})
    else:
        layers.append({"id": "true", "title": "True colour (B04/B03/B02)" if r.modality is Modality.S2_MS
                       else "True colour (RGB)", "png": png_data_uri(to_rgb(r))})
        fc = false_color(r)
        if fc is not None:
            layers.append({"id": "false", "title": "False colour NIR (B08/B04/B03)", "png": png_data_uri(fc)})
    return layers


def image_entry(i: int, r: Raster) -> dict[str, Any]:
    return {"index": i, "name": r.name, "modality": r.modality.value, "bands": r.bands,
            "acquired": r.acquired.isoformat() if r.acquired else None, "bbox_wgs84": r.bbox_wgs84,
            "gsd_m": r.gsd_m, "crs": r.crs, "size": list(r.shape), "layers": image_layers(r)}


def geo_block(r: Raster) -> dict[str, Any]:
    return {"crs": r.crs, "transform": list(r.transform) if r.transform else None, "bbox_wgs84": r.bbox_wgs84,
            "gsd_m": r.gsd_m, "georeferenced": r.georeferenced}


def build_viewer(rasters: list[Raster], results: dict[str, ToolResult], boxes: list[Box]) -> dict[str, Any]:
    ref = rasters[0]
    h, w = ref.shape
    overlays = []
    for res in results.values():
        if not res.ok:
            continue
        for k, layer in enumerate(res.layers):
            if layer.array.shape != (h, w):
                continue
            if layer.kind == "mask":
                if not layer.array.any():
                    continue
                png = png_data_uri(_rgba_mask(layer.array, layer.color))
                color = _hex(layer.color)
            else:
                png = png_data_uri(_rgba_heat(layer.array, layer.cmap, layer.vmin, layer.vmax))
                color = None
            overlays.append({"id": f"{res.tool}:{k}", "title": layer.name, "kind": layer.kind, "tool": res.tool,
                             "legend": layer.legend, "color": color, "png": png})
    return {
        "width": w, "height": h,
        "images": [image_entry(i, r) for i, r in enumerate(rasters)],
        "overlays": overlays,
        "boxes": [{"id": f"b{i}", **b.to_dict()} for i, b in enumerate(boxes)],
        "geo": geo_block(ref),
    }
