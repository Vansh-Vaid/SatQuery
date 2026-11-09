"""Input validation: count, modality, format, metadata and pair compatibility.

Produces an auditable :class:`ValidationReport` (one entry per check) and the
aligned/reordered rasters that the tools will consume. The controller refuses
to execute any tool when the report has a failing check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from satquery.imagery.raster import Modality, Raster


class InputConfig(str, Enum):
    SINGLE = "single"
    CROSS_MODAL = "cross_modal"   # co-registered optical/MS + SAR
    BITEMPORAL = "bitemporal"     # same area, two dates
    INVALID = "invalid"


@dataclass
class Check:
    id: str
    status: str   # pass | warn | fail
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "status": self.status, "message": self.message}


@dataclass
class ValidationReport:
    config: InputConfig
    rasters: list[Raster]
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    @property
    def warnings(self) -> list[str]:
        return [c.message for c in self.checks if c.status == "warn"]

    def add(self, id: str, status: str, message: str) -> None:
        self.checks.append(Check(id, status, message))

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.value, "ok": self.ok,
                "images": [r.summary() for r in self.rasters],
                "checks": [c.to_dict() for c in self.checks]}


MIN_SIDE = 32
MAX_CROSS_MODAL_GAP_DAYS = 15


def validate_inputs(rasters: list[Raster], pair_hint: str | None = None) -> ValidationReport:
    """Validate 1-2 rasters; ``pair_hint`` in {None, 'cross_modal', 'bitemporal'}."""
    rep = ValidationReport(InputConfig.INVALID, list(rasters))
    n = len(rasters)
    if n == 0:
        rep.add("count", "fail", "No image supplied.")
        return rep
    if n > 2:
        rep.add("count", "fail", f"{n} images supplied; SatQuery accepts one image or one pair.")
        return rep
    rep.add("count", "pass", f"{n} image(s) supplied.")

    for i, r in enumerate(rasters):
        _check_single(rep, r, i)

    if n == 1:
        rep.config = InputConfig.SINGLE
        return rep

    a, b = rasters
    config = _infer_pair_config(a, b, pair_hint, rep)
    rep.config = config
    if config is InputConfig.INVALID:
        return rep
    if config is InputConfig.CROSS_MODAL:
        # canonical order: optical first, SAR second
        if a.modality is Modality.SAR:
            a, b = b, a
        _check_dates_cross_modal(rep, a, b)
    else:
        a, b = _order_bitemporal(rep, a, b)
    b = _align(rep, a, b)
    rep.rasters = [a, b]
    _check_overlap_validity(rep, a, b)
    return rep


# ----------------------------------------------------------------- single image
def _check_single(rep: ValidationReport, r: Raster, i: int) -> None:
    tag = f"image{i + 1}"
    if r.source_format in ("png", "jpeg"):
        rep.add(f"{tag}.format", "warn",
                f"{r.name}: {r.source_format.upper()} accepted for benchmark imagery only; no georeferencing, "
                f"radiometry is 8-bit so spectral indices are approximate.")
    else:
        rep.add(f"{tag}.format", "pass", f"{r.name}: {r.source_format.upper()} with {len(r.bands)} band(s).")
    src = r.meta.get("modality_source", "?")
    status = "warn" if src in ("band_statistics",) else "pass"
    rep.add(f"{tag}.modality", status,
            f"{r.name}: modality '{r.modality.value}' (determined from {src.replace('_', ' ')})."
            + (" Set the modality explicitly if this is wrong." if status == "warn" else ""))
    h, w = r.shape
    if min(h, w) < MIN_SIDE:
        rep.add(f"{tag}.size", "fail", f"{r.name}: {h}x{w} px is below the {MIN_SIDE}px minimum.")
    else:
        msg = f"{r.name}: {h}x{w} px"
        if r.meta.get("downsample", 1) < 1:
            msg += f" (downsampled from {r.meta['original_size'][0]}x{r.meta['original_size'][1]} for processing)"
        rep.add(f"{tag}.size", "pass", msg + ".")
    vf = float(r.valid_mask.mean()) if r.valid_mask is not None else 1.0
    if vf < 0.05:
        rep.add(f"{tag}.valid", "fail", f"{r.name}: only {vf:.1%} of pixels contain data.")
    elif vf < 0.5:
        rep.add(f"{tag}.valid", "warn", f"{r.name}: {vf:.1%} valid pixels; results cover the valid area only.")
    else:
        rep.add(f"{tag}.valid", "pass", f"{r.name}: {vf:.1%} valid pixels.")
    if r.modality is Modality.S2_MS and not r.has("B08"):
        rep.add(f"{tag}.bands", "warn", f"{r.name}: no NIR band identified; vegetation/water indices limited.")
    if r.modality is Modality.SAR and not r.has("VH"):
        rep.add(f"{tag}.bands", "warn", f"{r.name}: single-polarisation SAR; built-up detection less reliable.")


# ----------------------------------------------------------------- pairs
def _infer_pair_config(a: Raster, b: Raster, hint: str | None, rep: ValidationReport) -> InputConfig:
    mixed = (a.modality is Modality.SAR) != (b.modality is Modality.SAR)
    hint = (hint or "").lower() or None
    if hint == "bitemporal" and mixed:
        rep.add("pair.type", "fail", "Bi-temporal analysis needs two images of the same sensor family "
                                     "(optical/optical or SAR/SAR); got optical + SAR.")
        return InputConfig.INVALID
    if hint == "cross_modal" and not mixed:
        rep.add("pair.type", "fail", "Cross-modal analysis needs one optical/multispectral and one SAR image.")
        return InputConfig.INVALID
    config = InputConfig.CROSS_MODAL if mixed else InputConfig.BITEMPORAL
    rep.add("pair.type", "pass",
            f"Pair interpreted as {'cross-modal (optical + SAR)' if mixed else 'bi-temporal (same sensor family)'}"
            f"{' as requested' if hint else ' from image modalities'}.")
    return config


def _check_dates_cross_modal(rep: ValidationReport, opt: Raster, sar: Raster) -> None:
    if opt.acquired and sar.acquired:
        gap = abs((opt.acquired - sar.acquired).days)
        if gap > MAX_CROSS_MODAL_GAP_DAYS:
            rep.add("pair.dates", "warn", f"Optical and SAR acquired {gap} days apart; real changes may "
                                          f"appear as cross-modal disagreement.")
        else:
            rep.add("pair.dates", "pass", f"Optical and SAR acquired {gap} day(s) apart.")
    else:
        rep.add("pair.dates", "warn", "Acquisition dates unknown; assuming near-simultaneous acquisitions.")


def _order_bitemporal(rep: ValidationReport, a: Raster, b: Raster) -> tuple[Raster, Raster]:
    if a.acquired and b.acquired:
        if a.acquired == b.acquired:
            rep.add("pair.dates", "warn", f"Both images dated {a.acquired}; change analysis assumes upload order.")
            return a, b
        if b.acquired < a.acquired:
            rep.add("pair.dates", "pass", f"Reordered chronologically: {b.acquired} -> {a.acquired}.")
            return b, a
        rep.add("pair.dates", "pass", f"Chronological order {a.acquired} -> {b.acquired} "
                                      f"({(b.acquired - a.acquired).days} days).")
        return a, b
    rep.add("pair.dates", "warn", "Acquisition dates missing; first upload treated as the earlier date.")
    return a, b


def _align(rep: ValidationReport, ref: Raster, mov: Raster) -> Raster:
    """Bring ``mov`` onto ``ref``'s pixel grid (reproject when georeferenced, else resize)."""
    if ref.georeferenced and mov.georeferenced:
        same_grid = (str(ref.crs) == str(mov.crs) and ref.shape == mov.shape and
                     np.allclose(ref.transform, mov.transform, rtol=0, atol=1e-6))
        if same_grid:
            rep.add("pair.grid", "pass", "Images share CRS and pixel grid (co-registered).")
            return mov
        iou = _bounds_iou(ref, mov)
        if iou is not None and iou < 0.05:
            rep.add("pair.grid", "fail", f"Images do not cover the same area (footprint IoU {iou:.2f}).")
            return mov
        rep.add("pair.grid", "warn" if (iou or 1) < 0.8 else "pass",
                f"Second image reprojected onto the first image's grid ({ref.crs}); footprint IoU "
                f"{iou:.2f}." if iou is not None else "Second image reprojected onto first grid.")
        return _reproject(ref, mov)
    if ref.georeferenced != mov.georeferenced:
        rep.add("pair.georef", "warn", "Only one image is georeferenced; assuming identical footprints.")
    if ref.shape != mov.shape:
        rep.add("pair.grid", "warn", f"Pixel grids differ ({ref.shape} vs {mov.shape}); second image resized. "
                                     f"Spatial correspondence is assumed, not verified.")
        h, w = ref.shape
        data = np.stack([cv2.resize(np.nan_to_num(c), (w, h), interpolation=cv2.INTER_LINEAR) for c in mov.data])
        mask = cv2.resize(mov.valid_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        data[:, ~mask] = np.nan
        return Raster(data=data, bands=mov.bands, modality=mov.modality, source_format=mov.source_format,
                      name=mov.name, crs=ref.crs, transform=ref.transform, bounds=ref.bounds,
                      acquired=mov.acquired, valid_mask=mask, meta={**mov.meta, "resized_to_ref": True})
    rep.add("pair.grid", "pass" if not (ref.georeferenced or mov.georeferenced) else "warn",
            "Images have identical pixel dimensions; treated as co-registered.")
    return mov


def _bounds_iou(a: Raster, b: Raster) -> float | None:
    try:
        from rasterio.warp import transform_bounds
        bb = transform_bounds(b.crs, a.crs, *b.bounds) if str(a.crs) != str(b.crs) else b.bounds
    except Exception:
        return None
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = bb
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _reproject(ref: Raster, mov: Raster) -> Raster:
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    h, w = ref.shape
    out = np.full((mov.data.shape[0], h, w), np.nan, dtype=np.float32)
    for i in range(mov.data.shape[0]):
        reproject(source=mov.data[i], destination=out[i], src_transform=Affine(*mov.transform[:6]),
                  src_crs=mov.crs, dst_transform=Affine(*ref.transform[:6]), dst_crs=ref.crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return Raster(data=out, bands=mov.bands, modality=mov.modality, source_format=mov.source_format,
                  name=mov.name, crs=ref.crs, transform=ref.transform, bounds=ref.bounds,
                  acquired=mov.acquired, meta={**mov.meta, "reprojected_to_ref": True})


def _check_overlap_validity(rep: ValidationReport, a: Raster, b: Raster) -> None:
    joint = a.valid_mask & b.valid_mask
    frac = float(joint.mean())
    if frac < 0.05:
        rep.add("pair.overlap", "fail", f"Only {frac:.1%} of pixels are valid in both images.")
    elif frac < 0.5:
        rep.add("pair.overlap", "warn", f"{frac:.1%} of pixels valid in both images; analysis limited to that area.")
    else:
        rep.add("pair.overlap", "pass", f"{frac:.1%} of pixels valid in both images.")
