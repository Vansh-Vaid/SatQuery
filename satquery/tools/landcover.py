"""Physically-based land-cover mapping for optical/multispectral and SAR imagery.

These are the *algorithmic specialists* of SatQuery. They are deterministic,
run on CPU in milliseconds, and produce pixel-level evidence (masks, index
maps, per-pixel confidence) that the learned VLM cannot provide on its own.

Optical (Sentinel-2 conventions):
    NDVI  = (B08 - B04) / (B08 + B04)          vegetation vigour
    MNDWI = (B03 - B11) / (B03 + B11)          open water (Xu 2006)
    NDWI  = (B03 - B08) / (B03 + B08)          open water fallback without SWIR (McFeeters 1996)
    NDBI  = (B11 - B08) / (B11 + B08)          built-up (Zha et al. 2003)
    BSI   = ((B11+B04)-(B08+B02))/((B11+B04)+(B08+B02))  bare soil
SAR (Sentinel-1 VV/VH, dB, after a Lee speckle filter):
    open water   -> specular reflection, very low VV (adaptive Otsu threshold ~ -16 dB)
    built-up     -> double-bounce, very high VV
    vegetation   -> volume scattering, elevated VH cross-pol
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from satquery.imagery.raster import Modality, Raster
from satquery.tools.vocab import CLASSES, direction_of

EPS = 1e-6


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Classic Lee speckle filter on linear intensity."""
    img = np.nan_to_num(img.astype(np.float32), nan=float(np.nanmedian(img)) if np.isfinite(img).any() else 0.0)
    mean = cv2.blur(img, (size, size))
    sq_mean = cv2.blur(img * img, (size, size))
    var = np.maximum(sq_mean - mean * mean, 0)
    noise_var = float(np.mean(var))
    w = var / (var + noise_var + EPS)
    return mean + w * (img - mean)


def otsu(values: np.ndarray, lo: float, hi: float, bins: int = 256) -> float | None:
    """Otsu threshold on values within [lo, hi]; None when the histogram is degenerate."""
    v = values[np.isfinite(values)]
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 50:
        return None
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, EPS)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, EPS)
    between = w0 * w1 * (m0 - m1) ** 2
    if not np.any(between > 0):
        return None
    return float(centers[int(np.argmax(between))])


@dataclass
class LandCoverMap:
    """Exclusive per-pixel classes + supporting evidence on one image grid."""
    masks: dict[str, np.ndarray]                 # class -> bool (H,W); exclusive
    confidence: np.ndarray                       # (H,W) float in [0,1]
    valid: np.ndarray                            # (H,W) bool, analysed pixels
    indices: dict[str, np.ndarray] = field(default_factory=dict)
    cloud: np.ndarray | None = None              # optical only
    method: str = ""
    reliability: dict[str, float] = field(default_factory=dict)  # class -> typical precision
    params: dict[str, Any] = field(default_factory=dict)

    def fractions(self) -> dict[str, float]:
        n = max(int(self.valid.sum()), 1)
        return {c: float((self.masks[c] & self.valid).sum()) / n for c in CLASSES}

    def class_confidence(self, c: str) -> float:
        m = self.masks[c] & self.valid
        if not m.any():
            return self.reliability.get(c, 0.5)
        return float(np.mean(self.confidence[m]))

    def label_map(self) -> np.ndarray:
        out = np.full(self.valid.shape, -1, dtype=np.int8)
        for i, c in enumerate(CLASSES):
            out[self.masks[c] & self.valid] = i
        return out


# ------------------------------------------------------------------ optical
def classify_optical(r: Raster) -> LandCoverMap:
    if r.modality is Modality.S2_MS and r.has("B08", "B04", "B03"):
        return _classify_ms(r)
    return _classify_rgb(r)


def _margin(x: np.ndarray, thr: float, scale: float) -> np.ndarray:
    """0.6 at the threshold rising to 1.0 one ``scale`` away from it."""
    return np.clip(0.6 + 0.4 * np.abs(x - thr) / scale, 0.6, 1.0)


def _classify_ms(r: Raster) -> LandCoverMap:
    b2, b3, b4, b8 = (r.band(b) for b in ("B02", "B03", "B04", "B08"))
    b2 = b2 if b2 is not None else b3
    has_swir = r.has("B11")
    b11 = r.band("B11") if has_swir else None
    valid = r.valid_mask.copy()
    ndvi = nd(b8, b4)
    vis = (b2 + b3 + b4) / 3
    whiteness = np.std(np.stack([b2, b3, b4]), axis=0) / (vis + EPS)
    cloud = (vis > 0.25) & (whiteness < 0.18) & (ndvi < 0.3) & (b2 > 0.2)
    if has_swir:
        wi = nd(b3, b11)          # MNDWI
        wthr, wname = 0.1, "MNDWI"
        ndbi = nd(b11, b8)
        bsi = nd(b11 + b4, b8 + b2)
    else:
        wi = nd(b3, b8)           # NDWI
        wthr, wname = 0.05, "NDWI"
        ndbi = None
        bsi = nd(b4, b3)          # red-over-green proxy for soil without SWIR
    water = (wi > wthr) & (ndvi < 0.15) & (b8 < 0.2) & ~cloud
    if has_swir:
        # L2A atmospheric correction can push very dark water to ~0 in the visible/NIR bands while SWIR stays
        # slightly higher (seen on the Nile in Cairo), so MNDWI turns negative. Green-vs-NIR (NDWI) still holds;
        # accept it only for dark pixels so bright surfaces are unaffected.
        water |= (nd(b3, b8) > 0.2) & (b8 < 0.05) & (b11 < 0.06) & (ndvi < 0.15) & ~cloud
    veg = (ndvi > 0.35) & ~water & ~cloud
    rest = ~water & ~veg & ~cloud
    # sparse / low vegetation; urban surfaces reach NDVI ~0.2 so require a non-built SWIR signature too
    low_veg = rest & (ndvi > 0.25) & ((ndbi < 0) if ndbi is not None else True)
    veg |= low_veg
    rest &= ~low_veg
    bare_thr = 0.1 if has_swir else 0.05
    bare = rest & (bsi > bare_thr) & (b4 > 0.1) & (ndvi > 0.05 if has_swir else ndvi > -1)
    built = rest & ~bare
    conf = np.full(valid.shape, 0.5, np.float32)
    base = 0.85 if has_swir else 0.75
    conf[water] = (base * _margin(wi, wthr, 0.3))[water]
    conf[veg] = (base * _margin(ndvi, 0.35, 0.3))[veg]
    if ndbi is not None:
        conf[built] = (base * 0.85 * _margin(ndbi, -0.05, 0.25))[built]
    else:
        conf[built] = base * 0.7
    conf[bare] = (base * 0.8 * _margin(bsi, bare_thr, 0.2))[bare]
    valid &= ~cloud
    indices = {"NDVI": ndvi, wname: wi}
    if ndbi is not None:
        indices["NDBI"] = ndbi
    rel = {"water": base, "vegetation": base, "built_up": base * 0.75, "bare_soil": base * 0.7}
    return LandCoverMap(masks={"water": water & valid, "vegetation": veg & valid, "built_up": built & valid,
                               "bare_soil": bare & valid},
                        confidence=conf, valid=valid, indices=indices, cloud=cloud & r.valid_mask,
                        method=f"spectral indices (NDVI, {wname}{', NDBI, BSI' if has_swir else ''})",
                        reliability=rel, params={"water_index_threshold": wthr, "ndvi_threshold": 0.35})


def _classify_rgb(r: Raster) -> LandCoverMap:
    rr, gg, bb = (np.clip(r.data[i], 0, 1) for i in range(3))
    valid = r.valid_mask.copy()
    bright = (rr + gg + bb) / 3
    mx = np.maximum(np.maximum(rr, gg), bb)
    mn = np.minimum(np.minimum(rr, gg), bb)
    sat = (mx - mn) / (mx + EPS)
    exg = 2 * gg - rr - bb
    veg = (exg > 0.04) & (gg >= rr)
    water = ~veg & (((bb > rr + 0.02) & (bb >= gg - 0.02) & (bright < 0.45)) | ((bright < 0.1) & (sat < 0.35)))
    bare = ~veg & ~water & (rr > gg) & (gg > bb) & (sat > 0.18) & (bright > 0.25)
    built = ~veg & ~water & ~bare
    conf = np.full(valid.shape, 0.45, np.float32)
    conf[veg] = (0.6 * _margin(exg, 0.04, 0.2))[veg]
    conf[water] = 0.5
    conf[bare] = 0.45
    conf[built] = 0.45
    return LandCoverMap(masks={"water": water & valid, "vegetation": veg & valid, "built_up": built & valid,
                               "bare_soil": bare & valid},
                        confidence=conf, valid=valid, indices={"ExG": exg}, cloud=None,
                        method="RGB colour heuristics (ExG, blue dominance) - approximate, no NIR/SWIR",
                        reliability={"water": 0.5, "vegetation": 0.6, "built_up": 0.45, "bare_soil": 0.45})


# ------------------------------------------------------------------ SAR
def classify_sar(r: Raster, water_db: float | None = None, built_db: float = -7.0,
                 speckle_window: int = 5) -> LandCoverMap:
    valid = r.valid_mask.copy()
    vv_db = r.band("VV") if r.has("VV") else r.data[0]
    vh_db = r.band("VH") if r.has("VH") else None

    def filt(db: np.ndarray) -> np.ndarray:
        lin = np.power(10.0, np.nan_to_num(db, nan=-25.0) / 10.0)
        return 10.0 * np.log10(np.maximum(lee_filter(lin, speckle_window), 1e-6))

    vv = filt(vv_db)
    vh = filt(vh_db) if vh_db is not None else None
    auto_thr = otsu(vv[valid], -26.0, -8.0)
    thr = water_db if water_db is not None else float(np.clip(auto_thr if auto_thr is not None else -16.0, -20.0, -13.0))
    water = vv < thr
    if vh is not None:
        water &= vh < thr - 5.0
    built = (vv > built_db) & ~water
    if vh is not None:
        veg = (vh > -19.0) & ~built & ~water
    else:
        veg = (vv > -12.0) & ~built & ~water
    bare = ~water & ~built & ~veg
    conf = np.full(valid.shape, 0.45, np.float32)
    conf[water] = (0.8 * _margin(vv, thr, 6.0))[water]
    conf[built] = (0.7 * _margin(vv, built_db, 6.0))[built]
    conf[veg] = 0.55 if vh is not None else 0.4
    conf[bare] = 0.4
    indices = {"VV_filtered_dB": vv}
    if vh is not None:
        indices["VH_filtered_dB"] = vh
        indices["VV_minus_VH_dB"] = vv - vh
    return LandCoverMap(masks={"water": water & valid, "vegetation": veg & valid, "built_up": built & valid,
                               "bare_soil": bare & valid},
                        confidence=conf, valid=valid, indices=indices, cloud=None,
                        method=f"SAR backscatter rules (Lee {speckle_window}x{speckle_window}, water VV<{thr:.1f} dB"
                               f"{' (Otsu)' if water_db is None and auto_thr is not None else ''}, built-up VV>{built_db:.1f} dB)",
                        reliability={"water": 0.8, "vegetation": 0.55 if vh is not None else 0.4,
                                     "built_up": 0.65, "bare_soil": 0.4},
                        params={"water_threshold_db": round(thr, 2), "built_threshold_db": built_db})


def classify(r: Raster) -> LandCoverMap:
    return classify_sar(r) if r.modality is Modality.SAR else classify_optical(r)


# ------------------------------------------------------------------ regions
def clean_mask(mask: np.ndarray, open_px: int = 1) -> np.ndarray:
    if open_px <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)


def regions(mask: np.ndarray, min_px: int = 20, label: str = "", max_regions: int = 50) -> list[dict[str, Any]]:
    """Connected components -> list sorted by area (desc) with bbox, centroid and compass direction."""
    h, w = mask.shape
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_px:
            continue
        cx, cy = cents[i]
        out.append({"label": label, "area_px": int(area), "area_frac": area / (h * w),
                    "x0": int(x), "y0": int(y), "x1": int(x + bw), "y1": int(y + bh),
                    "cx": float(cx), "cy": float(cy), "direction": direction_of(cy, cx, h, w), "id": i})
    out.sort(key=lambda d: d["area_px"], reverse=True)
    return out[:max_regions]


def spatial_distribution(mask: np.ndarray) -> str:
    """Where a mask is concentrated, in words (e.g. 'concentrated in the north-east')."""
    if not mask.any():
        return "absent"
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    frac = mask.mean()
    cy, cx = ys.mean(), xs.mean()
    spread = (ys.std() / h + xs.std() / w) / 2
    where = direction_of(cy, cx, h, w)
    if frac > 0.6:
        return "covering most of the scene"
    if spread > 0.25:
        return "scattered across the scene"
    return f"concentrated in the {where}" if where != "centre" else "concentrated in the centre"
