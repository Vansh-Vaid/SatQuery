"""Synthetic, physically plausible Sentinel-1/Sentinel-2 scenes with known ground truth.

Used for the offline demo, unit tests and the end-to-end smoke test: every
scene comes with its label map, so tests can assert that tools recover the
truth (e.g. fused water fraction, change location, built-up increase).
Spectral signatures are typical Sentinel-2 L2A surface reflectances; SAR
values are typical Sentinel-1 IW GRD sigma0 in dB with 4-look gamma speckle.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from satquery.imagery.raster import S2_LAYOUTS, Modality, Raster

WATER, VEG, BUILT, BARE = 0, 1, 2, 3
CLASS_NAMES = {WATER: "water", VEG: "vegetation", BUILT: "built_up", BARE: "bare_soil"}

S2_BANDS = S2_LAYOUTS[12]
#                B01   B02   B03   B04   B05   B06   B07   B08   B8A   B09   B11   B12
_SIG = {
    WATER: [0.07, 0.06, 0.05, 0.03, 0.025, 0.02, 0.02, 0.02, 0.018, 0.01, 0.01, 0.005],
    VEG:   [0.03, 0.03, 0.06, 0.03, 0.10, 0.25, 0.30, 0.35, 0.36, 0.30, 0.18, 0.09],
    BUILT: [0.11, 0.12, 0.13, 0.13, 0.15, 0.17, 0.18, 0.20, 0.21, 0.18, 0.24, 0.21],
    BARE:  [0.09, 0.10, 0.14, 0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.25, 0.34, 0.28],
}
_CLOUD = [0.45, 0.50, 0.52, 0.53, 0.54, 0.55, 0.55, 0.56, 0.56, 0.4, 0.35, 0.28]
#                VV     VH
_SAR = {WATER: (-21.0, -28.0), VEG: (-10.0, -16.0), BUILT: (-3.0, -12.0), BARE: (-14.0, -22.0)}

UTM32N = "EPSG:32632"
PIXEL_M = 10.0


def _smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int = 16) -> np.ndarray:
    import cv2
    small = rng.standard_normal((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def base_layout(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """Vegetation background, a lake (NW), a river, a town (E) and a bare quarry (SW)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lab = np.full((h, w), VEG, np.int8)
    lake = ((yy - 0.22 * h) / (0.12 * h)) ** 2 + ((xx - 0.25 * w) / (0.16 * w)) ** 2 < 1
    centre = 0.5 * w + 0.08 * w * np.sin(yy / h * 2 * np.pi * 1.5) + 2 * _smooth_noise(rng, h, w, 32)
    river = np.abs(xx - centre) < 0.025 * w
    lab[lake | river] = WATER
    town = (yy > 0.35 * h) & (yy < 0.7 * h) & (xx > 0.66 * w) & (xx < 0.92 * w)
    lab[town] = BUILT
    quarry = ((yy - 0.8 * h) / (0.08 * h)) ** 2 + ((xx - 0.2 * w) / (0.1 * w)) ** 2 < 1
    lab[quarry & (lab == VEG)] = BARE
    return lab


def evolve(lab: np.ndarray) -> np.ndarray:
    """Second date: town expands west into fields, part of the lake dries to bare soil."""
    h, w = lab.shape
    out = lab.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    expansion = (yy > 0.40 * h) & (yy < 0.65 * h) & (xx > 0.57 * w) & (xx <= 0.66 * w) & (lab == VEG)
    out[expansion] = BUILT
    dried = (lab == WATER) & (yy < 0.34 * h) & (xx < 0.16 * w)
    out[dried] = BARE
    return out


def cloud_mask(h: int, w: int, seed: int = 1, cover: float = 0.18) -> np.ndarray:
    n = _smooth_noise(np.random.default_rng(seed), h, w, 48)
    n += np.linspace(1.0, -1.0, w)[None, :]  # clouds concentrated in the west
    return n > np.quantile(n, 1 - cover)


def _geo(h: int, w: int) -> dict:
    return {"crs": UTM32N, "transform": (PIXEL_M, 0.0, 500000.0, 0.0, -PIXEL_M, 5300000.0),
            "bounds": (500000.0, 5300000.0 - h * PIXEL_M, 500000.0 + w * PIXEL_M, 5300000.0)}


def render_s2(lab: np.ndarray, seed: int = 0, cloud: np.ndarray | None = None, gain: float = 1.0,
              acquired: date | None = None, name: str = "s2.tif") -> Raster:
    rng = np.random.default_rng(seed)
    h, w = lab.shape
    data = np.zeros((12, h, w), np.float32)
    texture = 1 + 0.12 * _smooth_noise(rng, h, w, 8)
    for k, sig in _SIG.items():
        m = lab == k
        for b in range(12):
            data[b][m] = sig[b]
    veg = lab == VEG
    data[:, veg] *= texture[veg]
    data *= gain
    data += rng.normal(0, 0.006, data.shape).astype(np.float32)
    if cloud is not None:
        for b in range(12):
            data[b][cloud] = _CLOUD[b] + rng.normal(0, 0.02, int(cloud.sum()))
    data = np.clip(data, 0, 1)
    return Raster(data=data, bands=list(S2_BANDS), modality=Modality.S2_MS, source_format="geotiff", name=name,
                  acquired=acquired, meta={"modality_source": "synthetic"}, **_geo(h, w))


def render_s1(lab: np.ndarray, seed: int = 0, acquired: date | None = None, name: str = "s1.tif",
              looks: int = 4) -> Raster:
    rng = np.random.default_rng(seed + 100)
    h, w = lab.shape
    data = np.zeros((2, h, w), np.float32)
    for k, (vv, vh) in _SAR.items():
        m = lab == k
        data[0][m] = vv
        data[1][m] = vh
    lin = 10 ** (data / 10)
    lin *= rng.gamma(looks, 1.0 / looks, lin.shape)
    data = 10 * np.log10(np.maximum(lin, 1e-6)).astype(np.float32)
    return Raster(data=data, bands=["VV", "VH"], modality=Modality.SAR, source_format="geotiff", name=name,
                  acquired=acquired, meta={"modality_source": "synthetic"}, **_geo(h, w))


def truth_fractions(lab: np.ndarray) -> dict[str, float]:
    return {CLASS_NAMES[k]: float((lab == k).mean()) for k in CLASS_NAMES}
