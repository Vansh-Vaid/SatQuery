"""Canonical in-memory representation of one remote-sensing image.

All loaders convert to the same units so downstream tools never guess:
  * optical / multispectral: surface reflectance in [0, 1] (float32)
  * SAR: backscatter in decibels (float32)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import numpy as np


class Modality(str, Enum):
    S2_MS = "multispectral"      # Sentinel-2-like, >= 4 bands incl. NIR
    OPTICAL_RGB = "optical_rgb"  # 3-band visible (benchmark PNG/JPEG or RGB GeoTIFF)
    SAR = "sar"                  # Sentinel-1-like, VV and/or VH in dB

    @property
    def is_optical(self) -> bool:
        return self in (Modality.S2_MS, Modality.OPTICAL_RGB)


# Band layouts we recognise purely from band count (when files carry no band names).
S2_LAYOUTS: dict[int, list[str]] = {
    13: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
    12: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    10: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    4: ["B02", "B03", "B04", "B08"],
}
RGB_BANDS = ["R", "G", "B"]
SAR_LAYOUTS: dict[int, list[str]] = {2: ["VV", "VH"], 1: ["VV"]}

# Aliases so an RGB image can answer requests for Sentinel-2 visible bands and vice versa.
_BAND_ALIASES = {"R": "B04", "G": "B03", "B": "B02", "RED": "B04", "GREEN": "B03",
                 "BLUE": "B02", "NIR": "B08", "SWIR1": "B11", "SWIR2": "B12"}


@dataclass
class Raster:
    data: np.ndarray                 # (C, H, W) float32, canonical units
    bands: list[str]
    modality: Modality
    source_format: str               # geotiff | tiff | png | jpeg | array
    name: str = "image"
    crs: str | None = None
    transform: tuple[float, ...] | None = None   # affine (a, b, c, d, e, f)
    bounds: tuple[float, float, float, float] | None = None  # (left, bottom, right, top)
    acquired: date | None = None
    valid_mask: np.ndarray | None = None          # (H, W) bool, False where nodata
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Raster data must be (C,H,W); got shape {self.data.shape}")
        if len(self.bands) != self.data.shape[0]:
            raise ValueError(f"{len(self.bands)} band names for {self.data.shape[0]} bands")
        self.data = self.data.astype(np.float32, copy=False)
        if self.valid_mask is None:
            self.valid_mask = np.all(np.isfinite(self.data), axis=0)

    # ------------------------------------------------------------------ access
    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[1]), int(self.data.shape[2])

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None

    def has(self, *names: str) -> bool:
        return all(self._index(n) is not None for n in names)

    def band(self, name: str) -> np.ndarray | None:
        idx = self._index(name)
        return None if idx is None else self.data[idx]

    def _index(self, name: str) -> int | None:
        name = name.upper()
        upper = [b.upper() for b in self.bands]
        if name in upper:
            return upper.index(name)
        alias = _BAND_ALIASES.get(name)
        if alias and alias in upper:
            return upper.index(alias)
        for short, s2 in _BAND_ALIASES.items():  # asking for B04 on an RGB image
            if s2 == name and short in upper:
                return upper.index(short)
        return None

    @property
    def pixel_area_m2(self) -> float | None:
        """Ground area of one pixel when the CRS is projected in metres."""
        if not self.transform or not self.crs:
            return None
        crs = str(self.crs).upper()
        if "4326" in crs or ("WGS 84" in crs and "UTM" not in crs):
            return None  # geographic degrees: area needs reprojection
        a, _, _, _, e, _ = self.transform[:6]
        return abs(a * e)

    @property
    def bbox_wgs84(self) -> list[float] | None:
        """[west, south, east, north] in degrees, or None when not georeferenced."""
        if not self.georeferenced or self.bounds is None:
            return None
        try:
            from rasterio.warp import transform_bounds
            w, s, e, n = transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21)
            return [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]
        except Exception:
            return None

    @property
    def gsd_m(self) -> float | None:
        pa = self.pixel_area_m2
        return round(pa ** 0.5, 3) if pa else None

    def summary(self) -> dict[str, Any]:
        h, w = self.shape
        return {
            "bbox_wgs84": self.bbox_wgs84,
            "gsd_m": self.gsd_m,
            "name": self.name,
            "modality": self.modality.value,
            "format": self.source_format,
            "bands": self.bands,
            "size": [h, w],
            "crs": self.crs,
            "georeferenced": self.georeferenced,
            "acquired": self.acquired.isoformat() if self.acquired else None,
            "valid_fraction": round(float(self.valid_mask.mean()), 4) if self.valid_mask is not None else None,
            "pixel_area_m2": self.pixel_area_m2,
        }
