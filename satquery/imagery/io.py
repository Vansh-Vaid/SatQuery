"""Loading and saving of supported image formats into :class:`Raster`.

Supported: GeoTIFF / TIFF (any band count) for geospatial imagery; PNG / JPEG
only for public benchmark images (RSVQA, VRSBench, CDVQA), per the brief.

Modality is resolved in this priority order:
  1. explicit hint from the caller (UI dropdown / API field)
  2. GeoTIFF tag ``SATQUERY_MODALITY`` or band descriptions (``VV``, ``B04`` ...)
  3. filename hints (``S1``, ``SAR``, ``S2`` ...)
  4. band count + value statistics
Whatever is inferred is recorded in ``raster.meta["modality_source"]`` so the
validation report can tell the user how confident the inference was.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from satquery.config import settings
from satquery.imagery.raster import RGB_BANDS, S2_LAYOUTS, SAR_LAYOUTS, Modality, Raster

GEO_EXT = {".tif", ".tiff", ".gtiff"}
BENCH_EXT = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXT = GEO_EXT | BENCH_EXT


class UnsupportedInput(ValueError):
    """Raised for inputs the system must reject (wrong format, unreadable...)."""


# ----------------------------------------------------------------- helpers
_DATE_PATTERNS = [
    (re.compile(r"(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])"), "%Y%m%d"),
]


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    text = str(text).strip()
    head = text[:10].replace(":", "-")  # TIFFTAG_DATETIME uses YYYY:MM:DD
    try:
        return datetime.strptime(head, "%Y-%m-%d").date()
    except ValueError:
        pass
    for pat, _ in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def _modality_from_hint(hint: str | None) -> Modality | None:
    if not hint:
        return None
    h = hint.strip().lower()
    if h in {"sar", "s1", "sentinel-1", "sentinel1", "radar"}:
        return Modality.SAR
    if h in {"ms", "s2", "multispectral", "sentinel-2", "sentinel2"}:
        return Modality.S2_MS
    if h in {"rgb", "optical", "optical_rgb"}:
        return Modality.OPTICAL_RGB
    return None


def _modality_from_name(name: str) -> Modality | None:
    n = Path(name).stem.lower()
    if re.search(r"(^|[^a-z])(s1[ab]?|sar|grd|vv|vh)([^a-z]|$)", n):
        return Modality.SAR
    if re.search(r"(^|[^a-z])(s2[ab]?|msi|l2a|l1c|multispectral)([^a-z]|$)", n):
        return Modality.S2_MS
    return None


def _modality_from_bands(descs: list[str]) -> Modality | None:
    up = {d.upper() for d in descs if d}
    if up & {"VV", "VH", "HH", "HV"}:
        return Modality.SAR
    if up & {"B08", "B8", "B8A", "B11", "NIR"}:
        return Modality.S2_MS
    if up and up <= {"R", "G", "B", "RED", "GREEN", "BLUE", "B02", "B03", "B04"}:
        return Modality.OPTICAL_RGB
    return None


def _modality_from_stats(arr: np.ndarray) -> Modality:
    c = arr.shape[0]
    finite = arr[np.isfinite(arr)]
    has_negative = finite.size > 0 and float(np.nanpercentile(finite, 5)) < 0
    is_float = np.issubdtype(arr.dtype, np.floating)
    if c <= 2 and (has_negative or is_float):
        return Modality.SAR
    if c == 3:
        return Modality.OPTICAL_RGB
    if c in S2_LAYOUTS or c > 4:
        return Modality.S2_MS
    return Modality.SAR if has_negative else Modality.OPTICAL_RGB


def _band_names(modality: Modality, count: int, descs: list[str]) -> list[str]:
    if descs and all(descs) and len(set(descs)) == count:
        return [d.upper() for d in descs]
    if modality is Modality.SAR and count in SAR_LAYOUTS:
        return list(SAR_LAYOUTS[count])
    if modality is Modality.S2_MS and count in S2_LAYOUTS:
        return list(S2_LAYOUTS[count])
    if modality is Modality.OPTICAL_RGB and count >= 3:
        return RGB_BANDS + [f"X{i}" for i in range(4, count + 1)]
    prefix = "SAR" if modality is Modality.SAR else "B"
    return [f"{prefix}{i + 1}" for i in range(count)]


def _to_canonical(arr: np.ndarray, modality: Modality, src_dtype: np.dtype) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert raw digital numbers to reflectance [0,1] or SAR dB."""
    info: dict[str, Any] = {}
    data = arr.astype(np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return data, {"scaling": "empty"}
    if modality is Modality.SAR:
        p50 = float(np.median(finite))
        if float(finite.min()) >= 0 and p50 < 2.0:  # linear power (sigma0)
            data = 10.0 * np.log10(np.clip(data, 1e-6, None))
            info["scaling"] = "linear->dB"
        else:
            info["scaling"] = "dB"
        return data, info
    # optical
    if np.issubdtype(src_dtype, np.uint8) or (float(finite.max()) <= 255 and float(finite.max()) > 2.0
                                               and modality is Modality.OPTICAL_RGB):
        data /= 255.0
        info["scaling"] = "uint8/255"
    elif float(np.percentile(finite, 99)) > 2.0:
        data /= 10000.0  # Sentinel-2 L1C/L2A digital numbers
        info["scaling"] = "DN/10000"
    else:
        info["scaling"] = "reflectance"
    np.clip(data, 0.0, 1.5, out=data)
    return data, info


# ----------------------------------------------------------------- loaders
def load_image(path: str | Path, modality_hint: str | None = None, acquired: str | None = None,
               display_name: str | None = None) -> Raster:
    path = Path(path)
    ext = path.suffix.lower()
    name = display_name or path.name
    if ext not in SUPPORTED_EXT:
        raise UnsupportedInput(f"{name}: unsupported format '{ext}'. Use GeoTIFF/TIFF "
                               f"(or PNG/JPEG for benchmark images).")
    if ext in BENCH_EXT:
        if not settings.allow_benchmark_png:
            raise UnsupportedInput(f"{name}: PNG/JPEG inputs are disabled on this deployment.")
        return _load_pil(path, name, modality_hint, acquired)
    return _load_tiff(path, name, modality_hint, acquired)


def _load_tiff(path: Path, name: str, hint: str | None, acquired: str | None) -> Raster:
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rasterio is required to read GeoTIFF inputs") from exc

    try:
        src = rasterio.open(path)
    except Exception as exc:
        raise UnsupportedInput(f"{name}: not a readable TIFF ({exc})") from exc
    with src:
        h, w, c = src.height, src.width, src.count
        scale = 1.0
        if h * w > settings.max_pixels:
            scale = (settings.max_pixels / (h * w)) ** 0.5
        out_shape = (c, max(1, int(h * scale)), max(1, int(w * scale)))
        arr = src.read(out_shape=out_shape, resampling=Resampling.average if scale < 1 else Resampling.nearest,
                       masked=True)
        raw = np.ma.filled(arr.astype(np.float64), np.nan)
        src_dtype = np.dtype(src.dtypes[0])
        tags = {k.upper(): v for k, v in src.tags().items()}
        descs = [d or "" for d in src.descriptions]
        crs = src.crs.to_string() if src.crs else None
        transform = src.transform
        if scale < 1:
            transform = transform * transform.scale(w / out_shape[2], h / out_shape[1])
        transform_t = tuple(transform)[:6] if crs else None
        bounds = tuple(src.bounds) if crs else None
        georef = bool(crs) and not src.transform.is_identity

    modality, source = _resolve_modality(hint, tags.get("SATQUERY_MODALITY"), descs, name, raw)
    bands = _band_names(modality, c, descs)
    data, info = _to_canonical(raw, modality, src_dtype)
    acq = parse_date(acquired) or parse_date(tags.get("ACQUISITION_DATE")) or parse_date(
        tags.get("DATE")) or parse_date(tags.get("TIFFTAG_DATETIME")) or parse_date(name)
    return Raster(
        data=data, bands=bands, modality=modality,
        source_format="geotiff" if georef else "tiff",
        name=name, crs=crs if georef else None,
        transform=transform_t if georef else None,
        bounds=bounds if georef else None,
        acquired=acq,
        valid_mask=np.all(np.isfinite(data), axis=0),
        meta={"modality_source": source, "downsample": round(scale, 4), "src_dtype": str(src_dtype),
              "original_size": [h, w], **info},
    )


def _load_pil(path: Path, name: str, hint: str | None, acquired: str | None) -> Raster:
    from PIL import Image

    try:
        with Image.open(path) as im:
            mode = im.mode
            im = im.convert("L") if mode in ("L", "I;16", "I", "F") else im.convert("RGB")
            if im.width * im.height > settings.max_pixels:
                s = (settings.max_pixels / (im.width * im.height)) ** 0.5
                im = im.resize((int(im.width * s), int(im.height * s)))
            arr = np.asarray(im)
    except Exception as exc:
        raise UnsupportedInput(f"{name}: not a readable image ({exc})") from exc
    arr = arr[None] if arr.ndim == 2 else np.transpose(arr, (2, 0, 1))
    hinted = _modality_from_hint(hint)
    if arr.shape[0] == 1:
        modality = hinted or Modality.SAR  # single-channel benchmark images are SAR amplitude
        data = arr.astype(np.float32) / 255.0
        if modality is Modality.SAR:
            # 8-bit amplitude rendering -> approximate dB over a [-25, 0] dB window
            data = data * 25.0 - 25.0
            bands = ["VV"]
        else:
            bands = ["PAN"]
        info = {"scaling": "uint8 amplitude->approx dB" if modality is Modality.SAR else "uint8/255"}
    else:
        modality = hinted or Modality.OPTICAL_RGB
        if modality is Modality.SAR:
            modality = Modality.OPTICAL_RGB  # a 3-channel PNG cannot be quantitative SAR
        data = arr.astype(np.float32) / 255.0
        bands = list(RGB_BANDS)
        info = {"scaling": "uint8/255"}
    return Raster(
        data=data, bands=bands, modality=modality,
        source_format="jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png",
        name=name, acquired=parse_date(acquired) or parse_date(name),
        meta={"modality_source": "hint" if hinted else "channels", **info},
    )


def _resolve_modality(hint: str | None, tag: str | None, descs: list[str], name: str,
                      raw: np.ndarray) -> tuple[Modality, str]:
    for source, value in (("hint", _modality_from_hint(hint)), ("tag", _modality_from_hint(tag)),
                          ("band_names", _modality_from_bands(descs)), ("filename", _modality_from_name(name))):
        if value is not None:
            return value, source
    if raw.shape[0] in (10, 12, 13):
        return Modality.S2_MS, "sentinel2_band_layout"
    return _modality_from_stats(raw), "band_statistics"


def raster_from_array(data: np.ndarray, modality: Modality, bands: list[str] | None = None,
                      name: str = "array", acquired: date | None = None, **kw: Any) -> Raster:
    """Construct a Raster from an array already in canonical units (used by tests/eval)."""
    if data.ndim == 2:
        data = data[None]
    bands = bands or _band_names(modality, data.shape[0], [])
    return Raster(data=data.astype(np.float32), bands=bands, modality=modality, source_format="array",
                  name=name, acquired=acquired, meta={"modality_source": "array"}, **kw)


def save_geotiff(raster: Raster, path: str | Path, dn_scale: bool = True) -> Path:
    """Write a Raster to GeoTIFF with band descriptions and SatQuery tags.

    Optical data is written as uint16 DN (reflectance*10000) when ``dn_scale``.
    """
    import rasterio
    from rasterio.transform import Affine

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raster.modality.is_optical and dn_scale:
        out = np.clip(np.nan_to_num(raster.data) * 10000.0, 0, 65535).astype(np.uint16)
        dtype = "uint16"
    else:
        out = raster.data.astype(np.float32)
        dtype = "float32"
    c, h, w = out.shape
    profile: dict[str, Any] = {"driver": "GTiff", "height": h, "width": w, "count": c, "dtype": dtype,
                               "compress": "deflate"}
    if raster.crs and raster.transform:
        profile["crs"] = raster.crs
        profile["transform"] = Affine(*raster.transform[:6])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out)
        for i, b in enumerate(raster.bands, start=1):
            dst.set_band_description(i, b)
        tags = {"SATQUERY_MODALITY": raster.modality.value}
        if raster.acquired:
            tags["ACQUISITION_DATE"] = raster.acquired.isoformat()
        dst.update_tags(**tags)
    return path
