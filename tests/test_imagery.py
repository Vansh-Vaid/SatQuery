from datetime import date

import numpy as np
import pytest
from PIL import Image

from satquery.demo import synth as S
from satquery.imagery.io import UnsupportedInput, load_image, parse_date, save_geotiff
from satquery.imagery.raster import Modality
from satquery.imagery.validate import InputConfig, validate_inputs


def test_parse_date_variants():
    assert parse_date("2024-06-01") == date(2024, 6, 1)
    assert parse_date("2024:06:01 10:00:00") == date(2024, 6, 1)
    assert parse_date("S2A_MSIL2A_20230715T103031_N0509") == date(2023, 7, 15)
    assert parse_date("no date here") is None


def test_geotiff_roundtrip_keeps_modality_bands_date(tmp_path, s2, s1):
    p2 = save_geotiff(s2, tmp_path / "scene.tif")
    p1 = save_geotiff(s1, tmp_path / "radar.tif")
    r2, r1 = load_image(p2), load_image(p1)
    assert r2.modality is Modality.S2_MS and r2.bands == s2.bands and r2.acquired == s2.acquired
    assert r1.modality is Modality.SAR and r1.bands == ["VV", "VH"]
    assert r2.georeferenced and r2.pixel_area_m2 == pytest.approx(100.0)
    assert np.allclose(r2.data, s2.data, atol=2e-4)       # uint16 DN quantisation
    assert np.allclose(r1.data, s1.data, atol=1e-5)


def test_untagged_tiff_modality_from_statistics(tmp_path):
    import rasterio
    sar = (np.random.default_rng(0).normal(-12, 3, (2, 64, 64))).astype("float32")
    with rasterio.open(tmp_path / "x.tif", "w", driver="GTiff", height=64, width=64, count=2, dtype="float32") as d:
        d.write(sar)
    r = load_image(tmp_path / "x.tif")
    assert r.modality is Modality.SAR and r.meta["modality_source"] == "band_statistics"
    assert r.source_format == "tiff"  # no georeferencing


def test_png_accepted_as_benchmark_rgb(tmp_path):
    Image.fromarray(np.full((64, 64, 3), 120, np.uint8)).save(tmp_path / "rsvqa.png")
    r = load_image(tmp_path / "rsvqa.png")
    assert r.modality is Modality.OPTICAL_RGB and r.source_format == "png"
    rep = validate_inputs([r])
    assert any(c.id == "image1.format" and c.status == "warn" for c in rep.checks)


def test_unsupported_format_rejected(tmp_path):
    (tmp_path / "a.bmp").write_bytes(b"x")
    with pytest.raises(UnsupportedInput):
        load_image(tmp_path / "a.bmp")


def test_pair_config_detection(s2, s2_t2, s1):
    assert validate_inputs([s2]).config is InputConfig.SINGLE
    assert validate_inputs([s1, s2]).config is InputConfig.CROSS_MODAL
    rep = validate_inputs([s2_t2, s2])
    assert rep.config is InputConfig.BITEMPORAL
    assert rep.rasters[0].acquired < rep.rasters[1].acquired  # reordered chronologically


def test_invalid_pairs_and_counts(s2, s1, s2_t2):
    assert not validate_inputs([]).ok
    assert not validate_inputs([s2, s1, s2_t2]).ok
    assert not validate_inputs([s2, s1], pair_hint="bitemporal").ok
    assert not validate_inputs([s2, s2_t2], pair_hint="cross_modal").ok


def test_non_overlapping_footprints_fail(s2):
    lab = S.base_layout(128, 128)
    far = S.render_s2(lab, seed=9)
    far.transform = (10.0, 0.0, 900000.0, 0.0, -10.0, 5300000.0)
    far.bounds = (900000.0, 5298720.0, 901280.0, 5300000.0)
    rep = validate_inputs([s2, far])
    assert not rep.ok and any(c.id == "pair.grid" and c.status == "fail" for c in rep.checks)


def test_mismatched_grid_is_resized_with_warning(s2):
    small = S.render_s2(S.base_layout(64, 64), seed=3)
    small.crs = small.transform = small.bounds = None
    rep = validate_inputs([s2, small])
    assert rep.ok and rep.rasters[1].shape == s2.shape


# ---- Landsat 8/9 -----------------------------------------------------------------------------------------
def _write_landsat(path, s2, names=None, scale="c2"):
    """The synthetic Sentinel-2 scene re-encoded as a Landsat 8/9 Collection-2 Level-2 stack."""
    import rasterio
    from rasterio.transform import Affine
    order = ["B01", "B02", "B03", "B04", "B08", "B11", "B12"]
    refl = np.stack([s2.data[s2.bands.index(b)] for b in order])
    dn = np.round((refl + 0.2) / 0.0000275).astype(np.uint16) if scale == "c2" else refl.astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=dn.shape[1], width=dn.shape[2], count=7,
                       dtype=dn.dtype, crs=s2.crs, transform=Affine(*s2.transform)) as dst:
        dst.write(dn)
        for i, n in enumerate(names or []):
            dst.set_band_description(i + 1, n)
    return refl


def test_landsat_c2_stack_maps_onto_sentinel2_bands(tmp_path, s2):
    refl = _write_landsat(tmp_path / "scene.tif", s2, names=[f"SR_B{i}" for i in range(1, 8)])
    r = load_image(tmp_path / "scene.tif")
    assert r.modality is Modality.S2_MS and r.meta["sensor"] == "landsat"
    assert r.bands == ["B01", "B02", "B03", "B04", "B08", "B11", "B12"]
    assert np.allclose(r.data, np.clip(refl, 0, 1.5), atol=2e-3)       # USGS scale factor and offset undone


def test_landsat_recognised_from_file_name_and_gives_same_land_cover(tmp_path, s2):
    from satquery.tools.landcover import classify
    _write_landsat(tmp_path / "LC09_L2SP_146040_20240601_scene.tif", s2)
    r = load_image(tmp_path / "LC09_L2SP_146040_20240601_scene.tif")
    assert r.meta["sensor"] == "landsat" and "B08" in r.bands and "B11" in r.bands
    a, b = classify(s2).label_map(), classify(r).label_map()
    assert (a == b).mean() > 0.9        # the optical tools read Landsat like Sentinel-2


def test_sentinel_file_with_b4_style_names_is_not_mistaken_for_landsat(tmp_path, s2):
    import rasterio
    from rasterio.transform import Affine
    four = np.stack([s2.data[s2.bands.index(b)] for b in ("B02", "B03", "B04", "B08")]).astype(np.float32)
    with rasterio.open(tmp_path / "s2_scene.tif", "w", driver="GTiff", height=four.shape[1], width=four.shape[2],
                       count=4, dtype="float32", crs=s2.crs, transform=Affine(*s2.transform)) as dst:
        dst.write(four)
        for i, n in enumerate(["B2", "B3", "B4", "B8"]):
            dst.set_band_description(i + 1, n)
    r = load_image(tmp_path / "s2_scene.tif")
    assert r.meta.get("sensor") != "landsat" and r.modality is Modality.S2_MS


def test_short_sentinel2_band_names(tmp_path):
    """Earth Engine exports (and Sen1Floods11) name bands B2, B8, ...; tools ask for B02, B08, ..."""
    import rasterio
    from satquery.models.seg_net import model_input, variant_for

    names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    path = tmp_path / "chip_S2Hand.tif"
    arr = np.stack([np.full((16, 16), 1000 + 100 * i, np.int16) for i in range(13)])
    with rasterio.open(path, "w", driver="GTiff", height=16, width=16, count=13, dtype="int16") as dst:
        dst.write(arr)
        dst.descriptions = tuple(names)
    r = load_image(path, "s2")
    assert r.has("B02", "B03", "B04", "B08", "B11", "B12")
    assert r.bands[8] == "B8A"
    assert variant_for(r) == "s2"
    assert np.allclose(model_input(r, "s2")[:, 0, 0], [0.11, 0.12, 0.13, 0.17, 0.21, 0.22])
