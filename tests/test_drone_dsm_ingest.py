"""Tests for scripts/drone_dsm_ingest.py — no real drone data needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root and scripts/ to path so we can import both
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from drone_dsm_ingest import ingest_dsm, IngestResult


def _write_synthetic_raster(
    path: Path,
    epsg: int | None,
    width: int = 20,
    height: int = 20,
    z_base: float = 110.0,
    nodata: float = -9999.0,
    nodata_frac: float = 0.0,
    use_nan_nodata: bool = False,
) -> Path:
    """Write a tiny synthetic single-band GeoTIFF."""
    crs = CRS.from_epsg(epsg) if epsg is not None else None

    if epsg == 6347:
        left, bottom, right, top = 380000.0, 4762000.0, 380020.0, 4762020.0
    else:
        left, bottom, right, top = -76.14, 43.04, -76.13, 43.05

    transform = from_bounds(left, bottom, right, top, width, height)
    data = np.full((height, width), z_base, dtype=np.float32)

    actual_nodata = float("nan") if use_nan_nodata else nodata
    if nodata_frac > 0:
        n_nd = max(1, int(width * height * nodata_frac))
        data.ravel()[:n_nd] = actual_nodata

    kwargs = dict(
        driver="GTiff", height=height, width=width,
        count=1, dtype="float32", transform=transform,
        nodata=actual_nodata,
    )
    if crs is not None:
        kwargs["crs"] = crs

    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data, 1)

    return path


# ---------------------------------------------------------------------------
# CRS handling
# ---------------------------------------------------------------------------

def test_already_correct_crs_no_reproject(tmp_path):
    src = _write_synthetic_raster(tmp_path / "dem_6347.tif", epsg=6347)
    out_dir = tmp_path / "out"
    result = ingest_dsm(src, out_dir, target_epsg=6347, skip_reproject=False)

    assert result.reprojected is False
    assert result.out_path == src
    assert result.src_epsg == 6347
    assert result.checks_passed


def test_wrong_crs_reprojects(tmp_path):
    src = _write_synthetic_raster(tmp_path / "dem_4326.tif", epsg=4326)
    out_dir = tmp_path / "out"
    result = ingest_dsm(src, out_dir, target_epsg=6347, target_res=1.0)

    assert result.reprojected is True
    assert result.out_path is not None
    assert result.out_path.exists()
    assert result.src_epsg == 4326

    with rasterio.open(result.out_path) as dst:
        assert dst.crs.to_epsg() == 6347


def test_skip_reproject_leaves_source(tmp_path):
    src = _write_synthetic_raster(tmp_path / "dem_4326.tif", epsg=4326)
    out_dir = tmp_path / "out"
    result = ingest_dsm(src, out_dir, target_epsg=6347, skip_reproject=True)

    assert result.reprojected is False
    assert result.out_path == src


# ---------------------------------------------------------------------------
# Elevation stats
# ---------------------------------------------------------------------------

def test_elevation_range_reported(tmp_path):
    src = _write_synthetic_raster(tmp_path / "dem.tif", epsg=6347, z_base=112.5)
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    assert result.z_min is not None
    assert result.z_max is not None
    assert abs(result.z_min - 112.5) < 0.01
    assert abs(result.z_max - 112.5) < 0.01


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_fraction_with_nodata(tmp_path):
    src = _write_synthetic_raster(
        tmp_path / "dem_partial.tif", epsg=6347,
        width=10, height=10,
        nodata_frac=0.5,
    )
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    assert result.coverage_frac is not None
    # 50% nodata → coverage should be around 50%
    assert 0.40 <= result.coverage_frac <= 0.60


def test_full_nodata_raster_fails(tmp_path):
    """A raster where every cell is nodata should report a failure."""
    src = _write_synthetic_raster(
        tmp_path / "dem_empty.tif", epsg=6347,
        nodata_frac=1.0,
    )
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    assert "no valid cells" in result.failures
    assert result.checks_passed is False


# ---------------------------------------------------------------------------
# Vertical datum inference
# ---------------------------------------------------------------------------

def test_vertical_hint_unknown_returns_ellipsoid_default(tmp_path):
    """UTM EPSG:6347 has no embedded vertical datum → ingest should default to ellipsoidal."""
    src = _write_synthetic_raster(tmp_path / "dem.tif", epsg=6347)
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    # Should not be None; pipeline needs a datum hint
    assert result.vertical_hint is not None


# ---------------------------------------------------------------------------
# NaN nodata (Copilot bug fix regression tests)
# ---------------------------------------------------------------------------

def test_nan_nodata_cells_excluded_from_stats(tmp_path):
    """Rasters with NaN as the nodata value must not count NaN cells as valid."""
    src = _write_synthetic_raster(
        tmp_path / "dem_nan.tif", epsg=6347,
        z_base=110.0, nodata_frac=0.5, use_nan_nodata=True,
    )
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    assert result.coverage_frac is not None
    # ~50% of cells are NaN nodata → coverage should be around 50%, not ~100%
    assert 0.40 <= result.coverage_frac <= 0.60


def test_nan_nodata_all_invalid_reports_failure(tmp_path):
    """All-NaN raster should report failure even when nodata declared as NaN."""
    src = _write_synthetic_raster(
        tmp_path / "dem_all_nan.tif", epsg=6347,
        nodata_frac=1.0, use_nan_nodata=True,
    )
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    assert "no valid cells" in result.failures
    assert result.checks_passed is False


# ---------------------------------------------------------------------------
# Missing CRS (Copilot bug fix regression test)
# ---------------------------------------------------------------------------

def test_missing_crs_skips_reproject_and_fails(tmp_path):
    """A raster with no CRS should not attempt reprojection (would crash) and should fail."""
    src = _write_synthetic_raster(tmp_path / "dem_nocrs.tif", epsg=None)
    result = ingest_dsm(src, tmp_path / "out", target_epsg=6347)

    assert result.reprojected is False
    assert result.checks_passed is False
    assert any("CRS" in f or "crs" in f.lower() for f in result.failures)


# ---------------------------------------------------------------------------
# Nodata preserved in reprojected output
# ---------------------------------------------------------------------------

def test_nodata_preserved_in_reprojected_output(tmp_path):
    """The reprojected GeoTIFF must carry the same nodata value as the source."""
    src = _write_synthetic_raster(
        tmp_path / "dem_4326.tif", epsg=4326,
        z_base=110.0, nodata=-9999.0,
    )
    out_dir = tmp_path / "out"
    result = ingest_dsm(src, out_dir, target_epsg=6347, target_res=1.0)

    assert result.reprojected is True
    assert result.out_path is not None
    with rasterio.open(result.out_path) as dst:
        assert dst.nodata == pytest.approx(-9999.0)


# ---------------------------------------------------------------------------
# Combined sun elevation + month filter (both active simultaneously)
# ---------------------------------------------------------------------------

def test_filter_and_sort_combined(tmp_path):
    """Ingest checks_passed is True when raster is valid and already in target CRS."""
    src = _write_synthetic_raster(tmp_path / "dem.tif", epsg=6347, z_base=111.0)
    result = ingest_dsm(src, tmp_path / "out", skip_reproject=True)

    assert result.checks_passed is True
    assert result.z_min == pytest.approx(111.0, abs=0.01)
    assert result.z_max == pytest.approx(111.0, abs=0.01)
    assert result.coverage_frac == pytest.approx(1.0, abs=0.01)
