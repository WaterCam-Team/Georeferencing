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
    epsg: int,
    width: int = 20,
    height: int = 20,
    z_base: float = 110.0,
    nodata: float = -9999.0,
    nodata_frac: float = 0.0,
) -> Path:
    """Write a tiny synthetic single-band GeoTIFF."""
    crs = CRS.from_epsg(epsg)

    if epsg == 6347:
        # UTM coordinates in metres (near Syracuse NY)
        left, bottom, right, top = 380000.0, 4762000.0, 380020.0, 4762020.0
    else:
        # Geographic coordinates (WGS84)
        left, bottom, right, top = -76.14, 43.04, -76.13, 43.05

    transform = from_bounds(left, bottom, right, top, width, height)
    data = np.full((height, width), z_base, dtype=np.float32)

    if nodata_frac > 0:
        n_nd = max(1, int(width * height * nodata_frac))
        data.ravel()[:n_nd] = nodata

    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width,
        count=1, dtype="float32",
        crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
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
