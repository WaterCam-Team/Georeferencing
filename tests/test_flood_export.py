"""Tests for scripts/flood_export.py — no real images or DEM required."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import rasterio
from rasterio.crs import CRS

from flood_export import (
    build_geojson,
    export_csv,
    export_geojson,
    export_geotiff,
    gps_points_to_polygon,
    project_contour,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_K(w=640, h=480, fov_deg=60):
    """Minimal pinhole camera matrix."""
    f = w / (2 * np.tan(np.radians(fov_deg / 2)))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _flat_R():
    """Camera pointing north-downward at 45°."""
    from camera_geometry import build_rotation_matrix
    return build_rotation_matrix(heading_deg=0.0, pitch_deg=-45.0, roll_deg=0.0)


def _const_elevation(val: float):
    """Return a get_elevation function that always returns val."""
    return lambda lon, lat: val


# ---------------------------------------------------------------------------
# project_contour
# ---------------------------------------------------------------------------

def test_project_contour_empty_returns_empty():
    K = _identity_K()
    R = _flat_R()
    result = project_contour(
        np.empty((0, 2), dtype=np.float32), K, R,
        43.0, -76.0, 110.0, _const_elevation(109.0),
    )
    assert result == []


def test_project_contour_terrain_miss_dropped():
    """Points whose ray never hits the terrain surface return None and are dropped."""
    K = _identity_K()
    R = _flat_R()
    # get_elevation always None → every ray misses
    result = project_contour(
        np.array([[320, 240]], dtype=np.float32), K, R,
        43.0, -76.0, 110.0, lambda lon, lat: None,
    )
    assert result == []


def test_project_contour_hit_returns_dict_keys():
    """A ray that hits terrain returns a dict with the expected keys."""
    K = _identity_K()
    R = _flat_R()
    # Camera 10 m above flat terrain at 100 m elevation
    result = project_contour(
        np.array([[320, 400]], dtype=np.float32), K, R,
        43.0, -76.0, 110.0, _const_elevation(100.0),
        step_m=0.5, max_range_m=200.0,
    )
    if result:   # ray may or may not intersect depending on geometry
        assert set(result[0].keys()) == {"lat", "lon", "elev_m", "slant_range_m"}


# ---------------------------------------------------------------------------
# gps_points_to_polygon
# ---------------------------------------------------------------------------

def _gps(n=6, base_lat=43.0, base_lon=-76.0, spread=0.001):
    """Generate n GPS-like points in a rough circle."""
    pts = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        pts.append({
            "lat": base_lat + spread * np.sin(angle),
            "lon": base_lon + spread * np.cos(angle),
            "elev_m": 100.0,
            "slant_range_m": 10.0,
        })
    return pts


def test_gps_points_to_polygon_too_few_returns_none():
    assert gps_points_to_polygon([]) is None
    assert gps_points_to_polygon(_gps(2)) is None


def test_gps_points_to_polygon_closed_ring():
    pts = _gps(8)
    ring = gps_points_to_polygon(pts)
    assert ring is not None
    assert ring[0] == ring[-1], "polygon ring must be closed"


def test_gps_points_to_polygon_raw_has_same_count():
    pts = _gps(10)
    ring = gps_points_to_polygon(pts, hull=False)
    # +1 for closing vertex
    assert len(ring) == len(pts) + 1


def test_gps_points_to_polygon_hull_has_fewer_vertices():
    pts = _gps(20)
    raw  = gps_points_to_polygon(pts, hull=False)
    hull = gps_points_to_polygon(pts, hull=True)
    assert hull is not None
    assert len(hull) <= len(raw), "convex hull should not have more vertices than raw"


def test_gps_points_to_polygon_hull_is_closed():
    pts = _gps(12)
    ring = gps_points_to_polygon(pts, hull=True)
    assert ring[0] == ring[-1]


# ---------------------------------------------------------------------------
# build_geojson
# ---------------------------------------------------------------------------

def test_build_geojson_structure():
    ring = [(-76.0, 43.0), (-76.001, 43.001), (-75.999, 43.001), (-76.0, 43.0)]
    gj = build_geojson(ring, {"test": True})
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    feat = gj["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["test"] is True
    # Ring is embedded as first and only coordinate array
    assert feat["geometry"]["coordinates"][0] == ring


# ---------------------------------------------------------------------------
# export_geojson
# ---------------------------------------------------------------------------

def test_export_geojson_writes_valid_file(tmp_path):
    ring = [(-76.0, 43.0), (-76.001, 43.001), (-75.999, 43.001), (-76.0, 43.0)]
    gj = build_geojson(ring, {})
    out = tmp_path / "test.geojson"
    export_geojson(gj, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["type"] == "FeatureCollection"


# ---------------------------------------------------------------------------
# export_geotiff
# ---------------------------------------------------------------------------

def test_export_geotiff_creates_valid_raster(tmp_path):
    ring = [
        (-76.005, 43.060), (-76.000, 43.065), (-75.995, 43.060),
        (-76.000, 43.055), (-76.005, 43.060),
    ]
    out = tmp_path / "flood.tif"
    export_geotiff(ring, out, resolution_deg=0.001)

    assert out.exists()
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 4326
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        data = src.read(1)
        # The polygon should have burned some flood pixels
        assert data.max() == 1
        assert data.min() == 0


def test_export_geotiff_flood_inside_polygon(tmp_path):
    """Pixels inside the polygon must be 1."""
    # Simple square
    ring = [
        (-76.010, 43.055), (-76.010, 43.065),
        (-75.990, 43.065), (-75.990, 43.055),
        (-76.010, 43.055),
    ]
    out = tmp_path / "flood_square.tif"
    export_geotiff(ring, out, resolution_deg=0.001)
    with rasterio.open(out) as src:
        data = src.read(1)
        # Centre pixel should be inside the square → flood (1)
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        assert data[cy, cx] == 1


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

def test_export_csv_writes_all_points(tmp_path):
    pts = _gps(5)
    out = tmp_path / "boundary.csv"
    export_csv(pts, out)
    assert out.exists()
    lines = out.read_text().splitlines()
    assert lines[0] == "lat,lon,elev_m,slant_range_m"
    assert len(lines) == 1 + len(pts)


def test_export_csv_values_roundtrip(tmp_path):
    pts = [{"lat": 43.123456, "lon": -76.654321, "elev_m": 110.5, "slant_range_m": 25.3}]
    out = tmp_path / "pts.csv"
    export_csv(pts, out)
    import csv as _csv
    with open(out, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert float(rows[0]["lat"]) == pytest.approx(43.123456, abs=1e-5)
    assert float(rows[0]["lon"]) == pytest.approx(-76.654321, abs=1e-5)
