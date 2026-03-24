"""Tests for GCP module: load/save, TPS requirements, pose refinement with synthetic GCPs."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from camera_geometry import build_rotation_matrix
from geo_core import pixel_to_world_flat
from gcp import (
    GroundControlPoint,
    load_gcps,
    save_gcps,
    refine_pose_from_gcps,
    fit_tps_warp,
    gcp_residuals,
)


def test_gcp_save_load_roundtrip():
    gcps = [
        GroundControlPoint("A", 100.0, 200.0, 43.0, -76.0, None),
        GroundControlPoint("B", 300.0, 400.0, 43.001, -75.999, 120.5),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    try:
        save_gcps(path, gcps)
        loaded = load_gcps(path)
        assert len(loaded) == 2
        assert loaded[0].label == "A"
        assert loaded[0].pixel_u == 100.0 and loaded[0].lat == 43.0
        assert loaded[1].elev_m == 120.5
    finally:
        Path(path).unlink(missing_ok=True)


def test_fit_tps_returns_none_for_few_gcps():
    gcps = [GroundControlPoint(f"g{i}", 10 * i, 10 * i, 43 + i * 0.001, -76, None) for i in range(5)]
    assert fit_tps_warp(gcps) is None


def test_fit_tps_returns_callable_for_six_gcps():
    # Use non-collinear pixel positions so TPS is well-posed
    pts = [(100, 100), (400, 150), (200, 400), (500, 450), (150, 500), (450, 300)]
    gcps = [GroundControlPoint(f"g{i}", u, v, 43 + i * 0.001, -76 + i * 0.001, None) for i, (u, v) in enumerate(pts)]
    warp = fit_tps_warp(gcps)
    assert warp is not None
    lat, lon = warp(100, 100)
    assert abs(lat - 43) < 0.1 and abs(lon - (-76)) < 0.1


def test_refine_pose_with_consistent_gcps():
    """With GCPs generated from a known pose, refinement should recover that pose."""
    K = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]], dtype=np.float64)
    cam_lat, cam_lon = 43.05, -76.12
    height = 5.0
    heading, pitch, roll = 10.0, -60.0, 0.0
    R_true = build_rotation_matrix(heading, pitch, roll)

    # Generate 4 GCPs from this pose
    pixels = [(500, 300), (800, 400), (400, 600), (900, 500)]
    gcps = []
    for i, (u, v) in enumerate(pixels):
        pred = pixel_to_world_flat(u, v, K, R_true, cam_lat, cam_lon, height)
        assert pred is not None
        gcps.append(GroundControlPoint(f"P{i}", u, v, pred[0], pred[1], None))

    # Refine from slightly perturbed pose
    cam_lat_0 = cam_lat + 0.0001
    cam_lon_0 = cam_lon - 0.0001
    height_0 = height + 0.3
    heading_0 = heading + 1.0
    pitch_0 = pitch - 0.5
    roll_0 = roll

    out = refine_pose_from_gcps(
        K, gcps,
        cam_lat_0, cam_lon_0, height_0,
        heading_0, pitch_0, roll_0,
        bounds=True,
    )
    cam_lat_r, cam_lon_r, height_r, R_r, rms_deg, _, _, _ = out

    assert not np.isnan(rms_deg)
    assert rms_deg < 1e-5  # small residuals
    assert abs(cam_lat_r - cam_lat) < 1e-4
    assert abs(cam_lon_r - cam_lon) < 1e-4
    assert abs(height_r - height) < 0.5  # height can be coupled with orientation


def test_gcp_residuals():
    # Straight-down camera so image centre = nadir
    K = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]], dtype=np.float64)
    R = build_rotation_matrix(0, -90, 0)  # -90 = straight down
    cam_lat, cam_lon, height = 43.0, -76.0, 4.0
    gcps = [
        GroundControlPoint("C", 960, 540, 43.0, -76.0, None),  # centre = nadir
    ]
    res = gcp_residuals(K, R, cam_lat, cam_lon, height, gcps)
    assert len(res) == 1
    lat_err, lon_err, dist_m = res[0]
    assert abs(lat_err) < 1e-9 and abs(lon_err) < 1e-9
    assert dist_m < 0.01
