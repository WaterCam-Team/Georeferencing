"""scan_survey_package: bare-ground DEM filter, cloud axes, view low points."""
import base64
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scan_survey_package import cloud_payload, ground_dem, view_low_points  # noqa: E402


def _grid_points(rng, n=60000, size=6.0):
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)
    return x, y


def test_ground_dem_removes_box_and_flattens_water():
    rng = np.random.default_rng(0)
    x, y = _grid_points(rng)
    z = 100.0 + 0.1 * x                                  # gentle slope, 100.0-100.6
    creek = x < 1.5
    z[creek] = 99.5 + rng.normal(0, 0.02, creek.sum())   # bed below the water surface (99.6)
    box = (np.abs(x - 4) < 0.25) & (np.abs(y - 3) < 0.25)
    z[box] += 0.8                                        # a 0.5 m box on the ground
    dem, st = ground_dem(x, y, z, 0.0, 6.0, 60, 0.1, water_h=99.6)
    row, col = int((6.0 - 3.0) / 0.1), int(4.0 / 0.1)
    assert abs(dem[row, col] - (100.0 + 0.1 * 4.0)) < 0.1      # box gone, ground left
    assert np.allclose(dem[:, :10], 99.6)                      # creek flat at the water surface
    assert st["objects_removed"] > 0 and st["raised_to_water"] > 0
    assert np.isfinite(dem).all()


def test_ground_dem_leaves_big_gaps_empty():
    rng = np.random.default_rng(1)
    x, y = _grid_points(rng)
    keep = ~((x > 2) & (x < 4) & (y > 2) & (y < 4))            # a 2 m hole
    dem, _ = ground_dem(x[keep], y[keep], np.full(keep.sum(), 100.0), 0.0, 6.0, 60, 0.1, water_h=90.0)
    assert np.isnan(dem[30, 30]) and np.isfinite(dem[5, 5])


def test_cloud_axes_east_up_south():
    enu = np.array([[1.0, 2.0, 3.0], [1.001, 2.001, 3.001]])
    out = cloud_payload(enu, np.array([[10, 20, 30], [10, 20, 30]]), 0.015, "s", {})
    xyz = np.frombuffer(base64.b64decode(out["pos_b64"]), np.float32).reshape(-1, 3)
    rgb = np.frombuffer(base64.b64decode(out["col_b64"]), np.uint8).reshape(-1, 3)
    assert out["points"] == 1                                   # same voxel
    assert np.allclose(xyz[0], [1.0005, 3.0005, -2.0005], atol=1e-4)
    assert rgb[0].tolist() == [30, 20, 10]                      # BGR -> RGB


def test_view_low_points_uses_the_heading_wedge():
    rng = np.random.default_rng(2)
    r = rng.uniform(2.5, 4.5, 4000)
    az = np.radians(rng.uniform(0, 360, 4000))
    up = np.where(np.abs(((np.degrees(az) - 225 + 180) % 360) - 180) < 25, -2.0, -0.5)
    low = view_low_points(np.c_[r * np.sin(az), r * np.cos(az), up], 225.0)
    assert abs(low["p50"] + 2.0) < 1e-9
