"""Tests for the ufonet_geo package: lens-model parity with OpenCV and the ground lookup."""

import cv2
import numpy as np
import pytest

from ufonet_geo import geometry as g
from ufonet_geo import ortho
from ufonet_geo.pose import resolve_heading, HeadingUnavailableError

# Placeholder site (not a real deployment).
LAT, LON = 40.7128, -74.0060
W, H = 1296, 972
K = np.array([[1000.0, 0, 652.0], [0, 1005.0, 480.0], [0, 0, 1]])
D = np.array([-0.32, 0.12, 0.001, -0.0008, -0.02])


# ── Lens model vs OpenCV ─────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_undistorted_camera_matrix_matches_opencv(alpha):
    expected, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha, (W, H))
    got = g.undistorted_camera_matrix(K, D, W, H, alpha=alpha)
    np.testing.assert_allclose(got, expected, atol=0.05)


def test_undistort_matches_opencv():
    pts = np.array([[0, 0], [W - 1, 0], [300, 700], [652, 480], [W - 1, H - 1]], dtype=np.float64)
    expected = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    x, y = g.undistort_normalized(pts[:, 0], pts[:, 1], K, D, iterations=5)
    np.testing.assert_allclose(np.stack([x, y], axis=1), expected, atol=1e-6)


def test_project_matches_opencv():
    pts = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 3.0], [0.0, 0.0, 1.0]])
    expected, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    u, v = g.project_to_pixels(pts, K, D)
    np.testing.assert_allclose(np.stack([u, v], axis=1), expected.reshape(-1, 2), atol=1e-6)


def test_enu_latlon_round_trip():
    lat, lon = g.enu_to_latlon(np.array([12.5, -80.0]), np.array([-3.0, 140.0]), LAT, LON)
    e, n = g.latlon_to_enu(lat, lon, LAT, LON)
    np.testing.assert_allclose(e, [12.5, -80.0], atol=1e-9)
    np.testing.assert_allclose(n, [-3.0, 140.0], atol=1e-9)


def test_heading_needs_a_source():
    with pytest.raises(HeadingUnavailableError):
        resolve_heading({}, None, None)
    assert resolve_heading({"heading_deg": 90}, None, 270.0, calib_mag=1) == (90.0, "unit_config")


# ── Ground lookup ────────────────────────────────────────────────────────────

R = g.build_rotation_matrix(heading_deg=30.0, pitch_deg=-25.0, roll_deg=2.0)
K_FRAME = g.undistorted_camera_matrix(K, D, W, H)   # the node's mask frame


def _flat_lookup(**kw):
    kw.setdefault("res_m", 0.5)
    return ortho.build_ground_lookup(K_FRAME, None, W, H, R, LAT, LON, 3.0, **kw)


def test_flat_cells_project_back_to_themselves():
    lk = _flat_lookup()
    rows, cols = np.nonzero(lk.seen())
    assert rows.size > 1000
    pick = np.random.default_rng(0).choice(rows.size, 200, replace=False)
    for r, c in zip(rows[pick], cols[pick]):
        hit = g.pixel_to_world_flat(float(lk.u[r, c]), float(lk.v[r, c]), K_FRAME, R, LAT, LON, 3.0)
        assert hit is not None
        e, n = g.latlon_to_enu(hit[0], hit[1], LAT, LON)
        assert abs(e - (lk.east0 + c * lk.res_m)) < 0.01
        assert abs(n - (lk.north0 - r * lk.res_m)) < 0.01


def test_flat_lookup_respects_grazing_and_range():
    lk = _flat_lookup(max_range_m=60.0)
    rows, cols = np.nonzero(lk.seen())
    e = lk.east0 + cols * lk.res_m
    n = lk.north0 - rows * lk.res_m
    d = np.hypot(e, n)
    assert d.max() <= 60.0
    assert np.degrees(np.arctan2(3.0, d)).min() >= g.MIN_GRAZING_ANGLE_DEG
    # Looking north-east: the seen ground's centroid is to the north-east.
    bearing = np.degrees(np.arctan2(e.mean(), n.mean()))
    assert 15 < bearing < 45


def test_render_near_and_far():
    lk = _flat_lookup()
    mask = np.zeros((120, 160), dtype=np.uint8)   # LoRa-size bitmap
    mask[60:, :] = 1                              # bottom half of the frame is water
    out = lk.render(mask)
    rows, cols = np.nonzero(out)
    water = out[rows, cols] == ortho.WATER
    assert water.any() and (~water).any()
    v = lk.v[rows, cols]
    assert v[water].min() >= H / 2 - 0.5 and v[~water].max() < H / 2 - 0.5
    d = np.hypot(lk.east0 + cols * lk.res_m, lk.north0 - rows * lk.res_m)
    assert d[water].mean() < d[~water].mean()      # the bottom of the frame is the near ground
    assert lk.area_m2(out) == pytest.approx(water.sum() * 0.25)
    assert np.count_nonzero(out) == np.count_nonzero(lk.seen())


def test_render_rejects_other_aspect():
    with pytest.raises(ValueError):
        _flat_lookup().render(np.zeros((100, 100)))


def test_bounds_are_north_up_rectangle():
    lk = _flat_lookup()
    south, west, north, east = lk.bounds()
    rows, cols = lk.shape
    e_w, n_s = g.latlon_to_enu(north, east, LAT, LON)
    e0, n0 = g.latlon_to_enu(south, west, LAT, LON)
    assert e_w - e0 == pytest.approx(cols * lk.res_m, abs=1e-6)
    assert n_s - n0 == pytest.approx(rows * lk.res_m, abs=1e-6)


def test_lookup_round_trips_bytes():
    lk = _flat_lookup()
    back = ortho.GroundLookup.from_bytes(lk.to_bytes())
    np.testing.assert_array_equal(back.u, lk.u)
    assert back.bounds() == lk.bounds() and back.meta == lk.meta


def test_dem_lookup_hides_ground_behind_a_ridge():
    # Flat DEM at 100 m with a 1 m ridge (below the 2 m camera) 20-22 m north of the camera.
    def sampler(lat, lon):
        _, n = g.latlon_to_enu(lat, lon, LAT, LON)
        return np.where((n >= 20) & (n <= 22), 101.0, 100.0)

    Rn = g.build_rotation_matrix(0.0, -15.0, 0.0)
    lk = ortho.build_ground_lookup(K_FRAME, None, W, H, Rn, LAT, LON, 2.0,
                                   dem_sampler=sampler, res_m=0.5, max_range_m=80)
    flat = ortho.build_ground_lookup(K_FRAME, None, W, H, Rn, LAT, LON, 2.0,
                                     res_m=0.5, max_range_m=80)

    def seen_north(lookup, lo, hi):
        rows, cols = np.nonzero(lookup.seen())
        n = lookup.north0 - rows * lookup.res_m
        e = lookup.east0 + cols * lookup.res_m
        return np.count_nonzero((n > lo) & (n < hi) & (np.abs(e) < 2))

    assert seen_north(lk, 10, 19) > 0            # in front of the ridge: seen
    assert seen_north(lk, 20, 22) > 0            # the ridge itself: seen
    assert seen_north(lk, 23, 35) == 0           # in its shadow: hidden
    assert seen_north(flat, 23, 35) > 0          # flat ground would have shown it


def test_sample_dem_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from pyproj import Transformer
    from rasterio.transform import from_origin

    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform(LON, LAT)
    data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    data[0, 0] = -9999
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=100, height=100, count=1, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(x0 - 50, y0 + 50, 1.0, 1.0),
                       nodata=-9999) as dst:
        dst.write(data, 1)

    from ufonet_geo.terrain import sample_dem_grid
    lat, lon = g.enu_to_latlon(np.array([0.2, -49.8, 500.0]), np.array([0.2, 49.8, 0.0]), LAT, LON)
    got = sample_dem_grid(str(path), lat, lon, method="nearest")
    assert got[0] == data[49, 50]
    assert np.isnan(got[1])          # nodata corner
    assert np.isnan(got[2])          # outside the DEM


def test_auto_cell_size_fits_a_low_camera():
    # 0.84 m up, 30 deg down: the camera sees only to the grazing limit, ~24 m.
    Rl = g.build_rotation_matrix(70.0, -30.0, 0.0)
    coarse = ortho.build_ground_lookup(K_FRAME, None, W, H, Rl, LAT, LON, 0.84, res_m=0.5)
    auto = ortho.build_ground_lookup_auto(K_FRAME, None, W, H, Rl, LAT, LON, 0.84)
    assert auto.res_m == pytest.approx(auto.meta["max_range_m"] / 300)   # ~300 cells out to the farthest ground
    assert np.count_nonzero(auto.seen()) > 0.7 * (0.5 / auto.res_m) ** 2 * np.count_nonzero(coarse.seen())
    # The seen area agrees, and is now measured in small cells rather than a few big ones.
    assert np.count_nonzero(auto.seen()) * auto.res_m ** 2 == pytest.approx(
        np.count_nonzero(coarse.seen()) * 0.25, rel=0.35)
    assert max(auto.shape) <= 2 * 300 + 4


def test_auto_cell_size_never_coarser_than_the_coarse_pass():
    auto = ortho.build_ground_lookup_auto(K_FRAME, None, W, H, R, LAT, LON, 3.0)
    assert auto.res_m <= 0.5



def test_sample_dem_grid_bilinear_follows_a_slope(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from ufonet_geo.terrain import sample_dem_grid

    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True).transform(LON, LAT)
    cols = np.arange(50, dtype=np.float32)
    data = np.tile(cols * 0.3, (50, 1))         # rises 0.3 m per 1 m cell to the east
    path = tmp_path / "slope.tif"
    with rasterio.open(path, "w", driver="GTiff", width=50, height=50, count=1, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(x0 - 25, y0 + 25, 1.0, 1.0)) as dst:
        dst.write(data, 1)
    e = np.linspace(-10, 10, 81)
    lat, lon = g.enu_to_latlon(e, np.zeros_like(e), LAT, LON)
    z = sample_dem_grid(str(path), lat, lon)
    steps = np.diff(z)
    assert np.allclose(steps, steps.mean(), atol=0.02)   # smooth, no 0.3 m stairs
    assert steps.mean() == pytest.approx(0.3 * 0.25, rel=0.05)
    zn = sample_dem_grid(str(path), lat, lon, method="nearest")
    assert np.abs(np.diff(zn)).max() == pytest.approx(0.3, abs=1e-4)
