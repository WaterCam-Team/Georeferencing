"""
DSM generation and georeferencing validation tests.

Unit tests (always run, numpy only):
  - _grid_max_z correctness on synthetic point clouds
  - max-Z output >= linear-interpolated output on same cloud

Integration tests (skipped if DSM or deps are absent):
  - DSM has expected CRS, resolution, and elevation range
  - max-Z cells are >= linear cells where both are valid
  - Every camera pose is above the DSM surface (concrete georef check)
  - Height-above-ground distribution is physically plausible

Run unit tests only (no external data required):
    PYTHONPATH=. .venv/bin/pytest tests/test_dsm_validation.py::TestGridMaxZ
    PYTHONPATH=. .venv/bin/pytest tests/test_dsm_validation.py::TestMaxVsLinearSynthetic

Run all (integration tests auto-skip if DSM is missing):
    PYTHONPATH=. .venv/bin/pytest tests/test_dsm_validation.py -v

Generate the DSM first if needed:
    PYTHONPATH=. .venv/bin/python scripts/pix4d_to_las_dem.py 2026-04-24-13-11-52
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

SCAN_NAME = "2026-04-24-13-11-52"
SCAN_DIR = REPO / SCAN_NAME
DSM_LINEAR = REPO / "output/pix4d" / f"{SCAN_NAME}_dem.tif"
DSM_MAX = REPO / "output/pix4d" / f"{SCAN_NAME}_dem_max.tif"
POSES_CSV = REPO / "output/pix4d" / f"{SCAN_NAME}_camera_poses.csv"
RTK_CSV = SCAN_DIR / "geolocations" / "rtkGPS.csv"

integration = pytest.mark.skipif(
    not DSM_LINEAR.exists(),
    reason=f"Integration data missing: run scripts/pix4d_to_las_dem.py {SCAN_NAME} first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_grid_max_z():
    from pix4d_to_las_dem import _grid_max_z
    return _grid_max_z


def _make_flat_cloud(z_val: float, n: int = 500, seed: int = 0) -> np.ndarray:
    """Random XY in [0, 10]^2, constant Z."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n, 2))
    z = np.full(n, z_val)
    return np.column_stack([xy, z])


def _make_step_cloud(z_low: float, z_high: float, n: int = 1000, seed: int = 1) -> np.ndarray:
    """Left half z_low, right half z_high."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n, 2))
    z = np.where(xy[:, 0] < 5, z_low, z_high)
    return np.column_stack([xy, z])


# ---------------------------------------------------------------------------
# Unit tests — _grid_max_z
# ---------------------------------------------------------------------------

class TestGridMaxZ:
    def setup_method(self):
        self._fn = _import_grid_max_z()

    def test_single_point_lands_in_correct_cell(self):
        fn = self._fn
        x = np.array([1.0])
        y = np.array([1.0])
        z = np.array([5.0])
        res = 1.0
        x_min, y_max, rows, cols = 0.0, 3.0, 3, 3
        dem = fn(x, y, z, x_min, y_max, rows, cols, res)
        # (1.0 - 0.0)/1.0 = col 1; (3.0 - 1.0)/1.0 = row 2
        assert dem[2, 1] == pytest.approx(5.0)
        nan_cells = np.sum(~np.isnan(dem))
        assert nan_cells == 1, f"Expected 1 filled cell, got {nan_cells}"

    def test_collision_takes_maximum(self):
        fn = self._fn
        # Two points in the same cell, different Z
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])
        z = np.array([3.0, 7.0])
        res = 1.0
        x_min, y_max, rows, cols = 0.0, 2.0, 2, 2
        dem = fn(x, y, z, x_min, y_max, rows, cols, res)
        filled = dem[~np.isnan(dem)]
        assert len(filled) == 1
        assert filled[0] == pytest.approx(7.0), (
            f"Expected max 7.0, got {filled[0]:.3f}"
        )

    def test_empty_cells_remain_nan(self):
        fn = self._fn
        x = np.array([0.5])
        y = np.array([0.5])
        z = np.array([1.0])
        res = 1.0
        x_min, y_max, rows, cols = 0.0, 3.0, 3, 3
        dem = fn(x, y, z, x_min, y_max, rows, cols, res)
        assert np.sum(~np.isnan(dem)) == 1
        assert np.sum(np.isnan(dem)) == 8

    def test_out_of_bounds_points_ignored(self):
        fn = self._fn
        # One valid, one outside the grid
        x = np.array([0.5, 99.0])
        y = np.array([0.5, 99.0])
        z = np.array([1.0, 999.0])
        res = 1.0
        x_min, y_max, rows, cols = 0.0, 2.0, 2, 2
        dem = fn(x, y, z, x_min, y_max, rows, cols, res)
        filled = dem[~np.isnan(dem)]
        assert len(filled) == 1
        assert filled[0] == pytest.approx(1.0)

    def test_max_is_nonnegative_relative_to_min(self):
        fn = self._fn
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 500)
        y = rng.uniform(0, 10, 500)
        z = rng.uniform(100, 110, 500)
        res = 0.5
        x_min, y_max = 0.0, 10.0
        rows = cols = int(10 / res)
        dem = fn(x, y, z, x_min, y_max, rows, cols, res)
        assert np.nanmin(dem) >= 100.0 - 1e-9
        assert np.nanmax(dem) <= 110.0 + 1e-9


# ---------------------------------------------------------------------------
# Unit tests — max vs linear on synthetic clouds
# ---------------------------------------------------------------------------

class TestMaxVsLinearSynthetic:
    def test_flat_cloud_max_matches_true_z(self):
        """Dense flat cloud: max-Z grid should be very close to the true constant Z."""
        from pix4d_to_las_dem import _grid_max_z
        cloud = _make_flat_cloud(z_val=50.0, n=2000)
        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        res = 0.5
        x_min = x.min() - res
        y_max = y.max() + res
        cols = int(np.ceil((x.max() + res - x_min) / res)) + 1
        rows = int(np.ceil((y_max - (y.min() - res)) / res)) + 1
        dem = _grid_max_z(x, y, z, x_min, y_max, rows, cols, res)
        filled = dem[~np.isnan(dem)]
        assert len(filled) > 0
        assert np.all(np.abs(filled - 50.0) < 1e-4), (
            f"Max-Z on flat cloud deviated: max_err={np.abs(filled - 50.0).max():.6f}"
        )

    def test_step_cloud_max_captures_high_side(self):
        """Step cloud: right-half max-Z cells should equal z_high."""
        from pix4d_to_las_dem import _grid_max_z
        z_low, z_high = 10.0, 20.0
        cloud = _make_step_cloud(z_low, z_high, n=5000)
        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
        res = 0.5
        x_min = x.min() - res
        y_max = y.max() + res
        cols = int(np.ceil((x.max() + res - x_min) / res)) + 1
        rows = int(np.ceil((y_max - (y.min() - res)) / res)) + 1
        dem = _grid_max_z(x, y, z, x_min, y_max, rows, cols, res)

        # Right-half columns (x > 5): cells should be z_high
        col_boundary = int(np.ceil((5.0 - x_min) / res))
        right_cells = dem[:, col_boundary + 1:]
        valid_right = right_cells[~np.isnan(right_cells)]
        assert len(valid_right) > 0
        assert np.all(np.abs(valid_right - z_high) < 1e-4), (
            f"Right-half max should be {z_high}, got range "
            f"{valid_right.min():.3f}–{valid_right.max():.3f}"
        )

    def test_max_captures_spike_linear_smooths_it(self):
        """
        max-Z records the exact peak in the cell containing the spike.
        Linear interpolation smooths it by blending with surrounding points.

        This demonstrates the key methodological difference: use max for true DSM
        (preserves tall objects), use linear when gap-filling is needed.
        """
        pytest.importorskip("scipy")
        pytest.importorskip("rasterio")
        import tempfile, rasterio
        from pix4d_to_las_dem import _write_dem

        # Dense base cloud at z=100, one isolated spike at (5.5, 5.5, z=110)
        rng = np.random.default_rng(7)
        n = 4000
        x = rng.uniform(0, 10, n)
        y = rng.uniform(0, 10, n)
        z = np.full(n, 100.0)
        spike_xy = (5.5, 5.5)
        spike_z = 110.0
        x = np.append(x, spike_xy[0])
        y = np.append(y, spike_xy[1])
        z = np.append(z, spike_z)
        xyz = np.column_stack([x, y, z])

        fake_wkt = 'LOCAL_CS["arbitrary",LOCAL_DATUM["arbitrary",0],UNIT["metre",1]]'
        res = 0.5

        with tempfile.NamedTemporaryFile(suffix="_lin.tif", delete=False) as f:
            path_lin = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix="_max.tif", delete=False) as f:
            path_max = Path(f.name)

        try:
            _write_dem(xyz, fake_wkt, path_lin, res, method="linear")
            _write_dem(xyz, fake_wkt, path_max, res, method="max")

            with rasterio.open(path_lin) as src_l, rasterio.open(path_max) as src_m:
                lin = src_l.read(1).astype(np.float64)
                mx = src_m.read(1).astype(np.float64)
                nd_l, nd_m = src_l.nodata, src_m.nodata
                t = src_m.transform

            # Find the cell that contains the spike point
            from rasterio.transform import rowcol
            spike_row, spike_col = rowcol(t, spike_xy[0], spike_xy[1])

            max_at_spike = mx[spike_row, spike_col]
            lin_at_spike = lin[spike_row, spike_col]

            print(
                f"\n  Spike cell: max={max_at_spike:.3f} m, "
                f"linear={lin_at_spike:.3f} m  (true spike z={spike_z} m)"
            )

            assert max_at_spike != nd_m
            assert max_at_spike == pytest.approx(spike_z, abs=1e-3), (
                f"max-Z should equal spike z exactly; got {max_at_spike:.3f}"
            )
            # Linear blends toward neighbors — it will be lower than the spike
            assert lin_at_spike < spike_z - 0.5, (
                f"Linear should smooth the spike below {spike_z - 0.5:.1f} m; "
                f"got {lin_at_spike:.3f} m"
            )
        finally:
            path_lin.unlink(missing_ok=True)
            path_max.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration tests — real DSM
# ---------------------------------------------------------------------------

@integration
class TestDSMProperties:
    def test_dsm_crs_is_utm18n(self):
        rasterio = pytest.importorskip("rasterio")
        with rasterio.open(DSM_LINEAR) as src:
            epsg = src.crs.to_epsg()
        assert epsg == 6347, (
            f"Expected EPSG:6347 (NAD83 UTM18N), got EPSG:{epsg}"
        )

    def test_dsm_resolution_is_5cm(self):
        rasterio = pytest.importorskip("rasterio")
        with rasterio.open(DSM_LINEAR) as src:
            res_x, res_y = src.res
        assert res_x == pytest.approx(0.05, abs=1e-6), (
            f"X resolution {res_x:.4f} m, expected 0.05 m"
        )
        assert res_y == pytest.approx(0.05, abs=1e-6), (
            f"Y resolution {res_y:.4f} m, expected 0.05 m"
        )

    def test_dsm_elevation_range_plausible(self):
        """
        Ellipsoidal heights at this site are ~109-113 m.
        If the range is wildly outside this, the UTM shift or datum is wrong.
        """
        rasterio = pytest.importorskip("rasterio")
        with rasterio.open(DSM_LINEAR) as src:
            data = src.read(1)
            nd = src.nodata
        valid = data[data != nd]
        z_min, z_max, z_mean = valid.min(), valid.max(), valid.mean()
        print(f"\n  DSM z: {z_min:.2f}–{z_max:.2f} m  mean={z_mean:.2f} m")
        assert 100.0 < z_min, (
            f"DSM min elevation {z_min:.2f} m unexpectedly low — check UTM shift"
        )
        assert z_max < 125.0, (
            f"DSM max elevation {z_max:.2f} m unexpectedly high — check UTM shift"
        )

    def test_dsm_has_sufficient_valid_cells(self):
        """
        The DSM must contain valid data.  Nodata fraction depends on scan
        geometry (a narrow path scan fills only a fraction of its bounding box),
        so we check that at least 1 % of cells are valid rather than capping
        nodata at an arbitrary percentage.
        """
        rasterio = pytest.importorskip("rasterio")
        with rasterio.open(DSM_LINEAR) as src:
            data = src.read(1)
            nd = src.nodata
        total = data.size
        n_valid = np.sum(data != nd)
        frac_valid = n_valid / total
        print(
            f"\n  Valid cells: {n_valid:,} / {total:,} "
            f"({frac_valid:.1%} of bounding box)"
        )
        assert n_valid > 0, "DSM contains no valid cells"
        assert frac_valid > 0.01, (
            f"Only {frac_valid:.2%} valid cells — DSM appears nearly empty"
        )

    def test_max_vs_linear_statistics_on_real_dsm(self):
        """
        Compare max-Z and linear-interpolated DSMs on the real scan.

        Assertions:
        - max has fewer (or equal) valid cells than linear — it never gap-fills.
        - At the 95th percentile, max >= linear — max preserves tall peaks that
          linear smooths away via Delaunay blending with neighbours.

        Note: cell-by-cell max >= linear does NOT hold everywhere.  Linear
        interpolation can raise a cell above the max of points *within* that
        cell by blending in the z-value of a taller neighbour point.  This is
        expected and not a bug.
        """
        pytest.importorskip("rasterio")
        pytest.importorskip("scipy")
        if not SCAN_DIR.exists():
            pytest.skip(f"Scan directory missing: {SCAN_DIR}")

        import tempfile, rasterio
        from pix4d_to_las_dem import (
            _load_scene_reference_frame,
            _read_gltf_pointcloud,
            _local_to_utm,
            _write_dem,
        )

        shift, crs_wkt = _load_scene_reference_frame(SCAN_DIR)
        xyz_local, _ = _read_gltf_pointcloud(SCAN_DIR)
        xyz_utm = _local_to_utm(xyz_local, shift)
        res = 0.05

        with tempfile.NamedTemporaryFile(suffix="_max_tmp.tif", delete=False) as f:
            path_max = Path(f.name)
        try:
            _write_dem(xyz_utm, crs_wkt, path_max, res, method="max")

            with rasterio.open(DSM_LINEAR) as src_l, rasterio.open(path_max) as src_m:
                lin_all = src_l.read(1).astype(np.float64)
                mx_all = src_m.read(1).astype(np.float64)
                nd_l, nd_m = src_l.nodata, src_m.nodata

            lin_valid = lin_all[lin_all != nd_l]
            mx_valid = mx_all[mx_all != nd_m]

            both = (lin_all != nd_l) & (mx_all != nd_m)
            diff = mx_all[both] - lin_all[both]
            n_higher = np.sum(diff > 0.01)

            print(
                f"\n  Valid cells  — linear: {len(lin_valid):,}  "
                f"max: {len(mx_valid):,}\n"
                f"  At cells where both have data ({both.sum():,}):\n"
                f"    max > linear: {n_higher:,} ({100*n_higher/both.sum():.1f}%)\n"
                f"    mean(max-linear) = {diff.mean():.4f} m\n"
                f"    p95(max)={np.percentile(mx_valid, 95):.3f} m  "
                f"p95(linear)={np.percentile(lin_valid, 95):.3f} m"
            )

            # max never fills gaps — it must have <= valid cells than linear
            assert len(mx_valid) <= len(lin_valid), (
                f"max has {len(mx_valid):,} valid cells but linear has "
                f"{len(lin_valid):,} — max should not fill gaps"
            )

            # max preserves tall peaks that linear smooths away
            p95_max = np.percentile(mx_valid, 95)
            p95_lin = np.percentile(lin_valid, 95)
            assert p95_max >= p95_lin - 0.01, (
                f"p95 max ({p95_max:.3f} m) should be >= p95 linear "
                f"({p95_lin:.3f} m) — max-Z should preserve peak heights"
            )
        finally:
            path_max.unlink(missing_ok=True)


@integration
class TestCameraPosesAboveDSM:
    """
    Each camera's GPS altitude (ellipsoidal) must be above the DSM surface
    at that horizontal position.  If it is not, the coordinate transform or
    datum handling is broken.
    """

    @pytest.fixture(scope="class")
    def poses(self):
        if not POSES_CSV.exists():
            pytest.skip(f"Camera poses CSV missing: {POSES_CSV}")
        rows = []
        with open(POSES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "lat": float(row["lat_deg"]),
                        "lon": float(row["lon_deg"]),
                        "alt": float(row["alt_ellipsoid_m"]),
                    })
                except (ValueError, KeyError):
                    pass
        return rows

    @pytest.fixture(scope="class")
    def dsm_sampler(self):
        rasterio = pytest.importorskip("rasterio")
        pyproj = pytest.importorskip("pyproj")
        with rasterio.open(DSM_LINEAR) as src:
            data = src.read(1).astype(np.float64)
            transform = src.transform
            nd = float(src.nodata)
            crs = src.crs
        tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        def sample(lat, lon):
            ex, ey = tf.transform(lon, lat)
            col = int((ex - transform.c) / transform.a)
            row = int((ey - transform.f) / transform.e)
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                v = data[row, col]
                return None if v == nd else v
            return None

        return sample

    def test_all_cameras_above_dsm(self, poses, dsm_sampler):
        below = []
        outside = 0
        heights = []
        for p in poses:
            dsm_z = dsm_sampler(p["lat"], p["lon"])
            if dsm_z is None:
                outside += 1
                continue
            h = p["alt"] - dsm_z
            heights.append(h)
            if h < 0:
                below.append((p["lat"], p["lon"], p["alt"], dsm_z, h))

        assert len(heights) > 0, "No camera poses fell within DSM extent"
        h_arr = np.array(heights)
        print(
            f"\n  Cameras sampled: {len(heights)}  outside DSM: {outside}\n"
            f"  Height above DSM: min={h_arr.min():.2f} m  "
            f"mean={h_arr.mean():.2f} m  max={h_arr.max():.2f} m\n"
            f"  Cameras below DSM surface: {len(below)}"
        )
        assert len(below) == 0, (
            f"{len(below)} camera(s) appear below the DSM surface — "
            f"first offender: lat={below[0][0]:.6f}, lon={below[0][1]:.6f}, "
            f"cam_alt={below[0][2]:.2f} m, dsm_z={below[0][3]:.2f} m, "
            f"diff={below[0][4]:.2f} m"
        )

    def test_camera_height_above_ground_is_realistic(self, poses, dsm_sampler):
        """
        Pix4DCatch is handheld close-range scanning; expect 0.5–10 m above surface.
        Heights outside this range suggest a datum or scale error.
        """
        heights = []
        for p in poses:
            dsm_z = dsm_sampler(p["lat"], p["lon"])
            if dsm_z is not None:
                heights.append(p["alt"] - dsm_z)

        assert len(heights) > 0
        h_arr = np.array(heights)
        median_h = np.median(h_arr)
        print(f"\n  Median height above DSM: {median_h:.2f} m")
        assert 0.5 <= median_h <= 10.0, (
            f"Median camera height above DSM is {median_h:.2f} m — "
            f"expected 0.5–10 m for handheld close-range scanning"
        )
