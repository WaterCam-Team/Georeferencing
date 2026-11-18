"""scan_to_pose: grid -> true north, rotation -> angles, and the PnP pose on synthetic matches."""
import math
import sys
from pathlib import Path

import numpy as np
from pyproj import CRS, Geod, Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scan_to_pose import angles, grid_to_enu, pnp  # noqa: E402
from ufonet_geo.geometry import build_rotation_matrix, project_to_pixels  # noqa: E402

UTM18 = CRS.from_epsg(6347)   # NAD83(2011) / UTM 18N, as Pix4DCatch writes in the northeast US
LAT, LON = 43.0, -76.1        # ~1.1 deg west of the zone's central meridian (-75)


def _utm(lat, lon):
    return Transformer.from_crs("EPSG:4326", UTM18, always_xy=True).transform(lon, lat)


def test_grid_north_true_azimuth_matches_geodesic():
    e, n = _utm(LAT, LON)
    M = grid_to_enu(UTM18, e, n, LAT, LON)
    east, north, _ = M @ [0.0, 1.0, 0.0]
    az = math.degrees(math.atan2(east, north))
    to_geo = Transformer.from_crs(UTM18, "EPSG:4326", always_xy=True)
    lo1, la1 = to_geo.transform(e, n)
    lo2, la2 = to_geo.transform(e, n + 100.0)
    want = Geod(ellps="GRS80").inv(lo1, la1, lo2, la2)[0]
    assert want < -0.5                      # west of the central meridian grid north is west of true north
    assert abs(az - want) < 0.01


def test_angles_round_trip_through_grid():
    e, n = _utm(LAT, LON)
    G2E = grid_to_enu(UTM18, e, n, LAT, LON)
    for h, p, r in ((226.2, -26.8, -8.9), (10.0, -5.0, 3.0), (300.0, -45.0, 0.0)):
        Rw2c = build_rotation_matrix(h, p, r) @ G2E        # scene (grid) -> camera
        h2, p2, r2, resid = angles(Rw2c, G2E)
        assert abs(((h2 - h + 180) % 360) - 180) < 0.02
        assert abs(p2 - p) < 0.02 and abs(r2 - r) < 0.02 and resid < 1e-3


def test_pnp_recovers_pose_from_synthetic_matches():
    K = np.array([[2900.0, 0, 1296], [0, 2900.0, 972], [0, 0, 1]])
    D = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(1)
    h, p, r = 226.0, -27.0, -9.0
    R = build_rotation_matrix(h, p, r)                       # ENU -> camera
    C = np.array([5.0, -3.0, 2.0])
    cam = np.c_[rng.uniform(-1.5, 1.5, 300), rng.uniform(-1.0, 1.0, 300), rng.uniform(3, 6, 300)]
    world = C + cam @ R                                      # camera -> ENU is R.T
    u, v = project_to_pixels(cam, K, D)
    Q = np.c_[u, v] + rng.normal(0, 0.5, (300, 2))
    Rw2c, C_est, inl, _ = pnp(world, Q, K, D)
    h2, p2, r2, _ = angles(Rw2c, np.eye(3))
    assert np.linalg.norm(C_est - C) < 0.02
    assert abs(h2 - h) < 0.1 and abs(p2 - p) < 0.1 and abs(r2 - r) < 0.1
    assert len(inl) > 280
