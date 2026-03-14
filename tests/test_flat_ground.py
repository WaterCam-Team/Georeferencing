"""
End-to-end tests: pixel → flat ground → (lat, lon) using geo_core and camera_geometry.

Camera at ENU origin (0, 0, height); ground plane Z = 0.
"""

import numpy as np
from pyproj import Proj

from camera_geometry import build_rotation_matrix
from geo_core import pixel_ray, intersect_flat, pixel_to_world_flat


def test_center_pixel_straight_down_returns_nadir():
    """
    Camera looking straight down (heading=0, pitch=-90, roll=0).
    Image center pixel must map to (lat, lon) = (camera_lat, camera_lon).
    """
    w, h = 1920, 1080
    fx = fy = 1200.0
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    R = build_rotation_matrix(0.0, -90.0, 0.0)
    cam_lat, cam_lon = 0.0, 0.0
    height = 10.0

    result = pixel_to_world_flat(cx, cy, K, R, cam_lat, cam_lon, height)
    assert result is not None
    lat_g, lon_g = result
    assert np.isclose(lat_g, cam_lat, atol=1e-6)
    assert np.isclose(lon_g, cam_lon, atol=1e-6)


def test_off_nadir_ground_point_round_trip():
    """
    Choose a ground point in ENU (east, north). Project it to a pixel via
    the camera model, then pixel_to_world_flat back to lat/lon. The result
    should match the original ground point (in lat/lon).
    """
    w, h = 1920, 1080
    fx = fy = 1200.0
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Camera 10 m high, looking straight down
    R = build_rotation_matrix(0.0, -90.0, 0.0)
    cam_lat, cam_lon = 43.0, -76.0
    height = 10.0

    # Ground point 5 m east, 3 m north of camera in ENU
    east_m, north_m = 5.0, 3.0
    proj = Proj(proj="aeqd", lat_0=cam_lat, lon_0=cam_lon, datum="WGS84")
    lon_expected, lat_expected = proj(east_m, north_m, inverse=True)

    # Ray from camera to ground point (in ENU): (east_m, north_m, -height)
    ray_world = np.array([east_m, north_m, -height], dtype=np.float64)
    ray_world /= np.linalg.norm(ray_world)

    # That ray in camera frame (camera looks down, so +Z is forward)
    ray_cam = R @ ray_world  # world→camera: same as R.T @ ray for direction
    # Actually R is world→camera, so a world direction d_w is d_c = R @ d_w. So ray_cam = R @ ray_world.
    # Project to pixel: p = K @ (ray_cam / ray_cam[2]) so z=1
    if ray_cam[2] <= 0:
        raise AssertionError("Ray should be in front of camera")
    p = K @ (ray_cam / ray_cam[2])
    u, v = p[0] / p[2], p[1] / p[2]

    # Back to lat/lon via geo_core
    result = pixel_to_world_flat(u, v, K, R, cam_lat, cam_lon, height)
    assert result is not None
    lat_g, lon_g = result

    # Should match to within small tolerance (sub-meter at this scale)
    assert np.isclose(lat_g, lat_expected, atol=1e-5)
    assert np.isclose(lon_g, lon_expected, atol=1e-5)


def test_intersect_flat_returns_none_for_upward_ray():
    """Ray pointing up should not intersect ground."""
    ray_up = np.array([0.0, 0.0, 1.0])
    assert intersect_flat(ray_up, 10.0) is None


def test_intersect_flat_returns_none_for_horizontal_ray():
    """Horizontal ray should not intersect ground."""
    ray_h = np.array([1.0, 0.0, 0.0])
    assert intersect_flat(ray_h, 10.0) is None


def test_intersect_flat_downward_gives_correct_enu():
    """Ray straight down should hit (0, 0) in ENU."""
    ray_down = np.array([0.0, 0.0, -1.0])
    enu = intersect_flat(ray_down, 10.0)
    assert enu is not None
    assert np.isclose(enu[0], 0.0, atol=1e-9)
    assert np.isclose(enu[1], 0.0, atol=1e-9)
