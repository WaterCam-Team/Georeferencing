"""Unit tests for terrain ray-casting helpers in georeference_terrain."""

import numpy as np

from georeference_terrain import enu_to_latlon, enu_to_lonlat, ray_intersect_terrain


def test_enu_conversion_origin_roundtrip():
    lat, lon = enu_to_latlon(0.0, 0.0, -76.0, 43.0)
    assert np.isclose(lat, 43.0, atol=1e-10)
    assert np.isclose(lon, -76.0, atol=1e-10)


def test_enu_alias_matches_primary_function():
    lat1, lon1 = enu_to_latlon(5.0, -3.0, -76.0, 43.0)
    lat2, lon2 = enu_to_lonlat(5.0, -3.0, -76.0, 43.0)
    assert np.isclose(lat1, lat2, atol=1e-12)
    assert np.isclose(lon1, lon2, atol=1e-12)


def test_ray_intersect_terrain_hits_constant_surface():
    # Terrain at 100 m elevation everywhere; camera 10 m above at 110 m.
    def get_elevation(lon: float, lat: float):
        return 100.0

    hit = ray_intersect_terrain(
        ray_origin_enu=np.array([0.0, 0.0, 0.0]),
        ray_dir_enu=np.array([0.0, 0.0, -1.0]),
        get_elevation=get_elevation,
        origin_lon=-76.0,
        origin_lat=43.0,
        camera_elev_m=110.0,
        step_m=0.5,
        max_range_m=100.0,
        tol_m=0.05,
    )
    assert hit is not None
    lat, lon, elev, slant = hit
    assert np.isclose(lat, 43.0, atol=1e-8)
    assert np.isclose(lon, -76.0, atol=1e-8)
    assert np.isclose(elev, 100.0, atol=1e-8)
    assert np.isclose(slant, 10.0, atol=0.1)


def test_ray_intersect_terrain_no_hit_for_upward_ray():
    def get_elevation(lon: float, lat: float):
        return 100.0

    hit = ray_intersect_terrain(
        ray_origin_enu=np.array([0.0, 0.0, 0.0]),
        ray_dir_enu=np.array([0.0, 0.0, 1.0]),
        get_elevation=get_elevation,
        origin_lon=-76.0,
        origin_lat=43.0,
        camera_elev_m=110.0,
        step_m=0.5,
        max_range_m=100.0,
    )
    assert hit is None
