"""
The package against golden values recorded from the API server's own copies
of this math (app/georef_math.py, app/dem_math.py at the commit named in the
file), on synthetic inputs. The API moves onto the package next; this shows
what changes for it.

One difference is where a guard lives, not what it does: the API checks
the grazing angle and range in georeference_pixel() and leaves its
intersect_flat() unguarded, while the package's intersect_flat() applies
them itself. So rays within MIN_GRAZING_ANGLE_DEG of the horizon or landing
beyond MAX_GROUND_RANGE_M return None here. Everything else must match.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ufonet_geo import geometry as g
from ufonet_geo import pose, terrain

GOLDEN = json.loads((Path(__file__).parent / "golden" / "api_georef_math.json").read_text())
LAT, LON = GOLDEN["lat"], GOLDEN["lon"]
K = np.array(GOLDEN["K"])
D = GOLDEN["D"]


def _ray(px):
    x, y = g.undistort_normalized(px[0], px[1], K, D)
    r = np.array([x, y, 1.0])
    return r / np.linalg.norm(r)


def test_rotation():
    for case in GOLDEN["rotation"]:
        np.testing.assert_allclose(g.build_rotation_matrix(*case["pose"]), case["R"], atol=1e-12)


def test_undistort():
    for case in GOLDEN["undistort"]:
        np.testing.assert_allclose(g.undistort_normalized(*case["px"], K, D), case["xy"], atol=1e-12)


def test_enu_to_latlon():
    for case in GOLDEN["enu"]:
        np.testing.assert_allclose(g.enu_to_latlon(*case["en"], LAT, LON), case["latlon"], atol=1e-12)


def test_flat_ground_matches_except_the_new_guards():
    rejected = 0
    for case in GOLDEN["flat"]:
        ray_world = g.build_rotation_matrix(*case["pose"]).T @ _ray(case["px"])
        got = g.intersect_flat(ray_world, 3.0)
        if case["hit"] is None:
            assert got is None
        elif got is None:
            rejected += 1
            assert (g.depression_angle_deg(ray_world) < g.MIN_GRAZING_ANGLE_DEG
                    or np.hypot(*case["hit"]) > g.MAX_GROUND_RANGE_M)
        else:
            np.testing.assert_allclose(got, case["hit"], atol=1e-9)
    assert rejected > 0          # the golden set does exercise the guard


def test_heading_without_calib_mag():
    for case in GOLDEN["heading"]:
        got = pose.resolve_heading(case["cfg"], None, case["yaw"])
        assert got[1] == case["res"][1]
        assert got[0] == pytest.approx(case["res"][0], abs=1e-12)


def test_pitch_roll():
    for case in GOLDEN["pitch_roll"]:
        got = pose.resolve_pitch_roll(case["cfg"], None, None, *case["pr"])
        assert got[2] == case["res"][2]
        np.testing.assert_allclose(got[:2], case["res"][:2], atol=1e-12)


def test_terrain_ray_march():
    def elev(lon, lat):
        e = (lon - LON) * 84000.0
        n = (lat - LAT) * 111000.0
        return 100.0 + 0.02 * n + 0.3 * np.sin(e / 7.0)

    for case in GOLDEN["terrain"]:
        got = terrain.pixel_to_gps_terrain(_ray(case["px"]), g.build_rotation_matrix(*case["pose"]),
                                           LAT, LON, 103.0, elev, max_range_m=800.0)
        if case["hit"] is None:
            assert got is None
        else:
            np.testing.assert_allclose(got, case["hit"], atol=1e-9)
