"""Tests for vertical_datum module (same-datum pass-through; conversion best-effort)."""
import pytest
from vertical_datum import (
    VERTICAL_ELLIPSOID,
    VERTICAL_EGM96,
    transform_height,
    check_camera_elev_datum,
)


def test_transform_height_same_datum_returns_unchanged():
    h, ok = transform_height(0.0, 50.0, 100.0, VERTICAL_ELLIPSOID, VERTICAL_ELLIPSOID)
    assert ok is True
    assert h == 100.0
    h2, ok2 = transform_height(0.0, 50.0, 200.0, VERTICAL_EGM96, VERTICAL_EGM96)
    assert ok2 is True
    assert h2 == 200.0


def test_check_camera_elev_datum_no_warning_when_dem_plus_height():
    msg, conv = check_camera_elev_datum(
        "dem_plus_height", 100.0, VERTICAL_EGM96, None, height_above_ground_used=True
    )
    assert msg is None
    assert conv is None


def test_check_camera_elev_datum_no_warning_when_same_datum():
    msg, conv = check_camera_elev_datum(
        "user", 100.0, VERTICAL_EGM96, VERTICAL_EGM96, height_above_ground_used=False
    )
    assert msg is None
    assert conv is None


def test_check_camera_elev_datum_warns_when_mixing():
    msg, conv = check_camera_elev_datum(
        "exif", 100.0, VERTICAL_EGM96, None, height_above_ground_used=False
    )
    assert msg is not None
    assert "ellipsoid" in msg or "terrain" in msg.lower() or "EGM" in msg or "vertical" in msg.lower()
