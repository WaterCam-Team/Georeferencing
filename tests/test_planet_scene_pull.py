"""Tests for planet_scene_pull filtering and scoring helpers (no network calls)."""
from datetime import datetime, timezone

import pytest

from planet_scene_pull import (
    _month_range_from_photo,
    _parse_month_range,
    filter_scenes,
    score_and_sort_scenes,
)


def _scene(acquired, cloud_cover=0.05, sun_elevation=45.0, sid=None):
    """Build a minimal scene feature dict."""
    return {
        "id": sid or acquired,
        "properties": {
            "acquired": acquired,
            "cloud_cover": cloud_cover,
            "sun_elevation": sun_elevation,
        },
    }


# ---------------------------------------------------------------------------
# _parse_month_range
# ---------------------------------------------------------------------------

def test_parse_month_range_simple():
    assert _parse_month_range("5-9") == {5, 6, 7, 8, 9}


def test_parse_month_range_single():
    assert _parse_month_range("6-6") == {6}


def test_parse_month_range_wraparound():
    assert _parse_month_range("11-2") == {11, 12, 1, 2}


def test_parse_month_range_none():
    assert _parse_month_range(None) is None


# ---------------------------------------------------------------------------
# _month_range_from_photo
# ---------------------------------------------------------------------------

def test_month_range_from_photo_mid_year():
    dt = datetime(2025, 6, 15, tzinfo=timezone.utc)
    assert _month_range_from_photo(dt, window=1) == {5, 6, 7}


def test_month_range_from_photo_wraps_at_start():
    dt = datetime(2025, 1, 10, tzinfo=timezone.utc)
    months = _month_range_from_photo(dt, window=1)
    assert 12 in months
    assert 1 in months
    assert 2 in months


def test_month_range_from_photo_wraps_at_end():
    dt = datetime(2025, 12, 20, tzinfo=timezone.utc)
    months = _month_range_from_photo(dt, window=1)
    assert 11 in months
    assert 12 in months
    assert 1 in months


# ---------------------------------------------------------------------------
# filter_scenes — month range
# ---------------------------------------------------------------------------

def test_filter_by_month_range_drops_outside():
    scenes = [
        _scene("2024-01-15T10:00:00Z"),  # January — outside 5-9
        _scene("2024-06-01T10:00:00Z"),  # June — inside
        _scene("2024-09-30T10:00:00Z"),  # September — inside
        _scene("2024-10-01T10:00:00Z"),  # October — outside
    ]
    result = filter_scenes(scenes, month_range={5, 6, 7, 8, 9})
    ids = [s["id"] for s in result]
    assert "2024-06-01T10:00:00Z" in ids
    assert "2024-09-30T10:00:00Z" in ids
    assert "2024-01-15T10:00:00Z" not in ids
    assert "2024-10-01T10:00:00Z" not in ids


def test_filter_by_month_range_none_passes_all():
    scenes = [_scene("2024-01-15T10:00:00Z"), _scene("2024-07-01T10:00:00Z")]
    assert filter_scenes(scenes, month_range=None) == scenes


def test_filter_by_month_range_boundary_months_kept():
    scenes = [
        _scene("2024-05-01T00:00:00Z"),   # start boundary
        _scene("2024-09-30T23:59:59Z"),   # end boundary
    ]
    result = filter_scenes(scenes, month_range={5, 6, 7, 8, 9})
    assert len(result) == 2


# ---------------------------------------------------------------------------
# filter_scenes — sun elevation
# ---------------------------------------------------------------------------

def test_filter_by_sun_elevation_drops_below_threshold():
    scenes = [
        _scene("2024-06-01T06:00:00Z", sun_elevation=10.0),  # low sun — dropped
        _scene("2024-06-01T12:00:00Z", sun_elevation=55.0),  # high sun — kept
        _scene("2024-06-01T08:00:00Z", sun_elevation=30.0),  # at threshold — kept
    ]
    result = filter_scenes(scenes, sun_elevation_min=30.0)
    assert len(result) == 2
    elevations = [s["properties"]["sun_elevation"] for s in result]
    assert 10.0 not in elevations


def test_filter_by_sun_elevation_none_passes_all():
    scenes = [
        _scene("2024-06-01T06:00:00Z", sun_elevation=5.0),
        _scene("2024-06-01T12:00:00Z", sun_elevation=60.0),
    ]
    assert filter_scenes(scenes, sun_elevation_min=None) == scenes


def test_filter_drops_scene_with_missing_sun_elevation():
    s = _scene("2024-06-01T12:00:00Z")
    s["properties"].pop("sun_elevation")
    result = filter_scenes([s], sun_elevation_min=30.0)
    assert result == []


# ---------------------------------------------------------------------------
# score_and_sort_scenes
# ---------------------------------------------------------------------------

def test_score_sort_cloud_cover_primary():
    scenes = [
        _scene("2024-06-01T12:00:00Z", cloud_cover=0.15),
        _scene("2024-06-02T12:00:00Z", cloud_cover=0.05),
        _scene("2024-06-03T12:00:00Z", cloud_cover=0.10),
    ]
    photo_dt = datetime(2024, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
    result = score_and_sort_scenes(scenes, photo_dt)
    clouds = [s["properties"]["cloud_cover"] for s in result]
    assert clouds == sorted(clouds)


def test_score_sort_date_proximity_secondary():
    # Two scenes with identical cloud cover; the closer date should rank first.
    photo_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    scenes = [
        _scene("2024-06-20T12:00:00Z", cloud_cover=0.05),  # 5 days away
        _scene("2024-06-16T12:00:00Z", cloud_cover=0.05),  # 1 day away
        _scene("2024-06-10T12:00:00Z", cloud_cover=0.05),  # 5 days away
    ]
    result = score_and_sort_scenes(scenes, photo_dt)
    assert result[0]["id"] == "2024-06-16T12:00:00Z"


def test_score_sort_no_photo_dt():
    scenes = [
        _scene("2024-06-01T12:00:00Z", cloud_cover=0.18),
        _scene("2024-06-02T12:00:00Z", cloud_cover=0.02),
    ]
    result = score_and_sort_scenes(scenes, photo_dt=None)
    assert result[0]["properties"]["cloud_cover"] == 0.02


def test_score_sort_missing_cloud_cover_ranked_last():
    s_no_cloud = _scene("2024-06-01T12:00:00Z", cloud_cover=0.05)
    s_no_cloud["properties"].pop("cloud_cover")
    s_good = _scene("2024-06-02T12:00:00Z", cloud_cover=0.15)
    result = score_and_sort_scenes([s_no_cloud, s_good])
    assert result[-1]["id"] == "2024-06-01T12:00:00Z"


def test_score_sort_tz_naive_photo_dt_does_not_raise():
    """Passing a tz-naive photo_dt must not raise TypeError on tz-aware acquired dates."""
    naive_dt = datetime(2024, 6, 15, 12, 0, 0)  # no tzinfo
    scenes = [
        _scene("2024-06-15T12:00:00Z", cloud_cover=0.05),  # tz-aware acquired
        _scene("2024-06-20T12:00:00Z", cloud_cover=0.10),
    ]
    # Must not raise; closer date should rank first
    result = score_and_sort_scenes(scenes, photo_dt=naive_dt)
    assert result[0]["id"] == "2024-06-15T12:00:00Z"


def test_filter_scenes_sun_only_no_month_range():
    """sun_elevation_min filter works correctly when month_range is None."""
    scenes = [
        _scene("2024-06-01T06:00:00Z", sun_elevation=10.0),
        _scene("2024-06-01T12:00:00Z", sun_elevation=55.0),
    ]
    result = filter_scenes(scenes, month_range=None, sun_elevation_min=30.0)
    assert len(result) == 1
    assert result[0]["properties"]["sun_elevation"] == 55.0


def test_filter_scenes_unparseable_acquired_dropped_with_month_range():
    """Scenes with malformed 'acquired' are silently dropped when month filter is active."""
    bad = _scene("not-a-date", cloud_cover=0.01, sun_elevation=60.0)
    good = _scene("2024-06-15T12:00:00Z", cloud_cover=0.10)
    result = filter_scenes([bad, good], month_range={6}, sun_elevation_min=None)
    assert len(result) == 1
    assert result[0] is good
