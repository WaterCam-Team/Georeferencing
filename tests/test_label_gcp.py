"""Tests for label_gcp.py: control-point parsing, label cleanup, sorting, key assignment."""
import json
import tempfile
from pathlib import Path

from label_gcp import (
    ControlPoint,
    clean_control_point_label,
    load_control_points_pix4d_json,
    load_control_points_csv,
    load_control_points,
    natural_sort_key,
    _assignable_keys,
)


def _write(tmp_path_factory_dir: Path, name: str, content: str) -> Path:
    p = tmp_path_factory_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def test_clean_control_point_label_strips_project_prefix():
    assert clean_control_point_label("site 1_global_5") == "global_5"
    assert clean_control_point_label("global_1") == "global_1"
    assert clean_control_point_label("some project_marker_12") == "marker_12"


def test_clean_control_point_label_falls_back_to_full_id():
    assert clean_control_point_label("no-trailing-number-here") == "no-trailing-number-here"


def _make_pix4d_json(ids_coords):
    gcps = []
    for gid, (lat, lon, elev) in ids_coords.items():
        gcps.append({
            "id": gid,
            "geolocation": {"coordinates": [lat, lon, elev]},
        })
    return {"format": "application/opf-input-control-points+json", "gcps": gcps}


def test_load_control_points_pix4d_json(tmp_path):
    data = _make_pix4d_json({
        "site 1_global_1": (40.7128, -74.0060, 10.0),
        "site 1_global_2": (40.7129, -74.0061, 9.9),
    })
    path = tmp_path / "input-control-points.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    points = load_control_points_pix4d_json(path)
    labels = {p.label for p in points}
    assert labels == {"global_1", "global_2"}
    g1 = next(p for p in points if p.label == "global_1")
    assert g1.lat == 40.7128
    assert g1.lon == -74.0060
    assert g1.elev_m == 10.0


def test_load_control_points_pix4d_json_skips_incomplete_entries(tmp_path):
    data = {
        "gcps": [
            {"id": "proj_global_1", "geolocation": {"coordinates": [40.7, -74.0, 10.0]}},
            {"id": "proj_global_2", "geolocation": {}},  # no coordinates -> skipped
        ]
    }
    path = tmp_path / "input-control-points.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    points = load_control_points_pix4d_json(path)
    assert len(points) == 1
    assert points[0].label == "global_1"


def test_load_control_points_csv(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "label,lat,lon,elev_m\nglobal_1,40.71,-74.01,10.0\nglobal_2,40.72,-74.02,\n",
        encoding="utf-8",
    )
    points = load_control_points_csv(path)
    assert len(points) == 2
    p1 = next(p for p in points if p.label == "global_1")
    assert p1.elev_m == 10.0
    p2 = next(p for p in points if p.label == "global_2")
    assert p2.elev_m is None


def test_load_control_points_dispatches_on_extension(tmp_path):
    json_path = tmp_path / "input-control-points.json"
    json_path.write_text(json.dumps(_make_pix4d_json({"g_1": (1.0, 2.0, 3.0)})), encoding="utf-8")
    csv_path = tmp_path / "truth.csv"
    csv_path.write_text("label,lat,lon,elev_m\ng_1,1.0,2.0,3.0\n", encoding="utf-8")

    assert len(load_control_points(json_path)) == 1
    assert len(load_control_points(csv_path)) == 1


def test_natural_sort_key_orders_numerically_not_lexically():
    labels = ["global_10", "global_2", "global_1"]
    assert sorted(labels, key=natural_sort_key) == ["global_1", "global_2", "global_10"]


def test_assignable_keys_uses_digits_first_then_letters():
    keys = _assignable_keys(3)
    assert keys == [ord("1"), ord("2"), ord("3")]

    keys12 = _assignable_keys(12)
    assert keys12[:9] == [ord(str(d)) for d in range(1, 10)]
    assert len(keys12) == 12
    # reserved control keys must never be assignable
    reserved = {ord(c) for c in "suczxSUCZX"}
    assert not (set(keys12) & reserved)
