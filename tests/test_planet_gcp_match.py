import csv
from pathlib import Path

import numpy as np

from planet_gcp_match import _select_by_grid, _write_gcp_csv


def test_select_by_grid_spreads_points():
    # Cluster points in the left-top quadrant plus a few in other areas.
    # Build actual minimal objects with attributes used by _select_by_grid.
    class Obj:
        def __init__(self, u, v):
            self.field_u = u
            self.field_v = v
            self.planet_u = 0.0
            self.planet_v = 0.0

    pts_objs = []
    for i in range(50):
        pts_objs.append(Obj(u=10 + (i % 5), v=10 + (i // 5)))
    pts_objs.extend([Obj(u=300, v=10), Obj(u=300, v=300), Obj(u=10, v=300)])

    chosen = _select_by_grid(pts_objs, field_w=320, field_h=320, max_points=6, grid_cols=4, grid_rows=4)
    assert len(chosen) <= 6
    # We expect at least some coverage beyond the dense cluster.
    assert any(p.field_u > 100 for p in chosen)


def test_write_gcp_csv_format(tmp_path: Path):
    out = tmp_path / "gcp.csv"
    gcps = [("planet_1", 10.0, 20.0, 43.0, -76.0), ("planet_2", 11.0, 21.0, 44.0, -75.0)]
    _write_gcp_csv(out, gcps)

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == ["label", "pixel_u", "pixel_v", "lat", "lon", "elev_m"]
    assert len(rows) == 2
    assert rows[0]["label"] == "planet_1"
    assert float(rows[0]["pixel_u"]) == 10.0
    assert float(rows[0]["lat"]) == 43.0
    assert rows[0]["elev_m"] == ""

