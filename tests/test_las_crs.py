"""LAS terrain loading: an unknown CRS is an error unless given explicitly."""
import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from georeference_terrain import load_las_terrain


def _write_las(path, with_crs: bool):
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.array([583000.0, 4507000.0, 0.0])
    header.scales = np.array([0.01, 0.01, 0.01])
    if with_crs:
        from pyproj import CRS
        header.add_crs(CRS.from_epsg(32618))
    las = laspy.LasData(header)
    las.x = np.array([583000.0, 583010.0, 583020.0])
    las.y = np.array([4507000.0, 4507010.0, 4507020.0])
    las.z = np.array([1.0, 2.0, 3.0])
    las.write(str(path))


def test_missing_crs_raises(tmp_path):
    p = tmp_path / "nocrs.las"
    _write_las(p, with_crs=False)
    with pytest.raises(ValueError, match="--las-crs"):
        load_las_terrain(str(p))


def test_explicit_crs_override_loads(tmp_path):
    p = tmp_path / "nocrs.las"
    _write_las(p, with_crs=False)
    assert load_las_terrain(str(p), las_crs_epsg=32618) is not None


def test_embedded_crs_loads(tmp_path):
    p = tmp_path / "crs.las"
    _write_las(p, with_crs=True)
    assert load_las_terrain(str(p)) is not None
