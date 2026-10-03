"""Tests for the two Pix4DCatch point-cloud layouts read by scripts/pix4d_to_las_dem.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pix4d_to_las_dem import _read_gltf_pointcloud  # noqa: E402

XYZ = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
RGB = np.array([[255, 0, 10], [0, 128, 255]], dtype=np.uint8)


def _write_opf(scan_dir: Path) -> None:
    d = scan_dir / "point_clouds" / "opf_format"
    d.mkdir(parents=True)
    (d / "positions.glbin").write_bytes(XYZ.tobytes())
    rgba = np.hstack([RGB, np.full((2, 1), 255, np.uint8)])
    (d / "colors.glbin").write_bytes(rgba.tobytes())
    (d / "pcl.gltf").write_text(json.dumps({
        "buffers": [{"uri": "positions.glbin", "byteLength": XYZ.nbytes},
                    {"uri": "colors.glbin", "byteLength": rgba.nbytes}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": XYZ.nbytes},
                        {"buffer": 1, "byteOffset": 0, "byteLength": rgba.nbytes}],
        "accessors": [{"bufferView": 0, "count": 2, "type": "VEC3", "componentType": 5126},
                      {"bufferView": 1, "count": 2, "type": "VEC4", "componentType": 5121}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}],
    }))


def test_reads_the_2025_opf_layout(tmp_path):
    _write_opf(tmp_path)
    xyz, rgb = _read_gltf_pointcloud(tmp_path)
    assert xyz.dtype == np.float64
    np.testing.assert_allclose(xyz, XYZ)
    np.testing.assert_array_equal(rgb, RGB)


def test_missing_point_cloud_names_both_layouts(tmp_path):
    (tmp_path / "point_clouds").mkdir()
    with pytest.raises(FileNotFoundError, match="opf_format/pcl.gltf or legacy/pointcloud.gltf"):
        _read_gltf_pointcloud(tmp_path)
