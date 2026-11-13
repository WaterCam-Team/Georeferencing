"""pix4d_to_las_dem --source depth: points from the frames' depth maps and OPF poses."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pix4d_to_las_dem import _depth_pointcloud  # noqa: E402


def _scan(tmp_path, depth_m=2.0, pos=(10.0, 20.0, 3.0), conf=2):
    opf = tmp_path / "opf_files"; opf.mkdir()
    img_dir = tmp_path / "images"; img_dir.mkdir()
    sensor = {"id": 0, "image_size_px": [64, 48],
              "internals": {"type": "perspective", "focal_length_px": 50.0, "principal_point_px": [32.0, 24.0]}}
    cams = [{"id": 10, "uri": "../images/Image_000001.jpg"},
            {"id": 11, "uri": "../images/DepthMap_000001.tiff"},
            {"id": 12, "uri": "../images/Confidence_000001.tiff"}]
    (opf / "camera_list.json").write_text(json.dumps({"cameras": cams}))
    (opf / "input_cameras.json").write_text(json.dumps({"sensors": [sensor], "captures": [{
        "id": 1, "cameras": [{"id": 10, "extensions": {"PIX4D_input_depth_map": {"id": 11}}}]}]}))
    # omega = phi = kappa = 0: camera axes = world axes; the camera looks along -z (down)
    (opf / "projected_input_cameras.json").write_text(json.dumps({"captures": [{
        "id": 1, "orientation": {"angles_deg": [0.0, 0.0, 0.0]}, "geolocation": {"position": list(pos)}}]}))
    img = np.zeros((48, 64, 3), np.uint8); img[:, :, 2] = 200          # red frame (BGR)
    cv2.imwrite(str(img_dir / "Image_000001.jpg"), img)
    cv2.imwrite(str(img_dir / "DepthMap_000001.tiff"), np.full((24, 32), depth_m, np.float32))
    cv2.imwrite(str(img_dir / "Confidence_000001.tiff"), np.full((24, 32), conf, np.uint8))
    return tmp_path


def test_points_lie_depth_below_a_downward_camera(tmp_path):
    xyz, rgb = _depth_pointcloud(_scan(tmp_path), pixel_step=1)
    assert len(xyz) == 24 * 32
    np.testing.assert_allclose(xyz[:, 2], 3.0 - 2.0)
    # the image's top-left corner lies west (-x) and north (+y) of the camera
    top_left = xyz[0]
    assert top_left[0] < 10.0 and top_left[1] > 20.0
    np.testing.assert_allclose(top_left[:2], [10.0 + (0 - 32) / 50 * 2, 20.0 - (0 - 24) / 50 * 2])
    assert rgb[:, 0].min() > 150 and rgb[:, 2].max() < 60               # colours are RGB


def test_low_confidence_depth_is_dropped(tmp_path):
    xyz, _ = _depth_pointcloud(_scan(tmp_path, conf=1), pixel_step=1)
    assert len(xyz) == 0


def test_no_depth_maps_is_an_error(tmp_path):
    import pytest
    scan = _scan(tmp_path)
    (scan / "images" / "DepthMap_000001.tiff").unlink()
    with pytest.raises(RuntimeError, match="depth maps"):
        _depth_pointcloud(scan)
