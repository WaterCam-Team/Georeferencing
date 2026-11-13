"""Pix4DCatch frames are stored landscape with EXIF orientation 6; the depth map,
intrinsics and pose refer to the stored pixels, so they must be read unrotated."""
import cv2
import numpy as np
from PIL import Image

from aruco_gcp import read_pix4d_frame


def _frame_with_exif_rotation(path, w=64, h=48):
    """A landscape JPEG, left half black / right half white, tagged orientation 6."""
    a = np.zeros((h, w, 3), np.uint8)
    a[:, w // 2:] = 255
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(a).save(path, exif=exif.tobytes(), quality=95)
    return a


def test_frame_is_read_in_stored_orientation(tmp_path):
    p = tmp_path / "Image_000001.jpg"
    stored = _frame_with_exif_rotation(p)
    img = read_pix4d_frame(p)
    assert img.shape == stored.shape                       # 48 x 64, not 64 x 48
    assert img[:, :20].mean() < 30 and img[:, -20:].mean() > 225   # same pixels, not rotated


def test_plain_imread_rotates_these_frames(tmp_path):
    """The trap this guards against: cv2.imread applies the EXIF rotation by default."""
    p = tmp_path / "Image_000001.jpg"
    _frame_with_exif_rotation(p)
    assert cv2.imread(str(p)).shape == (64, 48, 3)


def test_click_viewer_uses_the_same_reader():
    import gcp_from_pix4d
    assert gcp_from_pix4d.read_pix4d_frame is read_pix4d_frame


def test_depth_map_is_read_without_tifffile(tmp_path):
    from aruco_gcp import read_pix4d_depth
    d = np.full((192, 256), 2.5, np.float32)
    p = tmp_path / "DepthMap_000001.tiff"
    assert cv2.imwrite(str(p), d)
    out = read_pix4d_depth(p)
    assert out.shape == (192, 256) and out.dtype == np.float32 and np.allclose(out, 2.5)
    assert read_pix4d_depth(tmp_path / "missing.tiff") is None


def test_unreadable_depth_maps_are_an_error_not_silence(tmp_path, monkeypatch):
    """Every depth read failing used to return no markers without a word."""
    import pytest
    import aruco_gcp
    from tests.test_aruco_gcp import _make_session
    from unittest.mock import MagicMock
    session = _make_session(tmp_path, geoid_sep=None)
    det = MagicMock()
    det.detectMarkers.return_value = (np.array([[[[128., 96.], [130., 96.], [130., 98.], [128., 98.]]]], np.float32), np.array([[7]]), None)
    monkeypatch.setattr(aruco_gcp, "_make_detector", lambda *_: det)
    monkeypatch.setattr(aruco_gcp, "read_pix4d_frame", lambda p: np.zeros((192, 256, 3), np.uint8))
    monkeypatch.setattr(aruco_gcp, "read_pix4d_depth", lambda p: None)
    with pytest.raises(RuntimeError, match="depth map"):
        aruco_gcp.locate_in_pix4d(str(session), every_n_frames=1, min_views=1)
