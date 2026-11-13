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
