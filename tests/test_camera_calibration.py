"""calibrate_camera on synthetic chessboard views with a known camera matrix."""
import cv2
import numpy as np
import pytest

from camera_calibration import calibrate_camera

BW, BH, SQ = 9, 6, 0.03          # interior corners, square size (m)
W, H = 800, 600
K_TRUE = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])


def _board_texture(px=40, dark=0, light=255):
    """A (BW+1) x (BH+1) square chessboard with a one-square white margin."""
    nx, ny = BW + 1, BH + 1
    img = np.full(((ny + 2) * px, (nx + 2) * px), light, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                img[(j + 1) * px:(j + 2) * px, (i + 1) * px:(i + 2) * px] = dark
    return img, px


def _render_views(out_dir, dark, light, blur=0, noise=0.0, seed=0):
    tex, px = _board_texture(dark=dark, light=light)
    rng = np.random.default_rng(seed)
    # Board corners in metres (texture includes the margin square).
    tw, th = tex.shape[1] / px * SQ, tex.shape[0] / px * SQ
    obj = np.array([[0, 0, 0], [tw, 0, 0], [tw, th, 0], [0, th, 0]], np.float64)
    src = np.array([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]],
                    [0, tex.shape[0]]], np.float32)
    tilts = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15), (-15, 15), (15, -15)]
    for n, (ax, ay) in enumerate(tilts):
        rvec = np.deg2rad([ax, ay, 5 * n - 15]).reshape(3, 1)
        tvec = np.array([[-tw / 2], [-th / 2], [0.75]])
        dst, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, None)
        Hm = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                  borderValue=int(light))
        img = img.astype(np.float64)
        if blur:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        if noise:
            img += rng.normal(0, noise, img.shape)
        cv2.imwrite(str(out_dir / f"view_{n:02d}.png"),
                    np.clip(img, 0, 255).astype(np.uint8))


def _calibrate(tmp_path):
    return calibrate_camera(str(tmp_path), save_path=str(tmp_path / "calib.json"),
                            show_corners=False, board_w=BW, board_h=BH,
                            square_size_m=SQ)


def test_recovers_focal_length_on_clean_views(tmp_path):
    _render_views(tmp_path, dark=0, light=255)
    intr = _calibrate(tmp_path)
    K = np.array(intr.K)
    assert intr.rms < 0.5
    assert K[0, 0] == pytest.approx(700, rel=0.02)
    assert K[1, 1] == pytest.approx(700, rel=0.02)
    assert K[0, 2] == pytest.approx(W / 2, abs=10)
    assert K[1, 2] == pytest.approx(H / 2, abs=10)


def test_uses_sector_based_detector(tmp_path, monkeypatch):
    # The SB detector found boards on real low-contrast NIR frames where the
    # classic one did not (see calibrate_camera). Without the classic detector,
    # calibration must still work, and the SB corners need no cornerSubPix pass.
    def no_classic(*a, **k):
        raise AssertionError("classic detector used")
    monkeypatch.setattr(cv2, "findChessboardCorners", no_classic)
    monkeypatch.setattr(cv2, "cornerSubPix", no_classic)
    _render_views(tmp_path, dark=110, light=150, blur=1.2, noise=3.0)
    intr = _calibrate(tmp_path)
    assert intr.rms < 1.0
    assert np.array(intr.K)[0, 0] == pytest.approx(700, rel=0.05)


def test_falls_back_to_classic_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "findChessboardCornersSB",
                        lambda *a, **k: (False, None))
    _render_views(tmp_path, dark=0, light=255)
    intr = _calibrate(tmp_path)
    assert np.array(intr.K)[0, 0] == pytest.approx(700, rel=0.02)
