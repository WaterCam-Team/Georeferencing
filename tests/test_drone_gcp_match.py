"""Tests for drone_gcp_match.py — no real images or GeoTIFF needed."""
import numpy as np
import pytest
import cv2

from drone_gcp_match import build_detector, _norm_type


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def test_build_sift_detector():
    det = build_detector("sift", nfeatures=500)
    assert det is not None
    # Can detect keypoints on a synthetic image
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    kp, des = det.detectAndCompute(img, None)
    # SIFT should find at least some keypoints on random noise
    assert kp is not None


def test_build_akaze_detector():
    det = build_detector("akaze", nfeatures=500)
    assert det is not None
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    kp, des = det.detectAndCompute(img, None)
    assert kp is not None


def test_build_orb_detector():
    det = build_detector("orb", nfeatures=500)
    assert det is not None
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    kp, des = det.detectAndCompute(img, None)
    assert kp is not None


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector("superpoint", nfeatures=500)


# ---------------------------------------------------------------------------
# Norm type selection
# ---------------------------------------------------------------------------

def test_sift_uses_l2_norm():
    assert _norm_type("sift") == cv2.NORM_L2


def test_orb_uses_hamming_norm():
    assert _norm_type("orb") == cv2.NORM_HAMMING


def test_akaze_uses_hamming_norm():
    assert _norm_type("akaze") == cv2.NORM_HAMMING


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_detector_name_case_insensitive():
    det_lower = build_detector("sift", 100)
    det_upper = build_detector("SIFT", 100)
    assert type(det_lower) == type(det_upper)


# ---------------------------------------------------------------------------
# Shared utilities still importable (regression guard)
# ---------------------------------------------------------------------------

def test_select_by_grid_importable_from_planet():
    from planet_gcp_match import _select_by_grid
    assert callable(_select_by_grid)


def test_write_gcp_csv_importable_from_planet():
    from planet_gcp_match import _write_gcp_csv
    assert callable(_write_gcp_csv)
