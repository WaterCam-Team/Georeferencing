"""Tests for aruco_gcp.locate_in_pix4d geoid correction."""
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_session(tmp_path: Path, geoid_sep: float | None) -> Path:
    """Build a minimal fake Pix4DCatch session directory."""
    opf = tmp_path / "opf_files"
    opf.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    geolocations = tmp_path / "geolocations"
    geolocations.mkdir()

    # Minimal input_cameras.json — one capture, camera pointing straight down
    cap_lat, cap_lon, cap_alt_ellip = 43.15814, -76.13810, 87.0
    input_cameras = {
        "version": "1.0",
        "format": "application/opf-input-cameras+json",
        "sensors": [{
            "id": 0,
            "image_size_px": [256, 192],
            "internals": {
                "type": "perspective",
                "focal_length_px": 200.0,
                "principal_point_px": [128.0, 96.0],
            }
        }],
        "captures": [{
            "id": 1,
            "rig_model_source": "camera",
            "geolocation": {
                "coordinates": [cap_lat, cap_lon, cap_alt_ellip],
                "crs": {"definition": "GEOGCRS[\"NAD83(2011)\",CS[ellipsoidal,2]]"},
            },
            "orientation": {},
            "reference_camera_id": 10,
            "cameras": [{"id": 10}],
            "time": 0,
        }]
    }
    (opf / "input_cameras.json").write_text(json.dumps(input_cameras))

    # projected_input_cameras.json — camera pointing straight down (omega=phi=kappa=0)
    projected = {
        "captures": [{
            "id": 1,
            "orientation": {"angles_deg": [0.0, 0.0, 0.0]},
        }]
    }
    (opf / "projected_input_cameras.json").write_text(json.dumps(projected))

    # rtkGPS.csv with GeoidSeparation
    if geoid_sep is not None:
        with open(geolocations / "rtkGPS.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Timestamp", "GPSLatitude", "GPSLongitude",
                                               "GPSAltitude", "ReferenceAltitude",
                                               "GeoidSeparation", "HAccuracy", "VAccuracy",
                                               "qualityIndicator"])
            w.writeheader()
            w.writerow({
                "Timestamp": "0", "GPSLatitude": str(cap_lat), "GPSLongitude": str(cap_lon),
                "GPSAltitude": str(cap_alt_ellip),
                "ReferenceAltitude": str(cap_alt_ellip - geoid_sep),
                "GeoidSeparation": str(geoid_sep),
                "HAccuracy": "0.01", "VAccuracy": "0.01", "qualityIndicator": "rtkFixed",
            })

    return tmp_path


def _make_depth_image(depth_m: float, shape=(192, 256)) -> np.ndarray:
    return np.full(shape, depth_m, dtype=np.float32)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestLocateInPix4dGeoidCorrection:
    """locate_in_pix4d must convert ellipsoidal altitude to orthometric via geoid_sep."""

    GEOID_SEP = -34.452   # typical for Syracuse, NY

    def _run_locate(self, session_dir: Path, depth_m: float = 3.0):
        """
        Patch the expensive parts (cv2, tifffile, ArUco detector) and run
        locate_in_pix4d against a single synthetic frame with one marker at
        the image centre.
        """
        import aruco_gcp

        fake_depth = _make_depth_image(depth_m)

        # Fake ArUco detector that always returns one marker at image centre
        fake_corners = np.array([[[[128.0, 96.0], [130.0, 96.0],
                                    [130.0, 98.0], [128.0, 98.0]]]], dtype=np.float32)
        fake_ids = np.array([[7]])

        fake_detector = MagicMock()
        fake_detector.detectMarkers.return_value = (fake_corners, fake_ids, None)

        fake_color = np.zeros((192, 256, 3), dtype=np.uint8)

        with (
            patch("aruco_gcp._make_detector", return_value=fake_detector),
            patch("aruco_gcp.cv2.imread", return_value=fake_color),
            patch("aruco_gcp.tifffile") as mock_tifffile,
        ):
            mock_tifffile.imread.return_value = fake_depth
            # Inject tifffile into the module namespace for the import inside the fn
            import sys
            sys.modules.setdefault("tifffile", mock_tifffile)

            result = aruco_gcp.locate_in_pix4d(
                str(session_dir),
                dict_name="DICT_4X4_50",
                every_n_frames=1,
                min_depth_m=0.1,
                max_depth_m=20.0,
                min_views=1,
            )
        return result

    def test_elev_m_is_orthometric_when_geoid_sep_available(self, tmp_path):
        session = _make_session(tmp_path, geoid_sep=self.GEOID_SEP)
        result = self._run_locate(session, depth_m=3.0)
        assert 7 in result, "Marker 7 should be detected"

        # Camera ellipsoidal altitude = 87.0 m, depth = 3.0 m straight down
        # Marker ellipsoidal altitude ≈ 87.0 − 3.0 = 84.0 m
        # After geoid correction: 84.0 − (−34.452) = 118.452 m orthometric
        elev = result[7]["elev_m"]
        expected_ortho = 87.0 - 3.0 - self.GEOID_SEP   # 118.452
        assert abs(elev - expected_ortho) < 0.1, (
            f"Expected orthometric ~{expected_ortho:.1f} m, got {elev:.3f} m"
        )

    def test_elev_m_is_ellipsoidal_when_no_rtk_csv(self, tmp_path):
        # Session without rtkGPS.csv → falls back to ellipsoidal
        session = _make_session(tmp_path, geoid_sep=None)
        result = self._run_locate(session, depth_m=3.0)
        assert 7 in result

        elev = result[7]["elev_m"]
        expected_ellip = 87.0 - 3.0   # 84.0 m ellipsoidal
        assert abs(elev - expected_ellip) < 0.1, (
            f"Without geoid sep, expected ellipsoidal ~{expected_ellip:.1f} m, got {elev:.3f} m"
        )

    def test_geoid_not_applied_twice(self, tmp_path):
        """Applying the correction once must not differ from applying it twice."""
        session = _make_session(tmp_path, geoid_sep=self.GEOID_SEP)
        result = self._run_locate(session, depth_m=3.0)
        elev = result[7]["elev_m"]
        # If correction were applied twice the value would be ~152.9 m — clearly wrong
        assert elev < 130.0, f"Geoid applied twice? Got {elev:.1f} m"
        assert elev > 110.0, f"Correction not applied? Got {elev:.1f} m"
