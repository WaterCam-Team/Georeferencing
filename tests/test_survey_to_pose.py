"""scripts/survey_to_pose.py on a synthetic scene: a known pose, targets on flat ground,
their pixels from the same camera model the solver uses. No site data (a made-up
location), photo detection stubbed."""
import csv
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_script():
    spec = importlib.util.spec_from_file_location("survey_to_pose", os.path.join(ROOT, "scripts", "survey_to_pose.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene(heading=72.0, pitch=-18.0, roll=1.5, height=1.4, lat0=0.5, lon0=10.5):
    from pyproj import Proj
    from camera_geometry import build_rotation_matrix
    from geo_core import pixel_ray, intersect_flat

    K = np.array([[900.0, 0, 640.0], [0, 900.0, 480.0], [0, 0, 1.0]])
    R = build_rotation_matrix(heading, pitch, roll)
    proj = Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84")
    pixels = [(200, 700), (640, 650), (1080, 700), (400, 820), (900, 830), (640, 900)]
    targets = {}
    for i, (u, v) in enumerate(pixels):
        off = intersect_flat(R.T @ pixel_ray(K, u, v), height)
        assert off is not None
        lon, lat = proj(off[0], off[1], inverse=True)
        targets[i] = {"uv": (float(u), float(v)), "lat": lat, "lon": lon}
    return K, targets, (heading, pitch, roll, height, lat0, lon0)


@pytest.mark.parametrize("heading_off", [8, 150, -170])
def test_recovers_pose_and_writes_outputs(tmp_path, monkeypatch, heading_off):
    stp = _load_script()
    K, targets, (heading, pitch, roll, height, lat0, lon0) = _scene()

    # Inputs: calibration (no distortion), RTK export with the camera point and targets
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"K": K.tolist(), "D": [[0, 0, 0, 0, 0]], "img_size": [1280, 960]}))
    rtk = tmp_path / "rtk.csv"
    with open(rtk, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Latitude", "Longitude", "Ellipsoidal height"])
        w.writerow(["CAM", lat0, lon0, 101.4])
        for i, t in targets.items():
            w.writerow([f"M{i}", t["lat"], t["lon"], 100.0])
        w.writerow(["M40", lat0, lon0 + 1e-4, 100.0])          # surveyed, not in the photo
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"not used")

    # Detection stubbed: the markers at the synthetic pixels, plus one not surveyed
    found = {i: {"center_uv": t["uv"], "K_undist": K} for i, t in targets.items()}
    found[33] = {"center_uv": (10.0, 10.0), "K_undist": K}
    import aruco_gcp
    import cv2
    monkeypatch.setattr(aruco_gcp, "detect_in_photo", lambda img, K_, D_, d: found)
    monkeypatch.setattr(cv2, "imread", lambda p: np.zeros((960, 1280, 3), np.uint8))

    out = tmp_path / "private" / "survey"
    monkeypatch.setattr(stp, "ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "survey_to_pose.py", "--photo", str(photo), "--calibration", str(calib), "--rtk", str(rtk),
        "--unit-id", "UFOTEST", "--device-id", "watercam-test", "--valid-from", "2026-11-20T10:00:00-05:00",
        "--out-dir", str(out), "--mount-height", str(height),
        "--heading", str(heading + heading_off), "--pitch", str(pitch + 4), "--roll", "0"])
    assert stp.main() == 0

    pc = json.load(open(next(out.glob("pose_config_UFOTEST_*.json"))))
    assert pc["mount_height_m"] == height
    assert abs(pc["heading_deg"] - heading) < 0.5
    assert abs(pc["pitch_deg"] - pitch) < 0.5
    assert abs(pc["roll_deg"] - roll) < 0.5
    assert pc["mask_frame"] == "undistorted"
    uc = json.load(open(next(out.glob("unit_config_UFOTEST_surveyed_*.json"))))
    assert uc["camera_lat"] == lat0 and uc["camera_alt_ellipsoid_m"] == 101.4
    rep = json.load(open(next(out.glob("survey_report_UFOTEST_*.json"))))
    assert rep["rms_m"] < 0.05
    assert rep["flagged"] == [] and len(rep["leave_one_out_m"]) == len(targets)
    assert rep["unmatched"] == ["M33"] and rep["surveyed_not_seen"] == ["M40"]
    with open(next(out.glob("gcps_UFOTEST_*.csv"))) as f:
        assert len(list(csv.DictReader(f))) == len(targets)
    # The API survey import: an open survey with the solved pose as its base pose
    mf = json.load(open(next(out.glob("survey_manifest_UFOTEST_*.json"))))
    assert mf["device_id"] == "watercam-test" and mf["water_maps"] == []
    sv = mf["survey"]
    assert sv["valid_from"] == "2026-11-20T10:00:00-05:00" and sv["valid_to"] is None
    assert sv["base_pose"] == pc                       # ellipsoidal heights: no camera_alt_m
    assert sv["camera_calibration"]["img_size"] == [1280, 960]
    assert "held-out" in sv["accuracy"] and "imu_orientation" not in sv   # the stub photo has no EXIF
    assert (out / mf["reference_photo"]).read_bytes() == b"not used"


def test_refuses_out_dir_outside_private(tmp_path, monkeypatch):
    stp = _load_script()
    monkeypatch.setattr(stp, "ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "survey_to_pose.py", "--photo", "x", "--calibration", "x", "--rtk", "x", "--unit-id", "U", "--device-id", "d",
        "--out-dir", str(tmp_path / "public"), "--mount-height", "1", "--heading", "0"])
    with pytest.raises(SystemExit) as e:
        stp.main()
    assert "outside private" in str(e.value)


def _photo_with_imu(path, words):
    import piexif
    from PIL import Image
    exif = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: b"ASCII\0\0\0" + words.encode()}})
    Image.new("RGB", (64, 48)).save(path, exif=exif)


def test_imu_median_skips_garbled_and_unstarted_readings(tmp_path):
    stp = _load_script()
    paths = []
    for i, words in enumerate(["Roll -4.8 Pitch -33.6 Yaw 227 Gravity 1.68,-4.12,8.73",
                               "Roll -4.9 Pitch -33.7 Yaw 227 Gravity 1.70,-4.10,8.74",
                               "Roll -4.8 Pitch -33.6 Yaw 227 Gravity 1.76,-4.28,-319.04",   # garbled
                               "Roll 0.0 Pitch 0.0 Yaw 0.0"]):                               # not started
        p = tmp_path / f"p{i}.jpg"
        _photo_with_imu(p, words)
        paths.append(str(p))
    imu = stp.imu_median(paths)
    assert imu["gravity"] == pytest.approx([1.70, -4.12, 8.73], abs=0.02)
    assert imu["roll_deg"] == pytest.approx(-4.8) and imu["pitch_deg"] == pytest.approx(-33.6)
    assert stp.imu_median([paths[3]]) is None
