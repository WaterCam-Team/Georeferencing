"""
Compare auto-corrected EXIF pose vs. GCP-refined "true" pose, across one or
more IMU mount configurations, using RTK-surveyed ArUco marker ground truth.

Motivation: for UFO-006 (imu_mount_offset_deg=180), unit_config.resolve_pitch_roll()
resolves an upward-pointing pose for a real flood photo (confirmed independently
via validate_georef.py finding zero terrain intersections). This script gives a
controlled way to check whether the 180-degree negation formula actually recovers
a pose close to ground truth, using a backyard test rig with RTK-surveyed markers
instead of relying on indirect evidence from one field photo.

For each session (e.g. one mount orientation), it:
  1. Detects ArUco markers in the photo (aruco_gcp.detect_in_photo).
  2. Merges detections with RTK ground truth into a GCP CSV (aruco_gcp.write_gcp_csv).
  3. Computes the "auto" pose: unit_config.resolve_heading/resolve_pitch_roll
     applied to this session's EXIF + imu_mount_offset_deg.
  4. Computes the "refined" pose: gcp.refine_pose_from_gcps() fit against the
     RTK-surveyed GCPs (ground truth, independent of the IMU correction formula).
  5. Reports the delta between the two, plus per-GCP residuals of the refined
     fit (a sanity check that the RTK data + refinement itself are trustworthy).

Usage:
    python scripts/compare_mount_configs.py --manifest backyard_test.json

Manifest schema (JSON):
{
  "unit_config": "unit_config_TEST.json",     // shared base: mount_height_m, calibration, etc.
  "rtk_gcps_csv": "backyard_rtk_gcps.csv",    // marker_id,lat,lon,elev_m[,std_m]
  "aruco_dict": "DICT_4X4_50",
  "sessions": [
    {"name": "nominal_0deg",   "image": "backyard/nominal/photo.jpg",   "imu_mount_offset_deg": 0},
    {"name": "rotated_180deg", "image": "backyard/rotated/photo.jpg",   "imu_mount_offset_deg": 180}
  ]
}

RTK ground-truth CSV format:
    marker_id,lat,lon,elev_m,std_m
    1,43.xxxxxxxx,-76.xxxxxxxx,84.90,0.02
    ...
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_config as _uc
from georeference_tool import load_calibrated_intrinsics, scale_intrinsics_for_resolution, read_gps_from_exif
from exif_imu import read_gps_imu_from_exif
import aruco_gcp
from gcp import load_gcps, refine_pose_from_gcps, gcp_residuals


def load_rtk_truth(path):
    truth = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mid = int(row["marker_id"])
            truth[mid] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "elev_m": float(row["elev_m"]),
                "n_views": 1,
                "std_m": float(row["std_m"]) if row.get("std_m") else 0.0,
            }
    return truth


def angle_delta(a, b):
    """Smallest signed difference a-b, wrapped to [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def run_session(session, base_cfg_data, rtk_truth, aruco_dict, out_dir):
    name = session["name"]
    image_path = session["image"]
    offset_deg = float(session["imu_mount_offset_deg"])

    cfg_data = dict(base_cfg_data)
    cfg_data["imu_mount_offset_deg"] = offset_deg
    ucfg = _uc.UnitConfig(cfg_data)

    print("\n" + "=" * 66)
    print(f"SESSION: {name}  (imu_mount_offset_deg={offset_deg})")
    print("=" * 66)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None
    img_h, img_w = img.shape[:2]

    calib_dir = os.path.dirname(os.path.abspath(image_path))
    calib_path = ucfg.resolve_calibration(None, calib_dir)
    if not os.path.exists(calib_path):
        calib_path = ucfg.resolve_calibration(None, ".")
    if not os.path.exists(calib_path):
        print(f"[ERROR] Calibration file not found (tried {calib_dir} and '.').")
        return None
    K, D, calib_img_size, _ = load_calibrated_intrinsics(calib_path)
    if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (img_w, img_h):
        K = scale_intrinsics_for_resolution(K, calib_img_size[0], calib_img_size[1], img_w, img_h)

    # ── Marker detection -> GCP CSV (reuses aruco_gcp's existing merge/writer) ──
    detections = aruco_gcp.detect_in_photo(img, K, D, dict_name=aruco_dict)
    if not detections:
        print("[ERROR] No ArUco markers detected in this photo.")
        return None
    os.makedirs(out_dir, exist_ok=True)
    gcp_csv_path = os.path.join(out_dir, f"gcps_{name}.csv")
    rows = aruco_gcp.write_gcp_csv(
        detections, rtk_truth, gcp_csv_path, image_basename=os.path.basename(image_path)
    )
    if len(rows) < 3:
        print(f"[ERROR] Only {len(rows)} marker(s) matched RTK truth — need >=3 for pose refinement.")
        return None
    gcps = load_gcps(gcp_csv_path)
    K_undist = next(iter(detections.values()))["K_undist"]

    # ── Auto-corrected EXIF pose (the thing under test) ──
    gps = read_gps_from_exif(image_path)
    cam_lat, cam_lon = gps["lat"], gps["lon"]
    imu = read_gps_imu_from_exif(image_path)
    exif_yaw = gps.get("heading") if gps.get("heading") is not None else imu.get("yaw_deg")
    exif_pitch, exif_roll = imu.get("pitch_deg"), imu.get("roll_deg")

    heading0, h_src = ucfg.resolve_heading(None, exif_yaw)
    pitch0, roll0, pr_src = ucfg.resolve_pitch_roll(None, None, exif_pitch, exif_roll)
    height0, mh_src = ucfg.resolve_mount_height(None)
    if height0 is None:
        print("[ERROR] mount_height_m not set in unit config — required for flat-ground refinement.")
        return None

    print(f"[AUTO]    heading={heading0:.2f}° ({h_src})  pitch={pitch0:.2f}° ({pr_src})  roll={roll0:.2f}° ({pr_src})")

    # ── GCP-refined "true" pose (ground truth, independent of the IMU formula) ──
    (cam_lat_r, cam_lon_r, height_r, R_r, rms_deg,
     heading_r, pitch_r, roll_r) = refine_pose_from_gcps(
        K_undist, gcps, cam_lat, cam_lon, height0, heading0, pitch0, roll0,
    )
    print(f"[REFINED] heading={heading_r:.2f}°  pitch={pitch_r:.2f}°  roll={roll_r:.2f}°  "
          f"(fit rms={rms_deg:.4f}° over {len(gcps)} GCPs)")

    residuals = gcp_residuals(K_undist, R_r, cam_lat_r, cam_lon_r, height_r, gcps)
    print("\n  Per-marker residual (refined pose vs RTK truth):")
    for g, (lat_e, lon_e, dist_m) in zip(gcps, residuals):
        print(f"    {g.label:>10}: {dist_m:.3f} m")
    dists = [d for (_, _, d) in residuals if not np.isnan(d)]
    if dists:
        median = float(np.median(dists))
        outliers = [d for d in dists if median > 0 and d > 3 * median]
        print(f"  median={median:.3f} m  max={max(dists):.3f} m  outliers(>3x median)={len(outliers)}")

    d_heading = angle_delta(heading_r, heading0)
    d_pitch = pitch_r - pitch0
    d_roll = roll_r - roll0
    print(f"\n  AUTO vs REFINED delta:  heading={d_heading:+.2f}°  pitch={d_pitch:+.2f}°  roll={d_roll:+.2f}°")

    return {
        "session": name, "imu_mount_offset_deg": offset_deg,
        "auto": {"heading": heading0, "pitch": pitch0, "roll": roll0},
        "refined": {"heading": heading_r, "pitch": pitch_r, "roll": roll_r, "rms_deg": rms_deg},
        "delta": {"heading": d_heading, "pitch": d_pitch, "roll": d_roll},
        "gcp_residuals_m": dists,
        "n_gcps": len(gcps),
    }


def main():
    p = argparse.ArgumentParser(
        description="Compare auto-corrected EXIF pose vs GCP-refined pose across mount configs"
    )
    p.add_argument("--manifest", required=True, help="JSON manifest (see module docstring)")
    p.add_argument("--out-dir", default="output/mount_config_comparison")
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    base_cfg_data = {}
    if manifest.get("unit_config"):
        with open(manifest["unit_config"]) as f:
            base_cfg_data = json.load(f)

    rtk_truth = load_rtk_truth(manifest["rtk_gcps_csv"])
    aruco_dict = manifest.get("aruco_dict", "DICT_4X4_50")

    results = []
    for session in manifest["sessions"]:
        r = run_session(session, base_cfg_data, rtk_truth, aruco_dict, args.out_dir)
        if r is not None:
            results.append(r)

    if len(results) >= 2:
        print("\n" + "=" * 66)
        print("CROSS-SESSION SUMMARY")
        print("=" * 66)
        for r in results:
            print(f"  {r['session']:>16} (offset={r['imu_mount_offset_deg']:>5.0f}°): "
                  f"Δheading={r['delta']['heading']:+6.2f}°  Δpitch={r['delta']['pitch']:+6.2f}°  "
                  f"Δroll={r['delta']['roll']:+6.2f}°  fit_rms={r['refined']['rms_deg']:.4f}°")

    if not results:
        print("\n[ERROR] No session produced usable results.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "comparison_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {args.out_dir}/comparison_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
