"""
Level 4 DSM source comparison (docs/ACCURACY_IMPROVEMENT_PLAN.md)
==================================================================
Georeferences the same fixed-camera photo twice — once against a
Pix4DCatch photogrammetric DSM, once against a national DEM (e.g. USGS
1 m) — and reports the horizontal displacement between the two results
across a grid of image points. This quantifies how much the choice of
terrain model contributes to total georeferencing error, independent of
the IMU/GCP work tracked as Levels 1-3.

Reuses the same pose-resolution and ray/terrain intersection code as
validate_georef.py and georeference_terrain.py rather than reimplementing
it, so results are directly comparable to the rest of the accuracy plan.

Usage:
    python scripts/flood_export.py \
        --image Meadowbrook-006/20260426-090402-NIR-OFF.jpg \
        --dsm-a output/pix4d/2026-04-24-13-11-52_dem.tif \
        --dsm-b USGS_1M_18_x41y477_NY_FEMAR2_Central_2018_D19.tif \
        --unit-config unit_config_UFO006.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unit_config as _uc
from camera_geometry import build_rotation_matrix
from georeference_tool import load_calibrated_intrinsics, scale_intrinsics_for_resolution
from georeference_terrain import pixel_to_gps_terrain
from geo_core import camera_elev_from_dem
from validate_georef import load_dem, haversine_m, read_exif_pose


def build_grid(img_w, img_h, spacing_px):
    us = np.linspace(spacing_px / 2, img_w - spacing_px / 2, max(2, round(img_w / spacing_px)))
    vs = np.linspace(spacing_px / 2, img_h - spacing_px / 2, max(2, round(img_h / spacing_px)))
    return us, vs


def bucket_label(slant_m):
    if slant_m < 10.0:
        return "<10m"
    if slant_m < 20.0:
        return "10-20m"
    return ">20m"


def summarize(displacements):
    arr = np.asarray(displacements, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.percentile(arr, 90)),
        "max_m": float(np.max(arr)),
    }


def to_geojson(points, extra_keys):
    features = []
    for p in points:
        props = {k: p[k] for k in extra_keys if k in p}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"], p.get("elev_m", 0.0)]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    p = argparse.ArgumentParser(description="Level 4 DSM source comparison")
    p.add_argument("--image", required=True, help="Path to photo")
    p.add_argument("--dsm-a", required=True, help="Pix4DCatch photogrammetric DSM GeoTIFF")
    p.add_argument("--dsm-b", required=True, help="Comparison DEM GeoTIFF (e.g. USGS 1m)")
    p.add_argument("--spacing", type=float, default=120.0, help="Grid spacing in pixels (default 120)")
    p.add_argument("--max-range", type=float, default=500.0, help="Max ray range in meters (default 500)")
    p.add_argument("--calibration", "-c", default=None)
    p.add_argument("--heading", type=float, default=None)
    p.add_argument("--height-agl", type=float, default=None)
    p.add_argument("--pitch", type=float, default=None)
    p.add_argument("--roll", type=float, default=None)
    p.add_argument("--out-dir", default="output/dsm_comparison")
    _uc.add_argument(p)
    args = p.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return 1
    if not os.path.exists(args.dsm_a):
        print(f"DSM-A not found: {args.dsm_a}")
        return 1
    if not os.path.exists(args.dsm_b):
        print(f"DSM-B not found: {args.dsm_b}")
        return 1

    ucfg = _uc.from_args(args)
    ucfg_dir = os.path.dirname(os.path.abspath(args.unit_config)) if args.unit_config else "."

    print("=" * 66)
    print("LEVEL 4 — DSM SOURCE COMPARISON")
    print("=" * 66)
    if ucfg.unit_id:
        print(f"Unit: {ucfg.unit_id}")

    # ── Pose (same resolution chain as validate_georef.py / georeference_terrain.py) ──
    pose = read_exif_pose(args.image)
    cam_lat, cam_lon = pose["lat"], pose["lon"]
    exif_alt = pose["altitude_m"]
    heading, h_src = ucfg.resolve_heading(args.heading, pose.get("heading"), pose.get("heading"))
    pitch, roll, pr_src = ucfg.resolve_pitch_roll(args.pitch, args.roll, pose.get("pitch"), pose.get("roll"))
    print(f"\n[POSE] lat={cam_lat:.6f} lon={cam_lon:.6f} alt={exif_alt:.2f}m")
    print(f"[POSE] heading={heading:.2f}° ({h_src})  pitch={pitch:.2f}° ({pr_src})  roll={roll:.2f}° ({pr_src})")

    height_agl, mh_src = ucfg.resolve_mount_height(args.height_agl)
    if height_agl is None:
        print("[ERROR] Mount height unknown — set mount_height_m in unit config or pass --height-agl.")
        return 1
    print(f"[POSE] mount height={height_agl:.4f} m ({mh_src})")

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        return 1
    img_h, img_w = img.shape[:2]
    print(f"[IMAGE] {args.image} — {img_w}x{img_h}")

    calib_path = ucfg.resolve_calibration(args.calibration, ucfg_dir)
    K, D, calib_img_size, _ = load_calibrated_intrinsics(calib_path)
    if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (img_w, img_h):
        K = scale_intrinsics_for_resolution(K, calib_img_size[0], calib_img_size[1], img_w, img_h)
    R = build_rotation_matrix(heading, pitch, roll)
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), alpha=0)
    x0, y0, _, _ = roi

    # ── Terrain sources ──
    print(f"\n[DSM-A] {args.dsm_a}")
    get_elev_a, bounds_a = load_dem(args.dsm_a)
    print(f"[DSM-B] {args.dsm_b}")
    get_elev_b, bounds_b = load_dem(args.dsm_b)

    terrain_at_cam_a = get_elev_a(cam_lon, cam_lat, warn=True)
    terrain_at_cam_b = get_elev_b(cam_lon, cam_lat, warn=True)
    if terrain_at_cam_a is None:
        print("[ERROR] DSM-A has no data at camera position.")
        return 1
    if terrain_at_cam_b is None:
        print("[ERROR] DSM-B has no data at camera position.")
        return 1
    cam_elev_a = camera_elev_from_dem(get_elev_a, cam_lat, cam_lon, height_agl)
    cam_elev_b = camera_elev_from_dem(get_elev_b, cam_lat, cam_lon, height_agl)
    print(f"[DSM-A] camera elev (terrain datum): {cam_elev_a:.2f} m")
    print(f"[DSM-B] camera elev (terrain datum): {cam_elev_b:.2f} m")

    # ── Grid comparison ──
    us, vs = build_grid(img_w, img_h, args.spacing)
    print(f"\n[GRID] {len(us)}x{len(vs)} points, spacing~{args.spacing:.0f}px")

    points_a, points_b, compared = [], [], []
    n_attempted = 0
    for v_r in vs:
        for u_r in us:
            n_attempted += 1
            u_ud, v_ud = u_r - x0, v_r - y0
            r_a = pixel_to_gps_terrain(
                (u_ud, v_ud), K_new, R, cam_lat, cam_lon, cam_elev_a,
                get_elev_a, step_m=0.5, max_range_m=args.max_range,
            )
            r_b = pixel_to_gps_terrain(
                (u_ud, v_ud), K_new, R, cam_lat, cam_lon, cam_elev_b,
                get_elev_b, step_m=0.5, max_range_m=args.max_range,
            )
            if r_a is not None:
                lat_a, lon_a, elev_a, slant_a = r_a
                points_a.append({"u": u_r, "v": v_r, "lat": lat_a, "lon": lon_a,
                                  "elev_m": elev_a, "slant_range_m": slant_a})
            if r_b is not None:
                lat_b, lon_b, elev_b, slant_b = r_b
                points_b.append({"u": u_r, "v": v_r, "lat": lat_b, "lon": lon_b,
                                  "elev_m": elev_b, "slant_range_m": slant_b})
            if r_a is not None and r_b is not None:
                disp_m = haversine_m(lat_a, lon_a, lat_b, lon_b)
                slant_m = 0.5 * (slant_a + slant_b)
                compared.append({
                    "u": u_r, "v": v_r,
                    "lat": lat_a, "lon": lon_a, "elev_m": elev_a,
                    "displacement_m": disp_m, "slant_range_m": slant_m,
                    "bucket": bucket_label(slant_m),
                })

    print(f"[GRID] attempted={n_attempted}  dsm_a_hits={len(points_a)}  "
          f"dsm_b_hits={len(points_b)}  both_hit={len(compared)}")

    if not compared:
        print("[ERROR] No grid points intersected both DSMs — check coverage/orientation.")
        return 1

    displacements = [c["displacement_m"] for c in compared]
    overall = summarize(displacements)

    print("\n" + "=" * 66)
    print("DISPLACEMENT SUMMARY (DSM-A vs DSM-B)")
    print("=" * 66)
    print(f"  n={overall['n']}  mean={overall['mean_m']:.3f}m  median={overall['median_m']:.3f}m  "
          f"p90={overall['p90_m']:.3f}m  max={overall['max_m']:.3f}m")

    buckets = {}
    for label in ("<10m", "10-20m", ">20m"):
        vals = [c["displacement_m"] for c in compared if c["bucket"] == label]
        if vals:
            buckets[label] = summarize(vals)
            b = buckets[label]
            print(f"  [{label:>7}] n={b['n']:4d}  mean={b['mean_m']:.3f}m  "
                  f"median={b['median_m']:.3f}m  p90={b['p90_m']:.3f}m  max={b['max_m']:.3f}m")
        else:
            print(f"  [{label:>7}] n=0")

    # ── Export ──
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dsm_a_points.geojson"), "w") as f:
        json.dump(to_geojson(points_a, ["u", "v", "elev_m", "slant_range_m"]), f, indent=2)
    with open(os.path.join(args.out_dir, "dsm_b_points.geojson"), "w") as f:
        json.dump(to_geojson(points_b, ["u", "v", "elev_m", "slant_range_m"]), f, indent=2)
    with open(os.path.join(args.out_dir, "dsm_comparison.geojson"), "w") as f:
        json.dump(to_geojson(compared, ["u", "v", "elev_m", "displacement_m", "slant_range_m", "bucket"]), f, indent=2)

    report = {
        "image": args.image,
        "dsm_a": args.dsm_a,
        "dsm_b": args.dsm_b,
        "camera": {"lat": cam_lat, "lon": cam_lon, "heading_deg": heading, "pitch_deg": pitch,
                    "roll_deg": roll, "mount_height_m": height_agl},
        "grid_spacing_px": args.spacing,
        "n_attempted": n_attempted,
        "n_dsm_a_hits": len(points_a),
        "n_dsm_b_hits": len(points_b),
        "n_both_hit": len(compared),
        "overall": overall,
        "buckets": buckets,
    }
    with open(os.path.join(args.out_dir, "dsm_comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SAVE] {args.out_dir}/dsm_a_points.geojson")
    print(f"[SAVE] {args.out_dir}/dsm_b_points.geojson")
    print(f"[SAVE] {args.out_dir}/dsm_comparison.geojson")
    print(f"[SAVE] {args.out_dir}/dsm_comparison_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
