"""
drone_gcp_match.py
==================
Automatic GCP generation using feature matching between:
  - a field photo from the fixed camera (oblique, ground-level)
  - a drone orthomosaic GeoTIFF (nadir, georeferenced)

Produces the same repo-compatible GCP CSV as planet_gcp_match.py:
  label,pixel_u,pixel_v,lat,lon,elev_m

Differences from planet_gcp_match.py
--------------------------------------
  - Default detector: SIFT (scale-invariant; handles the large scale ratio
    between the oblique field view and the nadir drone ortho at 3–5 cm GSD)
  - Tighter ratio test (0.70 vs 0.75) — drone texture is richer than Planet 3m
  - Tighter RANSAC threshold (2.0 px vs 3.0 px) — higher-res ortho warrants it
  - --detector flag to switch to AKAZE or ORB
  - Higher default feature count (8000 vs 5000)

SIFT is available in the opencv-python main module since version 4.4.
requirements.txt pins opencv-python>=4.8, so SIFT is always available.

Shared utilities imported from planet_gcp_match.py:
  _select_by_grid, _write_gcp_csv, _planet_pixel_to_latlon, _load_planet_for_matching
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from pyproj import CRS, Transformer

from planet_gcp_match import (
    MatchPair,
    _load_planet_for_matching as _load_ortho_for_matching,
    _planet_pixel_to_latlon as _ortho_pixel_to_latlon,
    _select_by_grid,
    _write_gcp_csv,
)

try:
    from georeference_tool import load_calibrated_intrinsics, scale_intrinsics_for_resolution, undistort
    _HAS_GEOREF_TOOL = True
except Exception:
    _HAS_GEOREF_TOOL = False


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def build_detector(name: str, nfeatures: int):
    """
    Factory for feature detectors.

    name: 'sift' | 'akaze' | 'orb'
    Returns an opencv feature detector/descriptor object.
    """
    name = name.lower()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    if name == "akaze":
        return cv2.AKAZE_create()
    if name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    raise ValueError(f"Unknown detector {name!r}; choose sift, akaze, or orb")


def _norm_type(detector_name: str) -> int:
    """BFMatcher norm: HAMMING for binary descriptors (ORB, AKAZE), L2 for float (SIFT)."""
    if detector_name.lower() in ("orb", "akaze"):
        return cv2.NORM_HAMMING
    return cv2.NORM_L2


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate GCPs by matching a field photo to a drone orthomosaic GeoTIFF."
    )
    p.add_argument("--field-image", required=True, help="Path to your field photo (jpg/png)")
    p.add_argument("--ortho-tif", required=True,
                   help="Drone orthomosaic GeoTIFF (georeferenced, any CRS)")
    p.add_argument("--output-csv", default="./drone_matched_gcp.csv",
                   help="Output GCP CSV path (default: ./drone_matched_gcp.csv)")
    p.add_argument("--pixel-space", choices=["original", "undistorted"], default="undistorted")
    p.add_argument("--calibration", default="./calibration.json",
                   help="Calibration JSON for undistortion (default: ./calibration.json)")
    p.add_argument("--detector", choices=["sift", "akaze", "orb"], default="sift",
                   help="Feature detector (default: sift)")
    p.add_argument("--max-gcps", type=int, default=12,
                   help="Max GCPs to output, spread across image grid (default: 12)")
    p.add_argument("--nfeatures", type=int, default=8000,
                   help="Max features per image (default: 8000; ignored by AKAZE)")
    p.add_argument("--ratio", type=float, default=0.70,
                   help="Lowe ratio test threshold (default: 0.70)")
    p.add_argument("--ransac-threshold", type=float, default=2.0,
                   help="Homography RANSAC pixel threshold (default: 2.0)")
    p.add_argument("--min-inlier-matches", type=int, default=10,
                   help="Minimum inliers to accept homography (default: 10)")
    p.add_argument("--ortho-max-width", type=int, default=2000,
                   help="Resize ortho image for matching (default: 2000 px; "
                        "higher than planet default due to finer drone GSD)")
    args = p.parse_args()

    field_path = Path(args.field_image)
    ortho_path = Path(args.ortho_tif)
    out_path = Path(args.output_csv)

    if not field_path.exists():
        print(f"[ERR] Field image not found: {field_path}")
        return 2
    if not ortho_path.exists():
        print(f"[ERR] Ortho GeoTIFF not found: {ortho_path}")
        return 2

    field = cv2.imread(str(field_path), cv2.IMREAD_COLOR)
    if field is None:
        print(f"[ERR] Could not read field image: {field_path}")
        return 2

    if args.pixel_space == "undistorted":
        if not _HAS_GEOREF_TOOL:
            print("[ERR] georeference_tool import failed; use --pixel-space original.")
            return 2
        if not Path(args.calibration).exists():
            print(f"[ERR] Calibration JSON not found: {args.calibration}")
            return 2
        h, w = field.shape[:2]
        K, D, calib_img_size, _ = load_calibrated_intrinsics(args.calibration)
        if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (w, h):
            K = scale_intrinsics_for_resolution(K, calib_img_size[0], calib_img_size[1], w, h)
        field, _K_undist = undistort(field, K, D)

    field_w, field_h = field.shape[1], field.shape[0]
    field_gray = _to_gray(field)

    ortho_gray, src, scale_x, scale_y = _load_ortho_for_matching(
        ortho_path, args.ortho_max_width
    )

    try:
        if src.crs is None:
            print("[ERR] Ortho GeoTIFF has no CRS; cannot convert pixels to lat/lon.")
            return 2
        # Transformer.from_crs() can raise; src must be closed via finally regardless.
        transformer_to_wgs84 = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)

        # --- Feature detection and matching ---
        det = build_detector(args.detector, args.nfeatures)
        norm = _norm_type(args.detector)

        kp1, des1 = det.detectAndCompute(field_gray, None)
        kp2, des2 = det.detectAndCompute(ortho_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"[ERR] Not enough keypoints: field={len(kp1) if kp1 else 0}, "
                  f"ortho={len(kp2) if kp2 else 0}")
            return 3

        print(f"[INFO] Detected keypoints: field={len(kp1)}, ortho={len(kp2)}  "
              f"(detector={args.detector})")

        bf = cv2.BFMatcher(norm)
        knn = bf.knnMatch(des1, des2, k=2)

        good: List[Tuple[np.ndarray, np.ndarray]] = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < args.ratio * n.distance:
                u1, v1 = kp1[m.queryIdx].pt
                u2, v2 = kp2[m.trainIdx].pt
                good.append((
                    np.array([u1, v1], dtype=np.float32),
                    np.array([u2, v2], dtype=np.float32),
                ))

        print(f"[INFO] Matches after ratio test ({args.ratio}): {len(good)}")

        if len(good) < 4:
            print(f"[ERR] Too few matches after ratio test: {len(good)}")
            return 3

        src_pts = np.array([g[0] for g in good], dtype=np.float32)
        dst_pts = np.array([g[1] for g in good], dtype=np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                     ransacReprojThreshold=args.ransac_threshold)
        if H is None or mask is None:
            print("[ERR] Homography estimation failed.")
            return 3

        inliers_mask = mask.ravel().astype(bool)
        inlier_pairs = [good[i] for i in range(len(good)) if inliers_mask[i]]
        print(f"[INFO] RANSAC inliers: {len(inlier_pairs)} / {len(good)}")

        if len(inlier_pairs) < args.min_inlier_matches:
            print(f"[ERR] Too few RANSAC inliers: {len(inlier_pairs)} "
                  f"(need >= {args.min_inlier_matches})")
            return 3

        match_pairs: List[MatchPair] = []
        for src_xy, dst_xy in inlier_pairs:
            match_pairs.append(MatchPair(
                field_u=float(src_xy[0]), field_v=float(src_xy[1]),
                planet_u=float(dst_xy[0]), planet_v=float(dst_xy[1]),
            ))

        selected = _select_by_grid(match_pairs, field_w=field_w, field_h=field_h,
                                    max_points=args.max_gcps)

        gcps_out: List[Tuple[str, float, float, float, float]] = []
        for i, s in enumerate(selected):
            ortho_col = s.planet_u / scale_x
            ortho_row = s.planet_v / scale_y
            lat, lon = _ortho_pixel_to_latlon(src, ortho_col, ortho_row, transformer_to_wgs84)
            gcps_out.append((f"drone_{i+1}", s.field_u, s.field_v, lat, lon))

    finally:
        src.close()

    _write_gcp_csv(out_path, gcps_out)
    print(f"[OK] Wrote {len(gcps_out)} GCPs to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
