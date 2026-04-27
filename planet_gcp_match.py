"""
planet_gcp_match.py
====================
Automatic GCP generation using feature matching between:
  - a field photo taken from the camera (your image)
  - a Planet orthorectified GeoTIFF (already mapped to Earth coordinates)

It produces a repo-compatible GCP CSV:
  label,pixel_u,pixel_v,lat,lon,elev_m

This is an initialization tool: the output GCPs can be fed into
`georeference_tool.py` (flat-ground refinement) and/or used as a sanity check.

Notes:
  - Pixel coordinates are output in either `original` or `undistorted` pixel space.
  - For `undistorted` pixel space, this script uses the same `undistort()` helper
    as `georeference_tool.py` so the pixel coordinate systems match.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rasterio
from pyproj import CRS, Transformer

try:
    from georeference_tool import load_calibrated_intrinsics, scale_intrinsics_for_resolution, undistort

    _HAS_GEOREF_TOOL = True
except Exception:
    _HAS_GEOREF_TOOL = False


@dataclass(frozen=True)
class MatchPair:
    field_u: float
    field_v: float
    planet_u: float
    planet_v: float


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    # OpenCV expects BGR; convert to gray via cv2. If input is RGB, cvtColor still
    # gives a reasonable luminance ordering.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _select_by_grid(
    pts: Sequence[MatchPair],
    field_w: int,
    field_h: int,
    max_points: int,
    grid_cols: int = 4,
    grid_rows: int = 3,
) -> List[MatchPair]:
    """
    Choose at most one point per grid cell (based on field image coords) to
    spread GCPs across the photo and reduce collinear degeneracy.
    """
    if max_points <= 0:
        return []
    if not pts:
        return []

    cell_w = max(1, field_w / grid_cols)
    cell_h = max(1, field_h / grid_rows)

    chosen: List[MatchPair] = []
    seen = set()
    for p in pts:
        c = int(p.field_u // cell_w)
        r = int(p.field_v // cell_h)
        c = min(grid_cols - 1, max(0, c))
        r = min(grid_rows - 1, max(0, r))
        key = (r, c)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(p)
        if len(chosen) >= max_points:
            return chosen

    return chosen


def _write_gcp_csv(path: Path, gcps: Sequence[Tuple[str, float, float, float, float]]) -> None:
    """
    Write rows as:
      label,pixel_u,pixel_v,lat,lon,elev_m
    with `elev_m` left empty.
    """
    fieldnames = ["label", "pixel_u", "pixel_v", "lat", "lon", "elev_m"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, pixel_u, pixel_v, lat, lon in gcps:
            w.writerow(
                {
                    "label": label,
                    "pixel_u": pixel_u,
                    "pixel_v": pixel_v,
                    "lat": lat,
                    "lon": lon,
                    "elev_m": "",
                }
            )


def _planet_pixel_to_latlon(
    src: rasterio.io.DatasetReader,
    col: float,
    row: float,
    transformer_to_wgs84: Transformer,
) -> Tuple[float, float]:
    x, y = src.transform * (col, row)  # (easting, northing) in src CRS
    lon, lat = transformer_to_wgs84.transform(x, y)
    return float(lat), float(lon)


def _load_planet_for_matching(planet_tif: Path, planet_max_width: int) -> Tuple[np.ndarray, rasterio.DatasetReader, float, float]:
    """
    Returns:
      - grayscale planet image for matching (resized if needed)
      - open rasterio dataset handle
      - scale_x, scale_y: resized / original
    """
    src = rasterio.open(planet_tif)
    band_count = src.count
    if band_count >= 3:
        rgb = src.read([1, 2, 3])
        img = np.moveaxis(rgb, 0, -1)
    else:
        img = src.read(1)
        img = img[:, :, None]

    gray = _to_gray(img)

    h0, w0 = gray.shape[:2]
    if w0 <= planet_max_width:
        return gray, src, 1.0, 1.0

    scale = planet_max_width / float(w0)
    w1 = int(round(w0 * scale))
    h1 = int(round(h0 * scale))
    resized = cv2.resize(gray, (w1, h1), interpolation=cv2.INTER_AREA)
    return resized, src, float(w1) / float(w0), float(h1) / float(h0)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate GCPs by matching a field photo to a Planet GeoTIFF.")
    p.add_argument("--field-image", required=True, help="Path to your field photo (jpg/png)")
    p.add_argument("--planet-tif", required=True, help="Planet orthorectified GeoTIFF (clipped or full)")
    p.add_argument("--output-csv", default="./planet_matched_gcp.csv", help="Output GCP CSV path")
    p.add_argument("--pixel-space", choices=["original", "undistorted"], default="undistorted")
    p.add_argument("--calibration", default="./calibration.json", help="Calibration JSON for undistortion")
    p.add_argument("--max-gcps", type=int, default=12, help="Max GCPs to output (spread across image)")
    p.add_argument("--orb-features", type=int, default=5000, help="ORB features per image")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    p.add_argument("--ransac-threshold", type=float, default=3.0, help="Homography RANSAC pixel threshold")
    p.add_argument("--min-inlier-matches", type=int, default=10, help="Minimum inliers to accept homography")
    p.add_argument("--planet-max-width", type=int, default=1600, help="Resize Planet image for matching")
    args = p.parse_args()

    field_path = Path(args.field_image)
    planet_path = Path(args.planet_tif)
    out_path = Path(args.output_csv)

    if not field_path.exists():
        print(f"[ERR] Field image not found: {field_path}")
        return 2
    if not planet_path.exists():
        print(f"[ERR] Planet GeoTIFF not found: {planet_path}")
        return 2

    field = cv2.imread(str(field_path), cv2.IMREAD_COLOR)
    if field is None:
        print(f"[ERR] Could not read field image: {field_path}")
        return 2

    if args.pixel_space == "undistorted":
        if not _HAS_GEOREF_TOOL:
            print("[ERR] georeference_tool import failed; cannot undistort. Use --pixel-space original.")
            return 2
        if not Path(args.calibration).exists():
            print(f"[ERR] Calibration JSON not found: {args.calibration}")
            return 2
        h, w = field.shape[:2]
        K, D, calib_img_size, _ = load_calibrated_intrinsics(args.calibration)
        if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (w, h):
            K = scale_intrinsics_for_resolution(K, calib_img_size[0], calib_img_size[1], w, h)
        field, K_undist = undistort(field, K, D)  # noqa: F841 (K_undist kept for future)

    field_w = int(field.shape[1])
    field_h = int(field.shape[0])

    field_gray = _to_gray(field)

    planet_gray, src, scale_x, scale_y = _load_planet_for_matching(planet_path, args.planet_max_width)

    # CRS transform to WGS84 for CSV output.
    if src.crs is None:
        print("[ERR] Planet GeoTIFF has no CRS; cannot convert pixels to lat/lon.")
        return 2
    transformer_to_wgs84 = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)

    orb = cv2.ORB_create(nfeatures=args.orb_features)
    kp1, des1 = orb.detectAndCompute(field_gray, None)
    kp2, des2 = orb.detectAndCompute(planet_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("[ERR] Not enough keypoints detected for matching.")
        return 3

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)

    good: List[Tuple[np.ndarray, np.ndarray]] = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < args.ratio * n.distance:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt
            good.append((np.array([u1, v1], dtype=np.float32), np.array([u2, v2], dtype=np.float32)))

    if len(good) < 4:
        print(f"[ERR] Too few matches after ratio test: {len(good)}")
        return 3

    src_pts = np.array([g[0] for g in good], dtype=np.float32)
    dst_pts = np.array([g[1] for g in good], dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=args.ransac_threshold)
    if H is None or mask is None:
        print("[ERR] Homography estimation failed.")
        return 3

    inliers_mask = mask.ravel().astype(bool)
    inlier_pairs = [good[i] for i in range(len(good)) if inliers_mask[i]]
    if len(inlier_pairs) < args.min_inlier_matches:
        print(f"[ERR] Too few homography inliers: {len(inlier_pairs)} (need >= {args.min_inlier_matches})")
        return 3

    match_pairs: List[MatchPair] = []
    for src_xy, dst_xy in inlier_pairs:
        field_u, field_v = float(src_xy[0]), float(src_xy[1])
        # dst_xy is in resized Planet pixel coordinates
        planet_u, planet_v = float(dst_xy[0]), float(dst_xy[1])
        match_pairs.append(MatchPair(field_u=field_u, field_v=field_v, planet_u=planet_u, planet_v=planet_v))

    selected = _select_by_grid(match_pairs, field_w=field_w, field_h=field_h, max_points=args.max_gcps)

    # Convert Planet pixels (resized) back to original to geolocate.
    gcps_out: List[Tuple[str, float, float, float, float]] = []
    for i, s in enumerate(selected):
        planet_col = s.planet_u / scale_x
        planet_row = s.planet_v / scale_y
        lat, lon = _planet_pixel_to_latlon(src, planet_col, planet_row, transformer_to_wgs84)
        label = f"planet_{i+1}"
        gcps_out.append((label, s.field_u, s.field_v, lat, lon))

    src.close()

    _write_gcp_csv(out_path, gcps_out)
    print(f"[OK] Wrote {len(gcps_out)} GCPs to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

