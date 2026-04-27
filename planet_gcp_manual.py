"""
planet_gcp_manual.py
=====================
Manual GCP generation using clicks on:
  - your field photo
  - a Planet orthorectified GeoTIFF

This is the human-in-the-loop fallback when automatic matching fails or when
you want to curate correspondences carefully.

For each GCP:
  1. Left-click on the field image.
  2. Left-click the corresponding location on the Planet GeoTIFF display.
The tool writes a repo-compatible GCP CSV:
  label,pixel_u,pixel_v,lat,lon,elev_m

Usage example:
  python planet_gcp_manual.py --field-image field.jpg --planet-tif planet.tif --output-csv gcp.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from pyproj import CRS, Transformer

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


def _planet_pixel_to_latlon(
    src: rasterio.io.DatasetReader,
    col: float,
    row: float,
    transformer_to_wgs84: Transformer,
) -> Tuple[float, float]:
    x, y = src.transform * (col, row)  # (easting, northing) in src CRS
    lon, lat = transformer_to_wgs84.transform(x, y)
    return float(lat), float(lon)


def _load_planet_display(planet_tif: Path, planet_max_width: int):
    src = rasterio.open(planet_tif)

    if src.count >= 3:
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


def _write_gcp_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "pixel_u", "pixel_v", "lat", "lon", "elev_m"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(description="Manually generate GCPs by clicking field image + Planet GeoTIFF.")
    p.add_argument("--field-image", required=True, help="Path to field photo")
    p.add_argument("--planet-tif", required=True, help="Path to Planet orthorectified GeoTIFF")
    p.add_argument("--output-csv", default="./planet_manual_gcp.csv", help="Output GCP CSV")
    p.add_argument("--label-prefix", default="planet_manual", help="Label prefix for generated GCPs")
    p.add_argument("--pixel-space", choices=["original", "undistorted"], default="undistorted")
    p.add_argument("--calibration", default="./calibration.json", help="Calibration JSON for undistortion")
    p.add_argument("--planet-max-width", type=int, default=1600, help="Resize Planet display for clicking")
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
        field, _K_undist = undistort(field, K, D)  # noqa: F841

    field_disp = field.copy()
    field_gray = _to_gray(field)
    field_h, field_w = field_gray.shape[:2]

    planet_gray, src, scale_x, scale_y = _load_planet_display(planet_path, args.planet_max_width)

    if src.crs is None:
        print("[ERR] Planet GeoTIFF has no CRS; cannot convert clicks to lat/lon.")
        return 2
    transformer_to_wgs84 = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)

    # State
    pending_field: Optional[Tuple[float, float]] = None
    rows: List[dict] = []
    counter = 0

    WIN_FIELD = "Field (click first)"
    WIN_PLANET = "Planet (click corresponding)"
    cv2.namedWindow(WIN_FIELD, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PLANET, cv2.WINDOW_NORMAL)

    def redraw():
        nonlocal field_disp
        field_disp = field.copy()
        if pending_field is not None:
            u, v = pending_field
            cv2.circle(field_disp, (int(round(u)), int(round(v))), 6, (0, 165, 255), -1)
            cv2.putText(
                field_disp,
                "Now click the corresponding point on the Planet window",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                1,
            )
        else:
            cv2.putText(
                field_disp,
                "Left-click field point. When done: S save, Q quit.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
        cv2.imshow(WIN_FIELD, field_disp)
        cv2.imshow(WIN_PLANET, planet_gray)

    def on_field(event, u, v, *_):
        nonlocal pending_field
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pending_field = (float(u), float(v))
        redraw()

    def on_planet(event, u, v, *_):
        nonlocal pending_field, counter
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if pending_field is None:
            print("[INFO] Click a field point first.")
            return

        field_u, field_v = pending_field
        planet_col = float(u) / scale_x
        planet_row = float(v) / scale_y
        lat, lon = _planet_pixel_to_latlon(src, planet_col, planet_row, transformer_to_wgs84)

        counter += 1
        label = f"{args.label_prefix}_{counter}"
        rows.append(
            {
                "label": label,
                "pixel_u": field_u,
                "pixel_v": field_v,
                "lat": lat,
                "lon": lon,
                "elev_m": "",
            }
        )
        pending_field = None
        print(f"[ADD] {label}: field({field_u:.1f},{field_v:.1f}) -> lat={lat:.6f}, lon={lon:.6f}")
        redraw()

    cv2.setMouseCallback(WIN_FIELD, on_field)
    cv2.setMouseCallback(WIN_PLANET, on_planet)

    print("[MANUAL] Workflow:")
    print("  1) Left-click field point.")
    print("  2) Left-click corresponding Planet point.")
    print("  3) Press S to save CSV, Q to quit.")
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            if rows:
                _write_gcp_csv(out_path, rows)
                print(f"[SAVE] {len(rows)} GCPs -> {out_path}")
            else:
                print("[SAVE] No GCPs yet.")
            redraw()

    src.close()
    cv2.destroyAllWindows()
    if rows:
        _write_gcp_csv(out_path, rows)
        print(f"[SAVE] {len(rows)} GCPs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

