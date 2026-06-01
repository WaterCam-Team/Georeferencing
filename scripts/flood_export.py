"""
flood_export.py
===============
Batch flood-extent export: project a binary flood mask to geographic coordinates
and write a GeoJSON polygon and GeoTIFF raster suitable for QGIS / hydraulic
models.

Workflow
--------
1. Load camera pose from unit config JSON + EXIF + CLI overrides.
2. Load terrain (DEM GeoTIFF and/or LAS/LAZ) via make_terrain_provider().
3. Load a pre-computed binary flood mask (white = flood water).
4. Extract the flood boundary contour (Douglas-Peucker simplified).
5. Project each contour pixel to (lat, lon, elev) via pixel_to_gps_terrain().
6. Optionally compute convex hull of the GPS boundary polygon.
7. Export:
     flood_extent.geojson   — WGS84 FeatureCollection (QGIS-ready)
     flood_extent.tif       — binary GeoTIFF (1=flood, 0=dry, EPSG:4326)
     flood_boundary.csv     — tabular GPS boundary points

Typical usage
-------------
    # With unit config (reads mount height + heading from JSON):
    python scripts/flood_export.py field.jpg \\
        --mask flood_mask.png \\
        --dem drone_dsm/dsm_ingest.tif \\
        --unit-config unit_config_UFO006.json \\
        --calib calibration.json \\
        --out-dir ./flood_export

    # Fully explicit pose:
    python scripts/flood_export.py field.jpg \\
        --mask flood_mask.png \\
        --dem dsm.tif \\
        --lat 43.065 --lon -76.173 \\
        --heading 265 --pitch -25 --roll 0 \\
        --height-above-ground 0.97 \\
        --out-dir ./flood_export

    # Convex hull instead of raw contour (cleaner polygon for hydraulic models):
    python scripts/flood_export.py field.jpg --mask mask.png --dem dsm.tif \\
        --unit-config unit_config_UFO006.json --hull

Requires: rasterio, numpy, opencv-python, scipy, pyproj, Pillow
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import rasterio
    from rasterio.crs import CRS as RasterioCRS
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds as transform_from_bounds
except ImportError:
    sys.exit("Install rasterio: pip install rasterio")

from camera_geometry import build_rotation_matrix
from exif_imu import read_gps_imu_from_exif
import unit_config as uc
from georeference_terrain import (
    _load_intrinsics,
    make_terrain_provider,
    pixel_to_gps_terrain,
)
from flood_map import mask_to_contour


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def project_contour(
    contour: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    cam_lat: float,
    cam_lon: float,
    cam_elev_m: float,
    get_elevation: Callable,
    step_m: float = 0.5,
    max_range_m: float = 500.0,
) -> list[dict]:
    """
    Project pixel contour points to GPS via terrain ray-casting.

    contour : (N, 2) array of (x, y) pixel coordinates
    Returns list of dicts with keys lat, lon, elev_m, slant_range_m.
    Points that miss the terrain surface are silently dropped.
    """
    results = []
    n = len(contour)
    for i, (px, py) in enumerate(contour):
        if i % max(1, n // 40) == 0:
            print(f"\r  Projecting {i}/{n} ({100*i//n}%) ...",
                  end="", flush=True)
        hit = pixel_to_gps_terrain(
            (float(px), float(py)), K, R,
            cam_lat, cam_lon, cam_elev_m,
            get_elevation,
            step_m=step_m,
            max_range_m=max_range_m,
        )
        if hit is None:
            continue
        lat, lon, elev_m, slant_range_m = hit
        results.append({
            "lat": lat, "lon": lon,
            "elev_m": elev_m, "slant_range_m": slant_range_m,
        })
    print(f"\r  Projected {len(results)}/{n} contour points to GPS.  ")
    return results


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def gps_points_to_polygon(
    gps_points: list[dict],
    hull: bool = False,
) -> list[tuple[float, float]] | None:
    """
    Convert GPS boundary points to a closed polygon ring [(lon, lat), ...].

    hull : if True, compute the convex hull in GPS space (fewer vertices,
           suitable for hydraulic model input).
    Returns None if fewer than 3 points.
    """
    if len(gps_points) < 3:
        return None

    coords = [(p["lon"], p["lat"]) for p in gps_points]

    if hull:
        try:
            from scipy.spatial import ConvexHull
            pts = np.array(coords)
            ch = ConvexHull(pts)
            # ConvexHull.vertices are in counter-clockwise order
            verts = list(ch.vertices)
            ring = [coords[i] for i in verts]
        except Exception as exc:
            print(f"  [WARN] ConvexHull failed ({exc}); using raw contour.")
            ring = coords
    else:
        ring = coords

    # Ensure the ring is closed
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]

    return ring


def build_geojson(
    ring: list[tuple[float, float]],
    properties: dict,
) -> dict:
    """Wrap a polygon ring in a GeoJSON FeatureCollection."""
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": properties,
    }
    return {"type": "FeatureCollection", "features": [feature]}


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_geojson(geojson: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    print(f"  GeoJSON → {out_path}")


def export_geotiff(
    ring: list[tuple[float, float]],
    out_path: Path,
    resolution_deg: float = 0.00001,  # ~1.1 m at 43 °N
) -> None:
    """
    Rasterize the flood polygon to a binary GeoTIFF (EPSG:4326).
    Pixel value 1 = flood, 0 = dry.
    """
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]

    pad_lon = max((max(lons) - min(lons)) * 0.1, resolution_deg * 10)
    pad_lat = max((max(lats) - min(lats)) * 0.1, resolution_deg * 10)
    west  = min(lons) - pad_lon
    east  = max(lons) + pad_lon
    south = min(lats) - pad_lat
    north = max(lats) + pad_lat

    width  = max(1, int((east - west) / resolution_deg))
    height = max(1, int((north - south) / resolution_deg))

    transform = transform_from_bounds(west, south, east, north, width, height)
    shape_geom = {"type": "Polygon", "coordinates": [ring]}

    burned = rio_rasterize(
        [(shape_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=height, width=width,
        count=1, dtype="uint8",
        crs=RasterioCRS.from_epsg(4326),
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(burned, 1)
    print(f"  GeoTIFF → {out_path}  ({width}×{height} px, {resolution_deg:.5f}°/px)")


def export_csv(gps_points: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lat", "lon", "elev_m", "slant_range_m"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in gps_points:
            w.writerow({k: f"{p[k]:.6f}" for k in fieldnames})
    print(f"  CSV     → {out_path}  ({len(gps_points)} points)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export flood extent from a binary mask to GeoJSON + GeoTIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image", help="Field photo (JPEG/PNG); provides EXIF GPS/IMU fallback")
    p.add_argument("--mask", required=True,
                   help="Binary flood mask image (white=flood, black=dry)")
    p.add_argument("--dem", default=None, help="DEM GeoTIFF for terrain ray-casting")
    p.add_argument("--las", default=None, help="LAS/LAZ point cloud (supplement to DEM)")
    p.add_argument("--las-crs", type=int, default=None, metavar="EPSG",
                   help="EPSG for LAS when file has no embedded CRS")

    uc.add_argument(p)   # adds --unit-config

    p.add_argument("--calib", default=None, help="calibration.json path")
    p.add_argument("--lat",     type=float, default=None, help="Camera latitude (WGS84)")
    p.add_argument("--lon",     type=float, default=None, help="Camera longitude (WGS84)")
    p.add_argument("--heading", type=float, default=None, help="Camera heading (deg, 0=N)")
    p.add_argument("--pitch",   type=float, default=None, help="Camera pitch (deg, neg=down)")
    p.add_argument("--roll",    type=float, default=None, help="Camera roll (deg)")
    p.add_argument("--height-above-ground", type=float, default=None, metavar="M",
                   help="Camera height above ground (m); used with DEM to get cam elevation")
    p.add_argument("--terrain-vertical-datum", default=None,
                   help="Vertical datum of DEM (e.g. wgs84_ellipsoid, navd88)")

    p.add_argument("--out-dir", default="./flood_export",
                   help="Output directory (default: ./flood_export)")
    p.add_argument("--hull", action="store_true",
                   help="Use convex hull of GPS boundary (fewer vertices, cleaner polygon)")
    p.add_argument("--epsilon", type=float, default=3.0,
                   help="Douglas-Peucker simplification tolerance in pixels (default: 3.0)")
    p.add_argument("--min-area", type=float, default=500.0,
                   help="Minimum contour area in pixels to process (default: 500)")
    p.add_argument("--step-m", type=float, default=0.5,
                   help="Terrain ray march step in metres (default: 0.5)")
    p.add_argument("--max-range-m", type=float, default=500.0,
                   help="Maximum terrain ray range in metres (default: 500)")
    p.add_argument("--resolution-deg", type=float, default=0.00001,
                   help="GeoTIFF output resolution in degrees (~1.1m at 43°N; default: 0.00001)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)
    mask_path  = Path(args.mask)
    out_dir    = Path(args.out_dir)

    if not image_path.exists():
        print(f"[ERR] Image not found: {image_path}", file=sys.stderr)
        return 2
    if not mask_path.exists():
        print(f"[ERR] Mask not found: {mask_path}", file=sys.stderr)
        return 2

    # ── Unit config + EXIF ───────────────────────────────────────────────────
    cfg = uc.from_args(args)
    exif = read_gps_imu_from_exif(image_path)
    config_dir = str(Path(args.unit_config).parent) if args.unit_config else "."

    cam_lat = args.lat or exif["lat"]
    cam_lon = args.lon or exif["lon"]
    if cam_lat is None or cam_lon is None:
        print("[ERR] Camera lat/lon not found in EXIF and not provided via --lat/--lon.",
              file=sys.stderr)
        return 2

    heading, heading_src = cfg.resolve_heading(args.heading, exif["yaw_deg"])
    pitch,   pitch_src   = cfg.resolve_pitch(args.pitch, exif["pitch_deg"])
    roll,    roll_src    = cfg.resolve_roll(args.roll, exif["roll_deg"])
    height,  height_src  = cfg.resolve_mount_height(args.height_above_ground)

    print(f"\n=== Camera pose ===")
    print(f"  Position:  lat={cam_lat:.6f}  lon={cam_lon:.6f}")
    print(f"  Heading:   {heading:.1f}°  ({heading_src})")
    print(f"  Pitch:     {pitch:.1f}°  ({pitch_src})")
    print(f"  Roll:      {roll:.1f}°  ({roll_src})")
    if height is not None:
        print(f"  Height AGL:{height:.4f} m  ({height_src})")

    # ── Terrain ──────────────────────────────────────────────────────────────
    if args.dem is None and args.las is None:
        print("[ERR] Provide at least one of --dem or --las for terrain ray-casting.",
              file=sys.stderr)
        return 2

    print(f"\n=== Loading terrain ===")
    try:
        get_elevation, dem_bounds, las_bounds, inferred_datum = make_terrain_provider(
            dem_path=args.dem,
            las_path=args.las,
            las_crs_epsg=args.las_crs,
        )
    except Exception as exc:
        # make_terrain_provider raises ValueError for missing files, but
        # rasterio raises RasterioIOError (not a ValueError subclass) for
        # corrupt or unreadable GeoTIFFs.
        print(f"[ERR] Could not load terrain: {exc}", file=sys.stderr)
        return 2

    # Resolve camera elevation in terrain datum
    terrain_datum = args.terrain_vertical_datum or inferred_datum or "wgs84_ellipsoid"
    ground_elev = get_elevation(cam_lon, cam_lat)
    if height is not None and ground_elev is not None:
        cam_elev_m = ground_elev + height
        print(f"  Camera elevation: {ground_elev:.2f} (ground) + {height:.4f} (AGL) "
              f"= {cam_elev_m:.2f} m  [{terrain_datum}]")
    elif height is None and ground_elev is not None:
        # DEM covers the camera position but mount height is unknown — warn
        # explicitly so the user knows the DEM is not being used for elevation.
        print(f"  [WARN] DEM ground elevation at camera = {ground_elev:.2f} m, "
              f"but --height-above-ground not provided.", file=sys.stderr)
        print(f"         Pass --height-above-ground <m> (e.g. {ground_elev:.2f} + mount_m) "
              f"for terrain-accurate camera elevation.", file=sys.stderr)
        if exif["alt"] is not None:
            cam_elev_m = float(exif["alt"])
            print(f"  Camera elevation from EXIF: {cam_elev_m:.2f} m  "
                  f"(datum may differ from terrain)", file=sys.stderr)
        else:
            print("[ERR] Cannot determine camera elevation — no EXIF altitude either. "
                  "Provide --height-above-ground.", file=sys.stderr)
            return 2
    elif exif["alt"] is not None:
        cam_elev_m = float(exif["alt"])
        print(f"  Camera elevation from EXIF: {cam_elev_m:.2f} m  "
              f"(datum may differ from terrain — prefer --height-above-ground)")
    else:
        print("[ERR] Cannot determine camera elevation. "
              "Provide --height-above-ground or a photo with EXIF altitude.",
              file=sys.stderr)
        return 2

    # ── Intrinsics ───────────────────────────────────────────────────────────
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERR] Could not read image: {image_path}", file=sys.stderr)
        return 2
    h_img, w_img = img.shape[:2]
    calib_path = cfg.resolve_calibration(args.calib, config_dir)
    K, D = _load_intrinsics(calib_path, w_img, h_img)
    R = build_rotation_matrix(heading_deg=heading, pitch_deg=pitch, roll_deg=roll)

    # ── Mask ─────────────────────────────────────────────────────────────────
    print(f"\n=== Loading mask ===")
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"[ERR] Could not read mask: {mask_path}", file=sys.stderr)
        return 2
    # Resize mask to match image if needed
    if mask_img.shape[:2] != (h_img, w_img):
        mask_img = cv2.resize(mask_img, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    flood_mask = mask_img > 127
    flood_px = int(flood_mask.sum())
    print(f"  Flood pixels: {flood_px:,} / {h_img * w_img:,} "
          f"({100 * flood_px / (h_img * w_img):.1f}%)")

    # ── Contour ──────────────────────────────────────────────────────────────
    print(f"\n=== Extracting contour (ε={args.epsilon} px, min_area={args.min_area} px) ===")
    contour = mask_to_contour(flood_mask, epsilon_px=args.epsilon,
                               min_area_px=args.min_area)
    if contour is None:
        print("[ERR] No flood contour found — mask may be empty or too small.",
              file=sys.stderr)
        return 1
    print(f"  Contour: {len(contour)} vertices after simplification")

    # ── Project to GPS ───────────────────────────────────────────────────────
    print(f"\n=== Projecting contour to GPS ===")
    gps_points = project_contour(
        contour, K, R,
        cam_lat, cam_lon, cam_elev_m,
        get_elevation,
        step_m=args.step_m,
        max_range_m=args.max_range_m,
    )
    if len(gps_points) < 3:
        print(f"[ERR] Only {len(gps_points)} GPS points projected "
              f"(need ≥ 3 for a polygon). "
              f"Check camera pose, terrain coverage, and ray range.",
              file=sys.stderr)
        return 1

    # ── Build polygon ────────────────────────────────────────────────────────
    ring = gps_points_to_polygon(gps_points, hull=args.hull)
    if ring is None:
        print("[ERR] Could not form polygon from GPS points.", file=sys.stderr)
        return 1
    if args.hull:
        print(f"  Convex hull: {len(ring) - 1} vertices")

    # ── Export ───────────────────────────────────────────────────────────────
    print(f"\n=== Exporting to {out_dir} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    props = {
        "image":            image_path.name,
        "camera_lat":       cam_lat,
        "camera_lon":       cam_lon,
        "camera_elev_m":    cam_elev_m,
        "terrain_datum":    terrain_datum,
        "heading_deg":      heading,
        "pitch_deg":        pitch,
        "roll_deg":         roll,
        "n_contour_pts":    len(contour),
        "n_gps_pts":        len(gps_points),
        "hull":             args.hull,
    }

    geojson = build_geojson(ring, props)
    export_geojson(geojson, out_dir / "flood_extent.geojson")
    export_geotiff(ring, out_dir / "flood_extent.tif",
                   resolution_deg=args.resolution_deg)
    export_csv(gps_points, out_dir / "flood_boundary.csv")

    print(f"\nDone. Open flood_extent.geojson in QGIS (drag-and-drop).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
