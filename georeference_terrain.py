"""
Georeference a photo using terrain (DEM and/or lidar point cloud)
=================================================================
Given a photo from a fixed-position camera and terrain from a GeoTIFF DEM
and/or LAS/LAZ point cloud, computes ground coordinates (lat, lon, elevation)
by intersecting the camera ray with the terrain surface instead of a flat plane.

USAGE:
    Run and follow prompts, or set the CONFIG block at the bottom.
    - Provide at least one of: --dem path.tif  and/or  --las path.las
    - Provide image, camera position (lat, lon, elevation), orientation,
      and calibration (or nominal). Click in the image to get (lat, lon, elev).

METHOD:
    1. Load terrain: GeoTIFF DEM and/or rasterize LAS/LAZ to a height grid.
    2. For each pixel: unproject to a 3D ray in ENU; march along the ray until
       ray height <= terrain elevation (same vertical datum); return (lon, lat, elev).
    3. Camera position and terrain elevations must use the same vertical datum
       (e.g. both orthometric/MSL or both ellipsoid).

DEPENDENCIES:
    pip install opencv-python numpy pyproj Pillow

    For GeoTIFF DEM:
        pip install rasterio

    For LAS/LAZ point cloud:
        pip install laspy

    Optional: use calibration and helpers from this repo
        (georeference_tool for rotation matrix, calibration loading, undistort)
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Optional, Callable, Tuple

import cv2
import numpy as np

# Optional: use georeference_tool for camera math and calibration
try:
    from georeference_tool import (
        build_rotation_matrix,
        load_calibrated_intrinsics,
        scale_intrinsics_for_resolution,
        undistort,
        read_gps_from_exif,
    )
    _HAS_GEOREF_TOOL = True
except ImportError:
    _HAS_GEOREF_TOOL = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: TERRAIN FROM GEOTIFF DEM
# ─────────────────────────────────────────────────────────────────────────────

def load_dem_geotiff(dem_path: str):
    """
    Load a GeoTIFF DEM and return a callable get_elevation(lon, lat) -> float or None.
    Uses rasterio; elevation is in the DEM's vertical units (usually meters).
    """
    try:
        import rasterio
        from rasterio.warp import transform
        from rasterio.crs import CRS
    except ImportError:
        raise ImportError("For GeoTIFF DEM support install: pip install rasterio")

    src = rasterio.open(dem_path)
    dem_crs = src.crs
    if dem_crs is None:
        dem_crs = CRS.from_epsg(4326)  # assume WGS84
    wgs84 = CRS.from_epsg(4326)
    transform_to_dem = None
    if dem_crs != wgs84:
        from pyproj import Transformer
        transform_to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)

    nodata = getattr(src, 'nodata', None)

    def get_elevation(lon: float, lat: float):
        if transform_to_dem is not None:
            x, y = transform_to_dem.transform(lon, lat)
        else:
            x, y = lon, lat
        row, col = src.index(x, y)
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None
        window = rasterio.windows.Window(col, row, 1, 1)
        data = src.read(1, window=window)
        if data is None or np.isnan(data).any():
            return None
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            if (data == nodata).any():
                return None
        return float(data.flat[0])

    return get_elevation, src


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TERRAIN FROM LAS/LAZ POINT CLOUD
# ─────────────────────────────────────────────────────────────────────────────

def load_las_terrain(las_path: str,
                    resolution_m: float = 1.0,
                    las_crs_epsg: Optional[int] = None):
    """
    Load a LAS/LAZ file, rasterize to a height grid (max Z per cell), and return
    get_elevation(lon, lat) and the raster extent/crs for consistency.

    resolution_m: grid cell size in meters (in the point cloud's XY CRS).
    las_crs_epsg: EPSG code for the point cloud (e.g. 32618 for UTM 18N).
                  If None, we try to read from the file (laspy 2.x may expose CRS).
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("For LAS/LAZ support install: pip install laspy")

    from pyproj import Transformer

    las = laspy.read(las_path)
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # CRS: try file first
    pc_crs = None
    if hasattr(las, 'header') and hasattr(las.header, 'parse_crs'):
        try:
            pc_crs = las.header.parse_crs()
        except Exception:
            pass
    if pc_crs is None and las_crs_epsg is not None:
        from pyproj import CRS
        pc_crs = CRS.from_epsg(las_crs_epsg)
    if pc_crs is None:
        pc_crs = "EPSG:4326"
    if hasattr(pc_crs, 'to_epsg'):
        pc_crs = f"EPSG:{pc_crs.to_epsg()}"
    transformer = Transformer.from_crs("EPSG:4326", pc_crs, always_xy=True)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cols = int(np.ceil((xmax - xmin) / resolution_m)) + 1
    rows = int(np.ceil((ymax - ymin) / resolution_m)) + 1
    grid = np.full((rows, cols), -np.inf, dtype=np.float64)

    col_idx = ((x - xmin) / resolution_m).astype(int)
    row_idx = ((y - ymin) / resolution_m).astype(int)
    np.clip(col_idx, 0, cols - 1, out=col_idx)
    np.clip(row_idx, 0, rows - 1, out=row_idx)
    # Max Z per cell (DSM-style) using numpy
    np.maximum.at(grid, (row_idx, col_idx), z)

    xmin_, ymin_ = xmin, ymin
    res = resolution_m

    def get_elevation(lon: float, lat: float):
        xx, yy = transformer.transform(lon, lat)
        c = int((xx - xmin_) / res)
        r = int((yy - ymin_) / res)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return None
        v = grid[r, c]
        if np.isnan(v) or v == -np.inf:
            return None
        return float(v)

    return get_elevation


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RAY–TERRAIN INTERSECTION
# ─────────────────────────────────────────────────────────────────────────────

def enu_to_lonlat(east: float, north: float,
                  origin_lon: float, origin_lat: float) -> Tuple[float, float]:
    """Convert ENU offset (meters) from origin to (lon, lat) in WGS84."""
    from pyproj import Proj
    proj = Proj(proj='aeqd', lat_0=origin_lat, lon_0=origin_lon, datum='WGS84')
    lon, lat = proj(east, north, inverse=True)
    return lat, lon


def ray_intersect_terrain(
    ray_origin_enu: np.ndarray,
    ray_dir_enu: np.ndarray,
    get_elevation: Callable[[float, float], Optional[float]],
    origin_lon: float,
    origin_lat: float,
    camera_elev_m: float,
    step_m: float = 0.5,
    max_range_m: float = 5000.0,
    tol_m: float = 0.2,
) -> Optional[Tuple[float, float, float]]:
    """
    March along the ray in ENU until the ray height drops to or below the
    terrain elevation. Terrain elevation is queried at (lon, lat) and must
    be in the same vertical datum as camera_elev_m.

    Ray origin is at the camera (0,0,0 in local ENU). So at parameter t,
    point = t * ray_dir_enu. We need terrain at (east, north) = (point[0], point[1]).
    Terrain Z in same datum as camera: elev. In our ENU (origin at camera),
    terrain height at (east, north) is (elev - camera_elev_m). When
    point[2] <= (elev - camera_elev_m) + tol, we've hit.

    Returns (lat, lon, elevation) or None if no intersection.
    """
    t = step_m
    while t <= max_range_m:
        point = ray_origin_enu + t * ray_dir_enu
        east, north, z_ray = point[0], point[1], point[2]
        lat, lon = enu_to_lonlat(east, north, origin_lon, origin_lat)
        elev = get_elevation(lon, lat)
        if elev is None:
            t += step_m
            continue
        terrain_z_enu = elev - camera_elev_m
        if z_ray <= terrain_z_enu + tol_m:
            return (lat, lon, elev)
        t += step_m
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PIXEL TO GPS USING TERRAIN
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_gps_terrain(
    pixel_uv: Tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    camera_lat: float,
    camera_lon: float,
    camera_elev_m: float,
    get_elevation: Callable[[float, float], Optional[float]],
    step_m: float = 0.5,
    max_range_m: float = 5000.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Convert image pixel (u, v) to (lat, lon, elevation) by intersecting
    the camera ray with the terrain surface.

    R: rotation matrix world ENU -> camera (same as georeference_tool).
    camera_elev_m: camera position elevation in meters (same vertical datum as terrain).
    """
    u, v = pixel_uv
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    ray_world = R.T @ ray_cam
    ray_origin_enu = np.array([0.0, 0.0, 0.0])
    return ray_intersect_terrain(
        ray_origin_enu,
        ray_world,
        get_elevation,
        camera_lon,
        camera_lat,
        camera_elev_m,
        step_m=step_m,
        max_range_m=max_range_m,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CAMERA SETUP (standalone fallback if georeference_tool missing)
# ─────────────────────────────────────────────────────────────────────────────

def _build_rotation_matrix(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Minimal copy of build_rotation_matrix for when georeference_tool is not available."""
    H, P, r = np.radians(heading_deg), np.radians(pitch_deg), np.radians(roll_deg)
    forward = np.array([
        np.sin(H) * np.cos(P),
        np.cos(H) * np.cos(P),
        np.sin(P),
    ])
    right = np.array([np.cos(H), -np.sin(H), 0.0])
    if abs(roll_deg) > 1e-6:
        cos_r, sin_r = np.cos(r), np.sin(r)
        right = cos_r * right + sin_r * np.cross(forward, right)
    down = np.cross(forward, right)
    right = right / np.linalg.norm(right)
    down = down / np.linalg.norm(down)
    R_cam_to_world = np.column_stack([right, down, forward])
    return R_cam_to_world.T


def _load_intrinsics(calib_path: str, image_w: int, image_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load K, D and scale to image size. Fallback to nominal if no georeference_tool."""
    if _HAS_GEOREF_TOOL and os.path.exists(calib_path):
        K, D, calib_img_size = load_calibrated_intrinsics(calib_path)
        if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (image_w, image_h):
            K = scale_intrinsics_for_resolution(
                K, calib_img_size[0], calib_img_size[1], image_w, image_h
            )
        return K, D
    # Nominal fallback
    fx = 2000.0 * image_w / 2592
    fy = 2000.0 * image_h / 1944
    K = np.array([[fx, 0, image_w / 2], [0, fy, image_h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    return K, D


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: INTERACTIVE CLICK-TO-GPS
# ─────────────────────────────────────────────────────────────────────────────

class TerrainGeoreferencer:
    """Interactive: click on image to get (lat, lon, elevation) from terrain."""

    def __init__(
        self,
        image: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        R: np.ndarray,
        camera_lat: float,
        camera_lon: float,
        camera_elev_m: float,
        get_elevation: Callable[[float, float], Optional[float]],
    ):
        self.orig_image = image.copy()
        self.K = K
        self.D = D
        self.R = R
        self.camera_lat = camera_lat
        self.camera_lon = camera_lon
        self.camera_elev_m = camera_elev_m
        self.get_elevation = get_elevation
        self.points = []
        self.pending = None
        if _HAS_GEOREF_TOOL:
            self.undist_image, self.K_undist = undistort(image, K, D)
        else:
            h, w = image.shape[:2]
            K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
            self.undist_image = cv2.undistort(image, K, D, None, K_new)
            x, y, cw, ch = roi
            self.undist_image = self.undist_image[y:y+ch, x:x+cw]
            # Adjust principal point for crop so (u,v) in cropped image maps correctly
            self.K_undist = K_new.copy()
            self.K_undist[0, 2] -= x
            self.K_undist[1, 2] -= y
        self.display = self.undist_image.copy()

    def _on_mouse(self, event, u, v, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        result = pixel_to_gps_terrain(
            (u, v), self.K_undist, self.R,
            self.camera_lat, self.camera_lon, self.camera_elev_m,
            self.get_elevation,
        )
        if result is None:
            print(f"[CLICK] ({u},{v}) — no terrain intersection (sky or out of bounds?)")
            return
        lat_g, lon_g, elev_g = result
        self.pending = {"u": u, "v": v, "lat": lat_g, "lon": lon_g, "elev": elev_g}
        print(f"[CLICK] ({u},{v}) → lat={lat_g:.6f}, lon={lon_g:.6f}, elev={elev_g:.2f} m")
        self._redraw()

    def _redraw(self):
        self.display = self.undist_image.copy()
        for pt in self.points:
            u, v = pt["u"], pt["v"]
            cv2.circle(self.display, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(self.display, f"{pt['lat']:.5f}, {pt['lon']:.5f} {pt['elev']:.1f}m",
                        (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if self.pending:
            u, v = self.pending["u"], self.pending["v"]
            cv2.circle(self.display, (u, v), 6, (0, 165, 255), -1)
            cv2.putText(self.display, f"Right-click to label",
                        (u + 8, v + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.putText(self.display, f"Points: {len(self.points)}  S=save  Q=quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("Terrain Georeferencer", self.display)

    def _on_rbutton(self, event, u, v, *_):
        if event == cv2.EVENT_RBUTTONDOWN and self.pending:
            label = input("Label for this point (Enter to skip): ").strip() or f"pt_{len(self.points)+1}"
            self.pending["label"] = label
            self.points.append(self.pending)
            self.pending = None
            self._redraw()

    def save_csv(self, path: str = "terrain_georeferenced_points.csv"):
        if not self.points:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "pixel_u", "pixel_v", "lat", "lon", "elev_m"])
            w.writeheader()
            for p in self.points:
                w.writerow({
                    "label": p.get("label", ""),
                    "pixel_u": p["u"], "pixel_v": p["v"],
                    "lat": p["lat"], "lon": p["lon"], "elev_m": p["elev"],
                })
        print(f"[SAVE] {len(self.points)} points → {path}")

    def run(self, output_csv: str = "terrain_georeferenced_points.csv"):
        cv2.namedWindow("Terrain Georeferencer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Terrain Georeferencer", 1200, 800)

        def mouse(event, u, v, flags, param):
            self._on_mouse(event, u, v, flags, param)
            self._on_rbutton(event, u, v, flags, param)

        cv2.setMouseCallback("Terrain Georeferencer", mouse)
        self._redraw()
        print("\n[TERRAIN] Left-click: get (lat, lon, elev). Right-click: label. S: save CSV. Q: quit.\n")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                self.save_csv(output_csv)
        cv2.destroyAllWindows()
        self.save_csv(output_csv)
        return self.points


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: COMBINED TERRAIN (DEM + LAS)
# ─────────────────────────────────────────────────────────────────────────────

def make_terrain_provider(dem_path: Optional[str] = None,
                          las_path: Optional[str] = None,
                          las_crs_epsg: Optional[int] = None,
                          las_resolution_m: float = 1.0):
    """
    Build get_elevation(lon, lat) from DEM and/or LAS.
    If both are given, DEM is used first; if DEM returns None (nodata/out of bounds),
    LAS raster is queried (if provided).
    """
    get_dem = None
    get_las = None
    if dem_path and os.path.exists(dem_path):
        get_dem, _ = load_dem_geotiff(dem_path)
        print(f"[TERRAIN] Loaded DEM: {dem_path}")
    if las_path and os.path.exists(las_path):
        get_las = load_las_terrain(las_path, resolution_m=las_resolution_m, las_crs_epsg=las_crs_epsg)
        print(f"[TERRAIN] Loaded LAS: {las_path} (resolution={las_resolution_m}m)")
    if get_dem is None and get_las is None:
        raise ValueError("Provide at least one of dem_path or las_path")

    def get_elevation(lon: float, lat: float):
        if get_dem is not None:
            v = get_dem(lon, lat)
            if v is not None:
                return v
        if get_las is not None:
            return get_las(lon, lat)
        return None

    return get_elevation


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Georeference a photo using DEM and/or LAS terrain")
    p.add_argument("image", nargs="?", default=None, help="Path to photo")
    p.add_argument("--dem", default=None, help="Path to GeoTIFF DEM")
    p.add_argument("--las", default=None, help="Path to LAS/LAZ point cloud")
    p.add_argument("--las-crs", type=int, default=None, help="EPSG for LAS (e.g. 32618)")
    p.add_argument("--las-resolution", type=float, default=1.0, help="LAS raster resolution (m)")
    p.add_argument("--calibration", "-c", default="./calibration.json", help="Calibration JSON")
    p.add_argument("--lat", type=float, default=None, help="Camera latitude (decimal degrees)")
    p.add_argument("--lon", type=float, default=None, help="Camera longitude")
    p.add_argument("--elev", type=float, default=None, help="Camera elevation (m, same datum as terrain)")
    p.add_argument("--heading", type=float, default=0.0, help="Camera heading (deg, 0=N)")
    p.add_argument("--pitch", type=float, default=0.0, help="Camera pitch (deg, negative=down)")
    p.add_argument("--roll", type=float, default=0.0, help="Camera roll (deg)")
    p.add_argument("--output-csv", "-o", default="./terrain_georeferenced_points.csv")
    args = p.parse_args()

    image_path = args.image
    if not image_path:
        image_path = input("Path to photo: ").strip()
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    dem_path = args.dem or os.environ.get("DEM_PATH")
    las_path = args.las or os.environ.get("LAS_PATH")
    if not dem_path and not las_path:
        dem_path = input("Path to GeoTIFF DEM (or Enter to skip): ").strip() or None
        las_path = input("Path to LAS/LAZ (or Enter to skip): ").strip() or None
    if not dem_path and not las_path:
        print("Need at least one of: --dem or --las (or DEM_PATH / LAS_PATH)")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)
    h, w = image.shape[:2]
    print(f"[IMAGE] {image_path} — {w}×{h}")

    get_elevation = make_terrain_provider(
        dem_path=dem_path,
        las_path=las_path,
        las_crs_epsg=args.las_crs,
        las_resolution_m=args.las_resolution,
    )

    K, D = _load_intrinsics(args.calibration, w, h)
    if _HAS_GEOREF_TOOL:
        R = build_rotation_matrix(args.heading, args.pitch, args.roll)
        if args.lat is None or args.lon is None:
            gps = read_gps_from_exif(image_path)
            args.lat = args.lat or gps.get("lat")
            args.lon = args.lon or gps.get("lon")
            if gps.get("heading") is not None:
                args.heading = gps["heading"]
    else:
        R = _build_rotation_matrix(args.heading, args.pitch, args.roll)

    if args.lat is None or args.lon is None:
        print("Set camera position: --lat and --lon (or use an image with EXIF GPS)")
        sys.exit(1)
    if args.elev is None:
        args.elev = float(input("Camera elevation in meters (same datum as terrain): "))

    tool = TerrainGeoreferencer(
        image=image,
        K=K, D=D, R=R,
        camera_lat=args.lat,
        camera_lon=args.lon,
        camera_elev_m=args.elev,
        get_elevation=get_elevation,
    )
    tool.run(output_csv=args.output_csv)
