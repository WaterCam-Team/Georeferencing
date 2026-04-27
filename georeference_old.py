"""
WaterCam Image Georeferencing Tool
====================================
Estimates GPS coordinates for elements in images taken with the
Dorhea/OV5647 camera used in the UFO-Net flood monitoring system.

USAGE MODES:
    1. Interactive: click points in the image to get their GPS coordinates
    2. Batch: provide a list of pixel coordinates, get GPS coords back
    3. Export: save a georeferenced GeoTIFF for use in GIS software

DEPENDENCIES:
    pip install opencv-python numpy scipy Pillow pyproj

OPTIONAL (for GeoTIFF export):
    conda install gdal   OR   pip install gdal

INPUTS REQUIRED:
    - A field image (.jpg or .png)
    - Either: a calibration.json from the calibration script
      Or:     use the built-in OV5647 nominal values (less accurate)
    - Camera mounting height above ground (meters)
    - Camera orientation: heading, pitch, roll (from IMU log or EXIF)
    - Camera GPS position (from EXIF or manual entry)
"""

import cv2
import numpy as np
import json
import os

from camera_geometry import build_rotation_matrix
import csv
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CAMERA PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# OmniVision OV5647 sensor physical specs (Dorhea 5MP camera)
# These are used if no calibration.json is available.
# For best results, replace with values from your calibration.json.
OV5647_SPECS = {
    "sensor_w_mm":   3.68,    # sensor width in mm
    "sensor_h_mm":   2.76,    # sensor height in mm
    "focal_len_mm":  3.6,     # nominal focal length in mm
    "img_w_px":      2592,    # full resolution width
    "img_h_px":      1944,    # full resolution height
}

def nominal_intrinsics_from_specs(specs: dict) -> tuple:
    """
    Derive a nominal camera matrix K from physical sensor specifications.
    This is less accurate than a calibrated K but useful as a fallback.

    The camera matrix K:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    fx, fy: focal length in pixels = focal_length_mm * (pixels / mm)
    cx, cy: principal point, assumed to be the image center
    """
    w, h    = specs["img_w_px"], specs["img_h_px"]
    f_mm    = specs["focal_len_mm"]
    sw, sh  = specs["sensor_w_mm"], specs["sensor_h_mm"]

    # Pixels per mm on the sensor
    px_per_mm_x = w / sw
    px_per_mm_y = h / sh

    fx = f_mm * px_per_mm_x   # focal length in pixels (horizontal)
    fy = f_mm * px_per_mm_y   # focal length in pixels (vertical)
    cx = w / 2.0               # principal point x (assumed center)
    cy = h / 2.0               # principal point y (assumed center)

    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)

    # No distortion correction when using nominal values
    D = np.zeros((1, 5), dtype=np.float64)

    print(f"[CAMERA] Using nominal OV5647 intrinsics:")
    print(f"  fx={fx:.1f} px, fy={fy:.1f} px")
    print(f"  cx={cx:.1f} px, cy={cy:.1f} px")
    print(f"  (Tip: run calibration script for better accuracy)")

    return K, D


def load_calibrated_intrinsics(calib_path: str) -> tuple:
    """
    Load empirically calibrated K and D from calibration.json
    produced by the calibration script.
    Preferred over nominal values — use this when available.
    """
    with open(calib_path) as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float64)
    D = np.array(d["D"], dtype=np.float64)
    print(f"[CAMERA] Loaded calibrated intrinsics from {calib_path}")
    print(f"  RMS reprojection error: {d.get('rms', 'N/A')} px")
    return K, D


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GPS / EXIF READING
# ─────────────────────────────────────────────────────────────────────────────

def read_gps_from_exif(image_path: str) -> dict:
    """
    Extract GPS position and orientation from image EXIF metadata.

    Most consumer cameras and smartphones embed:
        - Latitude / Longitude / Altitude
        - GPSImgDirection (compass heading)

    Pitch and roll are NOT standard EXIF fields. For the UFO-Net system,
    these come from the IMU log file captured alongside the image.
    See load_imu_orientation() below.

    Returns dict with keys: lat, lon, altitude, heading
    Missing fields are returned as None.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
    except ImportError:
        raise ImportError("Run: pip install Pillow")

    img     = Image.open(image_path)
    exif    = img._getexif()

    if not exif:
        print("[EXIF] No EXIF data found. You will need to provide GPS manually.")
        return {}

    decoded = {TAGS.get(k, k): v for k, v in exif.items()}
    gps_raw = decoded.get("GPSInfo", {})
    gps     = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

    def dms_to_decimal(dms, ref):
        d, m, s = [float(x) for x in dms]
        val = d + m / 60.0 + s / 3600.0
        return -val if ref in ('S', 'W') else val

    result = {}
    if "GPSLatitude" in gps:
        result["lat"] = dms_to_decimal(gps["GPSLatitude"],
                                        gps.get("GPSLatitudeRef", "N"))
    if "GPSLongitude" in gps:
        result["lon"] = dms_to_decimal(gps["GPSLongitude"],
                                        gps.get("GPSLongitudeRef", "E"))
    if "GPSAltitude" in gps:
        result["altitude"] = float(gps["GPSAltitude"])
    if "GPSImgDirection" in gps:
        result["heading"] = float(gps["GPSImgDirection"])

    print(f"[EXIF GPS] {result}")
    return result


def load_imu_orientation(imu_log_path: str, image_timestamp: str = None) -> dict:
    """
    Load pitch, roll, and yaw from the UFO-Net IMU log file.

    The IMU log is a CSV file with columns:
        timestamp, tilt_deg, roll_deg, yaw_deg, temp_c

    This matches the format described in the paper, where temperature (°C),
    tilt (°), roll (°), and yaw (°) are logged concomitant to image capture.

    If image_timestamp is provided, the nearest IMU reading is returned.
    Otherwise, the most recent reading is used.

    INSTRUCTIONS:
        Set imu_log_path to the path of your IMU CSV log file.
        Set image_timestamp to the image capture time (ISO format, e.g.
        "2024-03-15T14:32:01") to match the correct IMU reading.
    """
    readings = []
    with open(imu_log_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            readings.append(row)

    if not readings:
        raise ValueError(f"No IMU data found in {imu_log_path}")

    if image_timestamp:
        from datetime import datetime
        target = datetime.fromisoformat(image_timestamp)
        def time_diff(r):
            t = datetime.fromisoformat(r["timestamp"])
            return abs((t - target).total_seconds())
        reading = min(readings, key=time_diff)
    else:
        reading = readings[-1]  # most recent

    orientation = {
        "pitch": float(reading["tilt_deg"]),
        "roll":  float(reading["roll_deg"]),
        "yaw":   float(reading["yaw_deg"]),
    }
    print(f"[IMU] Orientation: {orientation}")
    return orientation


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ROTATION — use camera_geometry.build_rotation_matrix
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PIXEL → GPS COORDINATE
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_gps(pixel_uv: tuple,
                 K: np.ndarray,
                 R: np.ndarray,
                 camera_lat: float,
                 camera_lon: float,
                 camera_height_m: float) -> Optional[tuple]:
    """
    Convert a single image pixel (u, v) to a GPS coordinate (lat, lon)
    on the ground surface beneath.

    METHOD — Ground Plane Ray Casting:
        1. Unproject pixel through K⁻¹ → unit ray in camera space
        2. Rotate ray into world (ENU) frame using Rᵀ
        3. Solve for intersection of ray with ground plane (Z = 0 in ENU)
        4. Convert ENU offset (meters east/north) to lat/lon via
           azimuthal equidistant projection

    ASSUMPTION:
        The ground is a flat horizontal plane at Z = 0.
        This is reasonable for road surfaces monitored by UFO-Net nodes.
        For non-flat terrain, a Digital Elevation Model (DEM) would be needed.

    PARAMETERS:
        pixel_uv        : (u, v) — column, row in the undistorted image
        K               : 3×3 camera intrinsic matrix
        R               : 3×3 rotation matrix (world ENU → camera)
        camera_lat/lon  : GPS position of the camera
        camera_height_m : height of camera above ground in meters

    RETURNS:
        (lat, lon) tuple, or None if the pixel ray doesn't hit the ground
        (e.g., pixel is pointing at the sky)
    """
    try:
        from pyproj import Proj
    except ImportError:
        raise ImportError("Run: pip install pyproj")

    u, v = pixel_uv

    # ── Step 1: pixel → ray in camera frame ──────────────────────────────────
    K_inv   = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)   # unit vector

    # ── Step 2: rotate ray into world ENU frame ───────────────────────────────
    # R transforms world→camera, so R.T transforms camera→world
    ray_world = R.T @ ray_cam

    # ── Step 3: intersect ray with ground plane Z = 0 ────────────────────────
    # Camera origin in ENU: (0, 0, camera_height_m) — we set the camera
    # position as the ENU origin, so it is (0, 0, h) above the ground
    cam_origin = np.array([0.0, 0.0, camera_height_m])

    # Ray: P(λ) = cam_origin + λ * ray_world
    # Ground: Z = 0  →  cam_origin[2] + λ * ray_world[2] = 0
    if abs(ray_world[2]) < 1e-9:
        return None   # ray is horizontal — no ground intersection

    lam = -cam_origin[2] / ray_world[2]

    if lam < 0:
        return None   # intersection is behind the camera (sky pixels)

    # Ground point in ENU (meters east, north of camera)
    ground_enu = cam_origin + lam * ray_world
    east_m     = ground_enu[0]
    north_m    = ground_enu[1]

    # ── Step 4: ENU offset → lat/lon ─────────────────────────────────────────
    # Azimuthal equidistant projection: accurate for small areas (< a few km)
    proj       = Proj(proj='aeqd', lat_0=camera_lat, lon_0=camera_lon,
                      datum='WGS84')
    lon_g, lat_g = proj(east_m, north_m, inverse=True)

    return lat_g, lon_g


def compute_gsd(K: np.ndarray, img_w: int, img_h: int,
                camera_height_m: float, pitch_deg: float) -> dict:
    """
    Estimate Ground Sampling Distance (GSD) at image center.
    Follows the methodology in the paper (Eq. 14-15).
    Adjusts for camera pitch angle — a tilted camera has larger GSD
    in the far part of the image than the near part.
    """
    fx = K[0, 0]
    fy = K[1, 1]

    # Effective distance to ground at image center accounting for pitch
    # At pitch_deg=-90 (straight down), D_eff = camera_height
    # At shallower angles, the slant range is longer
    pitch_rad  = np.radians(abs(pitch_deg))
    if pitch_rad < np.radians(5):
        print("[GSD] Warning: camera nearly level, GSD will be very large")
    D_eff = camera_height_m / np.sin(pitch_rad) if pitch_rad > 0 else float('inf')

    gsd_h = D_eff / fx   # meters per pixel (horizontal)
    gsd_v = D_eff / fy   # meters per pixel (vertical)

    print(f"[GSD] Effective slant distance: {D_eff:.2f} m")
    print(f"[GSD] Horizontal: {gsd_h*100:.2f} cm/px | Vertical: {gsd_v*100:.2f} cm/px")

    return {"gsd_h_m": gsd_h, "gsd_v_m": gsd_v, "slant_dist_m": D_eff}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: UNDISTORT IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def undistort(image: np.ndarray, K: np.ndarray,
              D: np.ndarray) -> tuple:
    """
    Remove lens distortion from the image.
    MUST be applied before calling pixel_to_gps() for accurate results.

    Returns:
        undistorted image, new camera matrix K_new
        (pixel coordinates are relative to K_new after undistortion)
    """
    h, w = image.shape[:2]
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    undist      = cv2.undistort(image, K, D, None, K_new)
    x, y, cw, ch = roi
    undist      = undist[y:y+ch, x:x+cw]
    return undist, K_new


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: INTERACTIVE CLICK-TO-COORDINATE TOOL
# ─────────────────────────────────────────────────────────────────────────────

class ClickGeoreferencer:
    """
    Interactive tool: display image and click on elements to get GPS coords.

    INSTRUCTIONS:
        - Left click  : get GPS coordinate of clicked point
        - Right click : label the last clicked point
        - S key       : save all labeled points to CSV
        - U key       : toggle undistorted/original view
        - Q or ESC    : quit

    The result CSV contains: label, pixel_u, pixel_v, lat, lon, dist_from_camera_m
    """
    def __init__(self, image: np.ndarray, K: np.ndarray, D: np.ndarray,
                 R: np.ndarray, camera_lat: float, camera_lon: float,
                 camera_height_m: float, window_name: str = "WaterCam Georeferencer"):
        self.orig_image       = image.copy()
        self.K                = K
        self.D                = D
        self.R                = R
        self.cam_lat          = camera_lat
        self.cam_lon          = camera_lon
        self.height           = camera_height_m
        self.window           = window_name
        self.points           = []      # list of dicts {label, u, v, lat, lon}
        self.show_undistorted = True
        self.pending_point    = None    # last clicked, awaiting label

        # Prepare undistorted version
        self.undist_image, self.K_undist = undistort(image, K, D)
        self.display_image = self.undist_image.copy()

    def _get_active_K(self):
        return self.K_undist if self.show_undistorted else self.K

    def _get_active_image(self):
        return self.undist_image if self.show_undistorted else self.orig_image

    def _draw_overlay(self):
        """Redraw image with all labeled points and info overlay."""
        disp = self._get_active_image().copy()

        # Draw all confirmed points
        for i, pt in enumerate(self.points):
            u, v = pt["pixel_u"], pt["pixel_v"]
            cv2.circle(disp, (u, v), 6, (0, 255, 0), -1)
            label_text = f"{pt['label']} ({pt['lat']:.6f}, {pt['lon']:.6f})"
            cv2.putText(disp, label_text, (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Draw pending point (not yet labeled)
        if self.pending_point:
            u, v = self.pending_point["pixel_u"], self.pending_point["pixel_v"]
            cv2.circle(disp, (u, v), 6, (0, 165, 255), -1)
            info = (f"Lat: {self.pending_point['lat']:.7f}  "
                    f"Lon: {self.pending_point['lon']:.7f}  "
                    f"Dist: {self.pending_point['dist_m']:.1f} m")
            cv2.putText(disp, info, (u + 8, v + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            cv2.putText(disp, "Right-click to label | any key to skip",
                        (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Mode indicator
        mode = "UNDISTORTED" if self.show_undistorted else "ORIGINAL"
        cv2.putText(disp, f"[{mode}]  Points: {len(self.points)}  "
                    f"S=save  U=toggle  Q=quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(self.window, disp)

    def _on_mouse(self, event, u, v, flags, param):
        K_active = self._get_active_K()

        if event == cv2.EVENT_LBUTTONDOWN:
            result = pixel_to_gps((u, v), K_active, self.R,
                                   self.cam_lat, self.cam_lon, self.height)
            if result is None:
                print(f"[CLICK] ({u}, {v}) — no ground intersection (sky?)")
                return

            lat_g, lon_g = result

            # Approximate ground distance from camera nadir
            from pyproj import Proj
            proj     = Proj(proj='aeqd', lat_0=self.cam_lat,
                            lon_0=self.cam_lon, datum='WGS84')
            e, n     = proj(lon_g, lat_g)
            dist_m   = np.sqrt(e**2 + n**2)

            self.pending_point = {
                "label":   f"point_{len(self.points)+1}",
                "pixel_u": u,
                "pixel_v": v,
                "lat":     lat_g,
                "lon":     lon_g,
                "dist_m":  dist_m,
            }
            print(f"[CLICK] ({u}, {v}) → lat={lat_g:.7f}, lon={lon_g:.7f}, "
                  f"dist={dist_m:.1f} m from camera")
            self._draw_overlay()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pending_point:
                label = input(f"Enter label for point at ({u},{v}): ").strip()
                if label:
                    self.pending_point["label"] = label
                self.points.append(self.pending_point)
                self.pending_point = None
                self._draw_overlay()

    def save_points(self, output_csv: str = "georeferenced_points.csv"):
        if not self.points:
            print("[SAVE] No points to save.")
            return
        fieldnames = ["label", "pixel_u", "pixel_v", "lat", "lon", "dist_m"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.points)
        print(f"[SAVE] {len(self.points)} points saved to {output_csv}")

    def run(self, output_csv: str = "georeferenced_points.csv"):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1200, 800)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._draw_overlay()

        print("\n[TOOL] Click on image elements to get their GPS coordinates.")
        print("  Left click  : get coordinate")
        print("  Right click : label the last point")
        print("  S           : save to CSV")
        print("  U           : toggle undistorted/original")
        print("  Q or ESC    : quit\n")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self.save_points(output_csv)
            elif key == ord('u'):
                self.show_undistorted = not self.show_undistorted
                self._draw_overlay()

        cv2.destroyAllWindows()
        self.save_points(output_csv)
        return self.points


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: BATCH PIXEL → GPS (non-interactive)
# ─────────────────────────────────────────────────────────────────────────────

def batch_pixel_to_gps(pixels: list, K: np.ndarray, R: np.ndarray,
                        camera_lat: float, camera_lon: float,
                        camera_height_m: float) -> list:
    """
    Convert a list of pixel coordinates to GPS coordinates.

    PARAMETERS:
        pixels : list of (u, v) tuples — pixel column, row

    RETURNS:
        list of dicts: {pixel_u, pixel_v, lat, lon}
        Points with no ground intersection are skipped.

    USE CASE:
        Feed in pixel coordinates of detected flood edges or
        segmentation mask boundaries from the edge-AI classifier,
        then get their real-world GPS coordinates.
    """
    results = []
    for (u, v) in pixels:
        result = pixel_to_gps((u, v), K, R, camera_lat, camera_lon,
                               camera_height_m)
        if result:
            lat_g, lon_g = result
            results.append({"pixel_u": u, "pixel_v": v,
                             "lat": lat_g, "lon": lon_g})
        else:
            print(f"[BATCH] Skipped ({u}, {v}) — no ground intersection")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EXPORT GEOREFERENCED GEOTIFF (requires GDAL)
# ─────────────────────────────────────────────────────────────────────────────

def export_geotiff(image: np.ndarray, K: np.ndarray, R: np.ndarray,
                   camera_lat: float, camera_lon: float,
                   camera_height_m: float, output_path: str,
                   gcp_step: int = 100):
    """
    Export a GeoTIFF by computing Ground Control Points (GCPs) across
    the image and warping with GDAL's thin-plate spline method.

    INSTRUCTIONS:
        - gcp_step: pixel spacing between GCPs. 100 gives a good balance
          of accuracy vs. processing time for 2592×1944 images.
        - Output is loadable in QGIS, ArcGIS, or any GIS tool.
        - Requires GDAL: conda install gdal
    """
    try:
        from osgeo import gdal, osr
    except ImportError:
        print("[GEOTIFF] GDAL not available. Install with: conda install gdal")
        return

    h_px, w_px = image.shape[:2]
    gcps = []

    print(f"[GEOTIFF] Computing GCPs (step={gcp_step}px)...")
    for v in range(0, h_px, gcp_step):
        for u in range(0, w_px, gcp_step):
            result = pixel_to_gps((u, v), K, R, camera_lat, camera_lon,
                                   camera_height_m)
            if result:
                lat_g, lon_g = result
                gcps.append(gdal.GCP(lon_g, lat_g, 0, float(u), float(v)))

    print(f"[GEOTIFF] {len(gcps)} GCPs computed.")

    # Write image as temporary GeoTIFF
    tmp_path = output_path.replace(".tif", "_tmp.tif")
    driver   = gdal.GetDriverByName("GTiff")
    bands    = image.shape[2] if len(image.shape) == 3 else 1
    ds       = driver.Create(tmp_path, w_px, h_px, bands, gdal.GDT_Byte)

    if bands == 1:
        ds.GetRasterBand(1).WriteArray(image)
    else:
        for b in range(bands):
            ds.GetRasterBand(b + 1).WriteArray(image[:, :, b])

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetGCPs(gcps, srs.ExportToWkt())
    ds.FlushCache()
    ds = None

    # Warp using thin-plate spline for accurate reprojection
    gdal.Warp(output_path, tmp_path, format="GTiff",
              tps=True, dstSRS="EPSG:4326",
              resampleAlg="bilinear")
    os.remove(tmp_path)
    print(f"[GEOTIFF] Saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN — CONFIGURE AND RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIGURATION ─────────────────────────────────────────────────────────
    # Set these values before running.

    IMAGE_PATH = "./field_image.jpg"        # path to your field image

    # Camera intrinsics:
    # Option A — use calibrated values (recommended)
    CALIB_PATH = "./calibration.json"
    # Option B — use nominal OV5647 specs (fallback if no calibration file)

    # Camera position — from EXIF or manual entry
    # Leave as None to read from image EXIF automatically
    CAMERA_LAT  = None        # e.g. 43.0384     (decimal degrees)
    CAMERA_LON  = None        # e.g. -76.1340
    CAMERA_ALT  = None        # meters above sea level (used as height if below not set)

    # Camera orientation — from IMU log or manual measurement
    # Leave as None to attempt reading from EXIF
    IMU_LOG_PATH      = None  # path to IMU CSV log, e.g. "./imu_log.csv"
    IMAGE_TIMESTAMP   = None  # ISO timestamp, e.g. "2024-03-15T14:32:01"
    HEADING_DEG       = 45.0  # compass bearing camera faces (0=N, 90=E)
    PITCH_DEG         = -75.0 # tilt: 0=level horizon, -90=straight down
    ROLL_DEG          = 0.0   # roll: 0=level

    # Physical mounting
    CAMERA_HEIGHT_M = 5.0    # mounting height above ground in meters

    # Output
    OUTPUT_CSV     = "./georeferenced_points.csv"
    OUTPUT_GEOTIFF = "./georeferenced.tif"    # set to None to skip GeoTIFF

    # ── LOAD IMAGE ────────────────────────────────────────────────────────────
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
    print(f"[IMAGE] Loaded {IMAGE_PATH} — {image.shape[1]}×{image.shape[0]} px")

    # ── LOAD INTRINSICS ───────────────────────────────────────────────────────
    if os.path.exists(CALIB_PATH):
        K, D = load_calibrated_intrinsics(CALIB_PATH)
    else:
        print(f"[CAMERA] {CALIB_PATH} not found — using nominal OV5647 specs.")
        K, D = nominal_intrinsics_from_specs(OV5647_SPECS)

    # ── LOAD GPS ──────────────────────────────────────────────────────────────
    if CAMERA_LAT is None or CAMERA_LON is None:
        gps = read_gps_from_exif(IMAGE_PATH)
        CAMERA_LAT = gps.get("lat", CAMERA_LAT)
        CAMERA_LON = gps.get("lon", CAMERA_LON)
        CAMERA_ALT = gps.get("altitude", CAMERA_HEIGHT_M)
        if HEADING_DEG is None:
            HEADING_DEG = gps.get("heading", 0.0)

    if CAMERA_LAT is None or CAMERA_LON is None:
        raise ValueError("Camera GPS position not found in EXIF. "
                         "Set CAMERA_LAT and CAMERA_LON manually.")

    # ── LOAD IMU ORIENTATION ──────────────────────────────────────────────────
    if IMU_LOG_PATH and os.path.exists(IMU_LOG_PATH):
        imu = load_imu_orientation(IMU_LOG_PATH, IMAGE_TIMESTAMP)
        PITCH_DEG   = imu["pitch"]
        ROLL_DEG    = imu["roll"]
        HEADING_DEG = imu.get("yaw", HEADING_DEG)

    # ── BUILD ROTATION MATRIX ─────────────────────────────────────────────────
    R = build_rotation_matrix(HEADING_DEG, PITCH_DEG, ROLL_DEG)

    # ── PRINT GSD ESTIMATE ────────────────────────────────────────────────────
    compute_gsd(K, image.shape[1], image.shape[0], CAMERA_HEIGHT_M, PITCH_DEG)

    # ── RUN INTERACTIVE TOOL ──────────────────────────────────────────────────
    tool = ClickGeoreferencer(
        image=image,
        K=K, D=D, R=R,
        camera_lat=CAMERA_LAT,
        camera_lon=CAMERA_LON,
        camera_height_m=CAMERA_HEIGHT_M,
    )
    points = tool.run(output_csv=OUTPUT_CSV)

    # ── OPTIONAL: BATCH MODE EXAMPLE ─────────────────────────────────────────
    # If you have pixel coordinates from a flood segmentation mask:
    # flood_pixels = [(100, 200), (150, 300), (400, 500)]
    # _, K_undist = undistort(image, K, D)
    # results = batch_pixel_to_gps(flood_pixels, K_undist, R,
    #                               CAMERA_LAT, CAMERA_LON, CAMERA_HEIGHT_M)
    # print(results)

    # ── OPTIONAL: EXPORT GEOTIFF ──────────────────────────────────────────────
    if OUTPUT_GEOTIFF:
        undist_img, K_undist = undistort(image, K, D)
        export_geotiff(undist_img, K_undist, R,
                       CAMERA_LAT, CAMERA_LON, CAMERA_HEIGHT_M,
                       OUTPUT_GEOTIFF)
