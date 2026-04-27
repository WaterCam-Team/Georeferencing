#/usr/bin/env python3
# sudo apt install libcamera-apps  # for libcamera-jpeg
# pip install opencv-python gpsd-py3 adafruit-circuitpython-bno055 piexif pillow

import os
import json
import subprocess
import time
import datetime
import glob

import numpy as np
import cv2
from PIL import Image
import piexif
import piexif.helper
from fractions import Fraction

import gpsd  # gpsd-py3
import board
import busio
import adafruit_bno055


# -----------------------------
# Configuration
# -----------------------------

OUTPUT_DIR = "checkerboard_captures"
CALIB_RESULT_FILE = "ov5647_calibration.npz"
METADATA_JSON = "ov5647_checkerboard_metadata.json"

# Checkerboard parameters (inner corners)
CHECKERBOARD = (9, 6)          # (columns, rows)
SQUARE_SIZE_M = 0.025          # 25 mm squares

# Camera capture command (full resolution JPEG from OV5647)
# Adjust if your system uses a different resolution or tool.
CAPTURE_CMD = [
    "libcamera-jpeg",
    "-o", "",           # filename to be filled in at runtime
    "--width", "2592",
    "--height", "1944",
    "-q", "95",
    "-n"                # no preview
]


# -----------------------------
# Hardware initialization
# -----------------------------

def init_gps():
    gpsd.connect()  # connect to local gpsd
    print("Connected to gpsd")


def get_gps_fix(timeout=5.0):
    """
    Get one GPS fix from gpsd within timeout.
    Returns dict with lat, lon, alt, mode, time or None if no fix.
    """
    start = time.time()
    while time.time() - start < timeout:
        pkt = gpsd.get_current()
        if pkt.mode >= 2 and pkt.lat is not None and pkt.lon is not None:
            return {
                "lat": pkt.lat,
                "lon": pkt.lon,
                "alt": getattr(pkt, "alt", None),
                "mode": pkt.mode,
                "time": getattr(pkt, "time", None),
            }
        time.sleep(0.5)
    return None


def init_bno055():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    print("BNO055 initialized")
    return sensor


def get_orientation(sensor):
    """
    Get orientation as Euler angles (degrees) from BNO055.
    Returns dict yaw/pitch/roll or None if not ready.
    """
    # BNO055 euler: heading, roll, pitch (degrees)
    euler = sensor.euler
    if euler is None:
        return None
    heading, roll, pitch = euler
    if heading is None or roll is None or pitch is None:
        return None

    # Define yaw = heading
    return {
        "yaw_deg": float(heading),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll),
    }

def _deg_to_dms_rational(deg_float):
    """
    Convert decimal degrees to EXIF DMS rational tuples.
    """
    deg = int(deg_float)
    min_float = abs(deg_float - deg) * 60
    minute = int(min_float)
    sec_float = (min_float - minute) * 60

    def _to_frac(v):
        f = Fraction(v).limit_denominator(1000000)
        return (f.numerator, f.denominator)

    return (
        _to_frac(abs(deg)),
        _to_frac(minute),
        _to_frac(sec_float),
    )


# -----------------------------
# Capture and metadata helpers
# -----------------------------
def write_metadata_exif(image_path, meta):
    """
    Store:
      - JSON metadata in UserComment
      - GPS in standard GPS EXIF tags (if available)
    meta is the dict we assembled earlier with keys:
      "gps": {"lat", "lon", "alt", "mode", "time"} or None
      "imu": {...} or None
      "distance_m": float
      etc.
    """
    exif_dict = piexif.load(image_path)

    # ---- UserComment JSON ----
    comment_bytes = piexif.helper.UserComment.dump(
        json.dumps(meta), encoding="unicode"
    )
    if "Exif" not in exif_dict:
        exif_dict["Exif"] = {}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes

    # ---- Standard GPS tags ----
    gps = meta.get("gps")
    if gps and gps.get("lat") is not None and gps.get("lon") is not None:
        if "GPS" not in exif_dict:
            exif_dict["GPS"] = {}

        lat = gps["lat"]
        lon = gps["lon"]
        alt = gps.get("alt")

        # Latitude
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(lat)

        # Longitude
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lon)

        # Altitude (optional)
        if alt is not None:
            if alt >= 0:
                exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0
            else:
                exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 1
            exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (
                int(abs(alt) * 100),
                100,
            )

        # GPS version (required by some readers)
        exif_dict["GPS"][piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)

        # Optional: GPS time stamp and date (if gps["time"] is an ISO string or similar)
        # You can parse gps["time"] and fill GPSDateStamp and GPSTimeStamp here if needed.

    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)
    print(f"Embedded JSON + GPS EXIF into {image_path}")

def capture_image(idx):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(OUTPUT_DIR, f"calib_{idx:03d}_{timestamp}.jpg")

    cmd = CAPTURE_CMD.copy()
    cmd[cmd.index("-o") + 1] = filename

    print(f"Capturing image {filename} ...")
    subprocess.run(cmd, check=True)
    return filename


def write_usercomment_json(image_path, data_dict):
    """
    Store JSON metadata into EXIF UserComment.
    """
    exif_dict = piexif.load(image_path)
    comment_bytes = piexif.helper.UserComment.dump(
        json.dumps(data_dict), encoding="unicode"
    )
    if "Exif" not in exif_dict:
        exif_dict["Exif"] = {}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)
    print(f"Embedded metadata into {image_path}")


def detect_checkerboard(image_path):
    """
    Try to detect checkerboard corners in the image.
    Returns (found, gray_image, corners_subpix)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return False, gray, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)
    corners_subpix = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    # Optional: show detection
    vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners_subpix, True)
    cv2.imshow("Checkerboard detection", vis)
    cv2.waitKey(500)

    return True, gray, corners_subpix


# -----------------------------
# Calibration
# -----------------------------

def run_calibration(image_paths, all_imgpoints, image_shape):
    """
    Run OpenCV calibration given detected corner points.
    """
    # Prepare 3D object points for checkerboard in its local frame
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                           0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints = [objp for _ in all_imgpoints]

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, all_imgpoints, image_shape, None, None
    )

    print("Calibration RMS reprojection error:", ret)
    print("Camera matrix K:\n", K)
    print("Distortion coefficients:\n", dist.ravel())

    np.savez(CALIB_RESULT_FILE,
             K=K,
             dist=dist,
             rms_error=ret,
             image_shape=image_shape)

    print(f"Saved calibration to {CALIB_RESULT_FILE}")


# -----------------------------
# Main interactive routine
# -----------------------------

def main():
    init_gps()
    sensor = init_bno055()

    all_metadata = []
    all_imgpoints = []
    image_shape = None

    cv2.namedWindow("Checkerboard detection", cv2.WINDOW_NORMAL)

    idx = 0
    try:
        while True:
            ans = input("Capture a new checkerboard image? [y/n] ").strip().lower()
            if ans != "y":
                break

            # Ask user for distance
            dist_str = input("Enter distance from camera to checkerboard (meters): ").strip()
            try:
                distance_m = float(dist_str)
            except ValueError:
                print("Invalid distance, using NaN")
                distance_m = float("nan")

            # Get GPS
            gps_fix = get_gps_fix()
            if gps_fix is None:
                print("WARNING: No GPS fix within timeout; proceeding with None")

            # Get IMU orientation
            imu = None
            for _ in range(10):
                imu = get_orientation(sensor)
                if imu is not None:
                    break
                time.sleep(0.1)
            if imu is None:
                print("WARNING: No IMU orientation; proceeding with None")

            # Capture image
            img_path = capture_image(idx)

            # Build metadata dict
            meta = {
                "filename": os.path.basename(img_path),
                "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "distance_m": distance_m,
                "gps": gps_fix,
                "imu": imu,
            }

            # Write metadata to EXIF UserComment as JSON
            #write_usercomment_json(img_path, meta)
            write_metadata_exif(img_path, meta)

            # Detect checkerboard
            found, gray, corners_subpix = detect_checkerboard(img_path)
            if found:
                h, w = gray.shape[:2]
                image_shape = (w, h)
                all_imgpoints.append(corners_subpix)
                all_metadata.append(meta)
                print("Checkerboard detected and recorded.")
            else:
                print("WARNING: Checkerboard not detected, image will be skipped for calibration.")
                all_metadata.append(meta)

            idx += 1

    finally:
        cv2.destroyAllWindows()

    # Save metadata for all captures
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUTPUT_DIR, METADATA_JSON)
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Saved metadata for {len(all_metadata)} images to {meta_path}")

    # Run calibration if we have enough detections
    if len(all_imgpoints) < 10:
        print("Not enough valid checkerboard detections for calibration (need >= 10).")
        return

    if image_shape is None:
        print("No image shape recorded; cannot calibrate.")
        return

    run_calibration([m["filename"] for m in all_metadata if m],
                    all_imgpoints,
                    image_shape)


if __name__ == "__main__":
    main()

