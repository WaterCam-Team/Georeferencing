#!/usr/bin/env python3
"""
Single-image georeferencing example.

- Reads GPS from EXIF
- Reads IMU (orientation) from EXIF UserComment and XMP
- Builds a camera pose in a local ENU frame
- Projects a chosen pixel to a ground plane (z=0) and returns ENU + lat/lon

Install:
    pip install pillow piexif python-xmp-toolkit numpy pyproj
"""

import math
from dataclasses import dataclass

import numpy as np
from libxmp import XMPFiles
from libxmp.exempi import ExempiLoadError
from pyproj import CRS, Transformer

from exif_imu import read_gps_imu_from_exif

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraPoseENU:
    """Camera pose in local ENU coordinates."""
    x: float
    y: float
    z: float
    roll: float   # radians, rotation about x
    pitch: float  # radians, rotation about y
    yaw: float    # radians, rotation about z (heading)


# ----------------------------
# Helpers: EXIF GPS (uses central exif_imu)
# ----------------------------

def read_exif_gps(path):
    """Return (lat, lon, alt). Raises if lat/lon missing. alt may be None."""
    meta = read_gps_imu_from_exif(path)
    lat, lon, alt = meta.get("lat"), meta.get("lon"), meta.get("alt")
    if lat is None or lon is None:
        raise ValueError("No GPSInfo in EXIF")
    return lat, lon, alt


# ----------------------------
# Helpers: XMP IMU fallback
# ----------------------------

def _read_xmp_imu(path):
    """Read IMU from XMP (optional). Return dict or None."""
    xmpfile = None
    try:
        xmpfile = XMPFiles(file_path=path, open_forupdate=False)
        xmp = xmpfile.get_xmp()
        if xmp is None:
            return None
        imu_ns = "http://example.com/imu/1.0/"
        yaw = float(xmp.get_property(imu_ns, "yaw"))
        pitch = float(xmp.get_property(imu_ns, "pitch"))
        roll = float(xmp.get_property(imu_ns, "roll"))
        return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
    finally:
        if xmpfile is not None:
            try:
                xmpfile.close_file()
            except Exception:
                pass


def get_imu_orientation(path):
    """
    Merge IMU from EXIF (via exif_imu) and XMP (optional fallback).
    Returns yaw, pitch, roll in radians.
    """
    meta = read_gps_imu_from_exif(path)
    imu = {
        "roll_deg": meta.get("roll_deg"),
        "pitch_deg": meta.get("pitch_deg"),
        "yaw_deg": meta.get("yaw_deg"),
    }
    if not any(imu.values()):
        try:
            imu_xmp = _read_xmp_imu(path)
            if imu_xmp:
                imu.update(imu_xmp)
        except ExempiLoadError:
            pass
        except Exception:
            pass

    if imu.get("yaw_deg") is None or imu.get("pitch_deg") is None or imu.get("roll_deg") is None:
        raise ValueError("No IMU orientation found in EXIF or XMP")

    yaw = math.radians(imu["yaw_deg"])
    pitch = math.radians(imu["pitch_deg"])
    roll = math.radians(imu["roll_deg"])
    return yaw, pitch, roll


# ----------------------------
# Pose and projection
# ----------------------------

def build_local_enu(lat0, lon0):
    """
    Build ENU <-> WGS84 transformers centered at (lat0, lon0).
    """
    crs_geod = CRS.from_epsg(4979)  # WGS84 3D
    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    wgs_to_enu = Transformer.from_crs(crs_geod, crs_local, always_xy=True)
    enu_to_wgs = Transformer.from_crs(crs_local, crs_geod, always_xy=True)
    return wgs_to_enu, enu_to_wgs


def rotation_matrix_enu(roll, pitch, yaw):
    """
    Build R_enu_cam from roll, pitch, yaw (XYZ, ENU convention).
    Here we assume yaw is rotation about Z (up), pitch about Y, roll about X.
    """
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]])
    Ry = np.array([[cp, 0, sp],
                   [0,  1, 0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0,  0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    R = Rz @ Ry @ Rx
    return R


def project_pixel_to_ground(u, v, intrinsics: CameraIntrinsics,
                            pose: CameraPoseENU,
                            ground_z=0.0):
    """
    Project pixel (u, v) to a horizontal ground plane z = ground_z in ENU.

    Returns (x, y, z) in ENU.
    """
    # Camera intrinsics matrix
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1]
    ])

    # Pixel in homogeneous image coordinates
    pix = np.array([u, v, 1.0])

    # Back-project to a direction in camera frame
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ pix
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # Rotate to ENU frame
    R = rotation_matrix_enu(pose.roll, pose.pitch, pose.yaw)
    ray_enu = R @ ray_cam

    # Camera center in ENU
    cam_pos = np.array([pose.x, pose.y, pose.z])

    # Intersect ray with plane z = ground_z:
    # cam_pos + t * ray_enu has z = ground_z
    if abs(ray_enu[2]) < 1e-6:
        raise ValueError("Ray is nearly parallel to ground plane")

    t = (ground_z - cam_pos[2]) / ray_enu[2]
    if t <= 0:
        raise ValueError(
            "Intersection is behind camera (t=%g). "
            "Ray Z in ENU: %g (should be < 0 to look down); camera alt: %g; ground_z: %g. "
            "Check that the camera looks downward and ground_z is below the camera."
            % (t, float(ray_enu[2]), float(cam_pos[2]), ground_z)
        )

    pt_enu = cam_pos + t * ray_enu
    return pt_enu[0], pt_enu[1], pt_enu[2]


# ----------------------------
# Main example
# ----------------------------

def georeference_pixel(image_path, u, v,
                       fx, fy, cx, cy,
                       ground_z=0.0):
    """
    Full pipeline for a single pixel:
    - Read GPS, IMU from metadata
    - Build local ENU
    - Convert camera WGS84 to ENU
    - Project pixel to ground plane in ENU
    - Convert result back to lat/lon
    """
    # 1) GPS (camera center)
    lat, lon, alt = read_exif_gps(image_path)
    if alt is None:
        # If alt missing, set to some approximate camera height above ellipsoid
        alt = 0.0

    # 2) IMU (yaw, pitch, roll)
    yaw, pitch, roll = get_imu_orientation(image_path)

    # 3) Local ENU around camera position
    wgs_to_enu, enu_to_wgs = build_local_enu(lat, lon)

    # Camera center in ENU
    cam_x, cam_y, cam_z = wgs_to_enu.transform(lon, lat, alt)

    pose = CameraPoseENU(
        x=cam_x,
        y=cam_y,
        z=cam_z,
        roll=roll,
        pitch=pitch,
        yaw=yaw
    )

    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    # 4) Project pixel to ground plane
    x_enu, y_enu, z_enu = project_pixel_to_ground(u, v, intrinsics, pose, ground_z)

    # 5) Convert ENU back to WGS84
    lon_pt, lat_pt, alt_pt = enu_to_wgs.transform(x_enu, y_enu, z_enu)

    return {
        "camera_lat": lat,
        "camera_lon": lon,
        "camera_alt": alt,
        "target_lat": lat_pt,
        "target_lon": lon_pt,
        "target_alt": alt_pt,
    }

def params_from_npz(npz_path):
    data = np.load(npz_path)
    K = data["K"]       # 3x3 camera matrix
    dist = data["dist"] # distortion coefficients

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy

    # For a clicked pixel (u_raw, v_raw), undistort first:
    camera_matrix = K
    dist_coeffs = dist
    #return camera_matrix, dist_coeffs

    pts = np.array([[[u_raw, v_raw]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    u, v = float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])

# Then call your georeference_pixel() from the earlier script:
    result = georeference_pixel(
        image_path="your_image.jpg",
        u=u,
        v=v,
       fx=fx,
       fy=fy,
       cx=cx,
       cy=cy,
       ground_z=0.0,
    )
    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Project an image pixel to ground using EXIF GPS + IMU."
    )
    parser.add_argument("image", help="Path to JPEG image with EXIF/XMP metadata")
    parser.add_argument("--u", type=float, required=True, help="Pixel x coordinate")
    parser.add_argument("--v", type=float, required=True, help="Pixel y coordinate")
    parser.add_argument("--fx", type=float, required=True, help="Focal length fx in pixels")
    parser.add_argument("--fy", type=float, required=True, help="Focal length fy in pixels")
    parser.add_argument("--cx", type=float, required=True, help="Principal point cx in pixels")
    parser.add_argument("--cy", type=float, required=True, help="Principal point cy in pixels")
    parser.add_argument("--ground-z", type=float, default=0.0,
                        help="Ground plane height in local ENU (meters)")
    args = parser.parse_args()

    result = georeference_pixel(
        image_path=args.image,
        u=args.u,
        v=args.v,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        ground_z=args.ground_z,
    )

    print("Camera position (lat, lon, alt):", result["camera_lat"],
          result["camera_lon"], result["camera_alt"])
    print("Projected point (lat, lon, alt):", result["target_lat"],
          result["target_lon"], result["target_alt"])
