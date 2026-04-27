#!/usr/bin/env python3
"""
georeference3d.py — GPS-anchored 3D scene reconstruction and click-to-GPS tool

Takes one or more photographs with GPS + IMU EXIF metadata, reconstructs a
3D scene using Structure from Motion, aligns it to real-world GPS coordinates,
and provides an interactive tool to click on objects in photos to get their
GPS coordinates.

USAGE:
    python georeference3d.py photo1.jpg photo2.jpg [options]
    python georeference3d.py photos_dir/ [options]

OPTIONS:
    --calib calibration.json   Camera calibration file (optional but recommended)
    --height H                 Camera height above ground in metres (default 5.0)
    --output coords.csv        Output CSV file (default georef_points.csv)
    --geojson output.geojson   Output GeoJSON file (optional, for QGIS etc.)
    --downscale F              Image resize factor for SfM (default 0.5)
    --no-sfm                   Skip SfM; use flat-ground ray casting only

METADATA EXPECTED IN EACH IMAGE:
    - GPS lat/lon/alt in standard EXIF GPSInfo tags
    - IMU orientation in EXIF UserComment: "Roll R Pitch P Yaw Y" (degrees)
      (same format as SU-WaterCam / add_imu.py)

ALGORITHM:
    1. Load images, read GPS + IMU from EXIF for each photo
    2. Run incremental Structure from Motion (SIFT → FLANN → Essential matrix
       seed pair → PnP incremental registration + triangulation)
    3. Align the SfM point cloud to GPS coordinates via a similarity transform
       (Umeyama algorithm: scale + rotation + translation, ≥2 GPS cameras needed)
    4. Project geo-aligned 3D points onto each image for fast nearest-point lookup
    5. Interactive viewer: click on any image to get the GPS coordinate of that
       object.  Each click finds the nearest projected 3D point within the image;
       if no point is nearby it falls back to flat-ground ray casting using the
       camera's IMU orientation and GPS altitude from EXIF.

OUTPUT:
    - CSV: label, pixel_u, pixel_v, lat, lon, mode (3D or flat), image
    - GeoJSON FeatureCollection (optional) — load directly in QGIS / GIS tools

DEPENDENCIES:
    uv pip install opencv-python numpy pillow piexif pyproj scipy
"""

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from pyproj import Proj
from scipy.spatial import cKDTree

from camera_geometry import build_rotation_matrix
from geo_core import pixel_to_world_flat
from exif_imu import read_gps_imu_from_exif


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: EXIF / IMU METADATA READING
# ─────────────────────────────────────────────────────────────────────────────

def read_image_metadata(path):
    """
    Read GPS and IMU from image EXIF. Wrapper around exif_imu.read_gps_imu_from_exif
    with optional warnings for missing data.
    """
    meta = read_gps_imu_from_exif(path)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CAMERA INTRINSICS
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics(calib_path=None, img_w=None, img_h=None,
                    focal_mm=4.44, sensor_w_mm=4.614):
    """
    Return (K, D, calib_size).

    Priority:
      1. calibration.json  (recommended — checkerboard calibration)
      2. Nominal estimate from focal length + sensor width (Pixel 3a defaults)

    K is 3×3, D is distortion coefficients, calib_size is [w, h] or None.
    """
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            d = json.load(f)
        K = np.array(d["K"], dtype=np.float64)
        D = np.array(d["D"], dtype=np.float64)
        print(f"[CALIB] Loaded from {calib_path}  (RMS={d.get('rms', '?')} px)")
        return K, D, d.get("img_size")

    if img_w is None:
        raise ValueError("Need image size or calib file to build intrinsics")
    px_per_mm = img_w / sensor_w_mm
    fx = fy = focal_mm * px_per_mm
    K = np.array([[fx, 0, img_w / 2.0],
                  [0, fy, img_h / 2.0],
                  [0,  0,          1.0]], dtype=np.float64)
    D = np.zeros((1, 5), dtype=np.float64)
    print(f"[CALIB] Nominal intrinsics  fx={fx:.1f} px  (tip: use --calib for accuracy)")
    return K, D, None


def scale_K(K, src_w, src_h, dst_w, dst_h):
    """Scale camera matrix K from one resolution to another."""
    K_s = K.copy()
    K_s[0, 0] *= dst_w / src_w   # fx
    K_s[0, 2] *= dst_w / src_w   # cx
    K_s[1, 1] *= dst_h / src_h   # fy
    K_s[1, 2] *= dst_h / src_h   # cy
    return K_s


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: IMAGE LOADING WITH METADATA
# ─────────────────────────────────────────────────────────────────────────────

def load_images(paths, downscale=1.0, name_filter="*NIR-OFF*"):
    """
    Load images and read EXIF metadata for each.

    paths       : list of file paths or directories (searched recursively).
    name_filter : glob pattern applied to filenames found inside directories
                  (default "*NIR-OFF*").  Explicitly-named files bypass this.
    Returns list of dicts: {path, img (BGR), meta, orig_w, orig_h}.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  [WARN] Path not found: {p}  (quote paths that contain spaces)")
            continue
        if p.is_dir():
            found = sorted(
                x for x in p.rglob("*")
                if x.suffix.lower() in exts
                and fnmatch.fnmatch(x.name, name_filter)
            )
            if not found:
                print(f"  [WARN] No files matching {name_filter!r} found under {p}")
            image_paths.extend(found)
        elif p.suffix.lower() in exts:
            image_paths.append(p)
        else:
            print(f"  [WARN] Skipping {p.name} (unsupported extension)")

    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [WARN] Could not read {p.name}")
            continue
        orig_h, orig_w = img.shape[:2]
        if downscale != 1.0:
            img = cv2.resize(img, None, fx=downscale, fy=downscale,
                             interpolation=cv2.INTER_AREA)
        meta = read_image_metadata(str(p))
        images.append({
            "path": str(p),
            "img": img,
            "meta": meta,
            "orig_w": orig_w,
            "orig_h": orig_h,
        })
        lat_s = f"{meta['lat']:.5f}" if meta["lat"] else "no GPS"
        imu_s = (f"R={meta['roll_deg']:.1f} P={meta['pitch_deg']:.1f} "
                 f"Y={meta['yaw_deg']:.1f}") if meta["yaw_deg"] is not None else "no IMU"
        print(f"  {p.name}  {img.shape[1]}×{img.shape[0]}  {lat_s}  {imu_s}")

    print(f"[LOAD] {len(images)} images")
    return images


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STRUCTURE FROM MOTION (incremental)
# ─────────────────────────────────────────────────────────────────────────────

def detect_features(images):
    """SIFT keypoints + descriptors for every image."""
    sift = cv2.SIFT_create(nfeatures=4000)
    features = []
    for entry in images:
        gray = cv2.cvtColor(entry["img"], cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)
        features.append((kps, descs))
        print(f"  {Path(entry['path']).name}: {len(kps)} keypoints")
    print("[SIFT] Done")
    return features


def match_pair(descs_a, descs_b, ratio=0.75):
    """FLANN + Lowe ratio test. Returns (N, 2) array of matched indices."""
    if descs_a is None or descs_b is None:
        return np.empty((0, 2), dtype=int)
    index_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    raw = flann.knnMatch(descs_a, descs_b, k=2)
    good = [(m.queryIdx, m.trainIdx) for ml in raw
            if len(ml) == 2 for m, n in [ml] if m.distance < ratio * n.distance]
    return np.array(good, dtype=int) if good else np.empty((0, 2), dtype=int)


def find_best_seed_pair(features, n):
    """Find the pair with the most RANSAC inliers (Essential matrix test)."""
    best = (-1, -1, None, None, None, 0)
    search_range = min(n, 8)
    for i in range(n - 1):
        for j in range(i + 1, min(i + search_range, n)):
            kps_a, des_a = features[i]
            kps_b, des_b = features[j]
            matches = match_pair(des_a, des_b)
            if len(matches) < 20:
                continue
            pts_a = np.float32([kps_a[m].pt for m in matches[:, 0]])
            pts_b = np.float32([kps_b[m].pt for m in matches[:, 1]])
            E, mask = cv2.findEssentialMat(
                pts_a, pts_b, focal=1.0, pp=(0, 0),
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if mask is None:
                continue
            n_in = int(mask.sum())
            if n_in > best[5]:
                best = (i, j, matches, E, mask, n_in)
    return best[:5]


def init_reconstruction(i, j, features, K):
    """
    Bootstrap reconstruction from image pair (i, j).
    Camera i is placed at the world origin.
    Returns (poses, pts3d, obs, kp_to_pt).
    """
    kps_a, des_a = features[i]
    kps_b, des_b = features[j]
    matches = match_pair(des_a, des_b)
    if len(matches) < 20:
        raise RuntimeError(f"Too few matches between images {i} and {j}")

    pts_a = np.float64([kps_a[m].pt for m in matches[:, 0]])
    pts_b = np.float64([kps_b[m].pt for m in matches[:, 1]])

    pts_an = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K, np.zeros(5)).reshape(-1, 2)
    pts_bn = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K, np.zeros(5)).reshape(-1, 2)

    E, mask_e = cv2.findEssentialMat(pts_an, pts_bn, np.eye(3),
                                      method=cv2.RANSAC, prob=0.999, threshold=1e-3)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    mask_e = mask_e.ravel().astype(bool)
    _, R, t, mask_p = cv2.recoverPose(E, pts_an[mask_e], pts_bn[mask_e], np.eye(3))
    mask_p = mask_p.ravel().astype(bool)

    inlier_idx = np.where(mask_e)[0][mask_p]
    pts_a_in   = pts_a[inlier_idx]
    pts_b_in   = pts_b[inlier_idx]
    match_in   = matches[inlier_idx]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P1, P2, pts_a_in.T, pts_b_in.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    t2 = t.ravel()
    z1 = pts3d[:, 2]
    z2 = (R @ pts3d.T + t2[:, None])[2]
    valid = (z1 > 0) & (z2 > 0)
    pts3d    = pts3d[valid]
    match_in = match_in[valid]

    print(f"[INIT] Pair ({i},{j}): {len(pts3d)} 3D points")

    poses    = {i: (np.eye(3), np.zeros(3)), j: (R, t2)}
    obs      = {i: {}, j: {}}
    kp_to_pt = {i: {}, j: {}}
    for pt_idx, (ka, kb) in enumerate(match_in[:, :2]):
        obs[i][pt_idx] = int(ka);  kp_to_pt[i][int(ka)] = pt_idx
        obs[j][pt_idx] = int(kb);  kp_to_pt[j][int(kb)] = pt_idx
    return poses, pts3d, obs, kp_to_pt


def add_image(new_idx, features, poses, pts3d, obs, kp_to_pt, K, reproj_thresh=4.0):
    """
    Register a new image via PnP + triangulate new points.
    Returns updated (pts3d, obs, kp_to_pt), or None if registration fails.
    """
    kps_new, des_new = features[new_idx]
    if des_new is None:
        return None

    # 2D↔3D correspondences from all registered images
    pts3d_corr, pts2d_corr = [], []
    for reg_idx in poses:
        kps_reg, des_reg = features[reg_idx]
        matches = match_pair(des_reg, des_new)
        if len(matches) < 8:
            continue
        for m_reg, m_new in matches:
            pt3d_idx = kp_to_pt.get(reg_idx, {}).get(int(m_reg))
            if pt3d_idx is not None and pt3d_idx < len(pts3d):
                pts3d_corr.append(pts3d[pt3d_idx])
                pts2d_corr.append(kps_new[m_new].pt)

    if len(pts3d_corr) < 6:
        print(f"  [SKIP] Image {new_idx}: {len(pts3d_corr)} correspondences")
        return None

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.float64(pts3d_corr), np.float64(pts2d_corr), K, np.zeros(5),
        iterationsCount=1000, reprojectionError=reproj_thresh,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None or len(inliers) < 6:
        print(f"  [SKIP] Image {new_idx}: PnP failed")
        return None

    R_new, _ = cv2.Rodrigues(rvec)
    t_new    = tvec.ravel()
    poses[new_idx] = (R_new, t_new)
    print(f"  [PnP] Image {new_idx}: registered ({len(inliers)} inliers)")

    obs.setdefault(new_idx, {})
    kp_to_pt.setdefault(new_idx, {})
    new_pts = 0
    P_new = K @ np.hstack([R_new, t_new[:, None]])

    def reproj_err(P, p3, p2):
        ph = P @ np.vstack([p3.T, np.ones(len(p3))])
        return np.linalg.norm((ph[:2] / ph[2]).T - p2, axis=1)

    for reg_idx in list(poses.keys()):
        if reg_idx == new_idx:
            continue
        R_reg, t_reg = poses[reg_idx]
        P_reg = K @ np.hstack([R_reg, t_reg[:, None]])
        kps_reg, des_reg = features[reg_idx]
        matches = match_pair(des_reg, des_new)
        if len(matches) < 8:
            continue

        pts_reg, pts_new_, midx = [], [], []
        for m_reg, m_new in matches:
            if (int(m_reg) not in kp_to_pt.get(reg_idx, {}) and
                    int(m_new) not in kp_to_pt.get(new_idx, {})):
                pts_reg.append(kps_reg[m_reg].pt)
                pts_new_.append(kps_new[m_new].pt)
                midx.append((int(m_reg), int(m_new)))
        if len(pts_reg) < 4:
            continue

        pa = np.float64(pts_reg)
        pb = np.float64(pts_new_)
        pts4d = cv2.triangulatePoints(P_reg, P_new, pa.T, pb.T)
        new3d = (pts4d[:3] / pts4d[3]).T

        z_reg = (R_reg @ new3d.T + t_reg[:, None])[2]
        z_new = (R_new @ new3d.T + t_new[:, None])[2]
        valid = ((z_reg > 0) & (z_new > 0) &
                 (reproj_err(P_reg, new3d, pa) < reproj_thresh) &
                 (reproj_err(P_new, new3d, pb) < reproj_thresh))

        for vi, (good, (kr, kn)) in enumerate(zip(valid, midx)):
            if not good:
                continue
            pt_idx = len(pts3d)
            pts3d = np.vstack([pts3d, new3d[vi]])
            obs[reg_idx][pt_idx] = kr;  kp_to_pt[reg_idx][kr] = pt_idx
            obs[new_idx][pt_idx]  = kn;  kp_to_pt[new_idx][kn]  = pt_idx
            new_pts += 1

    print(f"         +{new_pts} new points (total {len(pts3d)})")
    return pts3d, obs, kp_to_pt


def run_sfm(images, features, K):
    """
    Full incremental SfM pipeline.
    Returns (poses, pts3d, obs, kp_to_pt).
    poses: {img_idx: (R, t)} — world→camera transforms in SfM space.
    pts3d: (N, 3) float64 — 3D points in SfM coordinate frame.
    """
    n = len(images)
    si, sj, _, _, _ = find_best_seed_pair(features, n)
    if si < 0:
        raise RuntimeError("Could not find a valid seed pair — "
                           "check that images overlap sufficiently")
    print(f"[SEED] Best pair: {si} ({Path(images[si]['path']).name}) "
          f"and {sj} ({Path(images[sj]['path']).name})")
    poses, pts3d, obs, kp_to_pt = init_reconstruction(si, sj, features, K)

    for idx in [i for i in range(n) if i not in poses]:
        result = add_image(idx, features, poses, pts3d, obs, kp_to_pt, K)
        if result is not None:
            pts3d, obs, kp_to_pt = result

    print(f"[SFM] {len(poses)}/{n} cameras registered, {len(pts3d)} 3D points")
    return poses, pts3d, obs, kp_to_pt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: GPS ↔ LOCAL ENU FRAME
# ─────────────────────────────────────────────────────────────────────────────

def gps_to_enu(lats, lons, alts, origin_lat, origin_lon, origin_alt=0.0):
    """
    Convert GPS coordinates to local East-North-Up in metres.
    Uses an azimuthal equidistant projection centred on origin (accurate < ~10 km).
    Returns (N, 3) float64 array.
    """
    proj = Proj(proj="aeqd", lat_0=origin_lat, lon_0=origin_lon, datum="WGS84")
    pts = []
    for lat, lon, alt in zip(lats, lons, alts):
        e, n = proj(lon, lat)
        u = (alt or 0.0) - (origin_alt or 0.0)
        pts.append([e, n, u])
    return np.array(pts, dtype=np.float64)


def enu_to_gps(enu_pts, origin_lat, origin_lon, origin_alt=0.0):
    """
    Convert local ENU metres back to GPS (lat, lon, alt).
    Returns list of (lat, lon, alt) tuples.
    """
    proj = Proj(proj="aeqd", lat_0=origin_lat, lon_0=origin_lon, datum="WGS84")
    results = []
    for e, n, u in enu_pts:
        lon, lat = proj(e, n, inverse=True)
        results.append((lat, lon, u + origin_alt))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: SIM3 GEO-ALIGNMENT (Umeyama similarity transform)
# ─────────────────────────────────────────────────────────────────────────────

def umeyama(src, dst):
    """
    Find similarity transform (s, R, t) so that dst ≈ s * R @ src + t.

    src, dst: (N, 3) arrays of corresponding 3D points.
    Minimises sum of squared distances (least squares).
    Reference: Umeyama, PAMI 1991.
    Returns (s, R, t).
    """
    n = len(src)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst

    var_src = (src_c ** 2).sum() / n
    cov = dst_c.T @ src_c / n

    U, S_sv, Vt = np.linalg.svd(cov)
    det_sign = np.linalg.det(U @ Vt)
    D = np.diag([1.0] * 2 + [round(det_sign)])   # handles reflection
    R = U @ D @ Vt

    s = float((S_sv * D.diagonal()).sum() / var_src)
    t = mu_dst - s * R @ mu_src
    return s, R, t


def align_to_gps(poses, pts3d, images):
    """
    Compute a Sim3 that maps the SfM world frame to local ENU (GPS-anchored).

    Uses the GPS positions of all registered cameras as ground-truth anchors.
    Requires GPS on at least 2 registered cameras.

    Returns (s, R_sim3, t_sim3, origin_lat, origin_lon, origin_alt).
    """
    reg_with_gps = [
        idx for idx in sorted(poses.keys())
        if images[idx]["meta"]["lat"] is not None
        and images[idx]["meta"]["lon"] is not None
    ]

    if len(reg_with_gps) < 2:
        raise RuntimeError(
            f"GPS found on only {len(reg_with_gps)} registered camera(s). "
            "Need at least 2 for geo-alignment."
        )

    lats = [images[i]["meta"]["lat"]        for i in reg_with_gps]
    lons = [images[i]["meta"]["lon"]        for i in reg_with_gps]
    alts = [images[i]["meta"]["alt"] or 0.0 for i in reg_with_gps]

    # ENU origin = centroid of GPS camera positions
    origin_lat = float(np.mean(lats))
    origin_lon = float(np.mean(lons))
    origin_alt = float(np.mean(alts))

    gps_enu = gps_to_enu(lats, lons, alts, origin_lat, origin_lon, origin_alt)

    # SfM camera centres:  C = -R.T @ t
    sfm_centers = np.array([-poses[i][0].T @ poses[i][1] for i in reg_with_gps])

    if len(reg_with_gps) == 2:
        # Umeyama is degenerate with 2 points; use Rodrigues rotation instead.
        sfm_dist = np.linalg.norm(sfm_centers[1] - sfm_centers[0])
        gps_dist = np.linalg.norm(gps_enu[1] - gps_enu[0])
        if sfm_dist < 1e-9:
            raise RuntimeError("SfM baseline is near-zero; cannot determine scale")
        s = gps_dist / sfm_dist

        v_sfm = (sfm_centers[1] - sfm_centers[0]) / sfm_dist
        v_gps = (gps_enu[1] - gps_enu[0]) / gps_dist
        axis  = np.cross(v_sfm, v_gps)
        ax_n  = np.linalg.norm(axis)
        if ax_n > 1e-9:
            axis  /= ax_n
            angle  = np.arccos(np.clip(np.dot(v_sfm, v_gps), -1.0, 1.0))
            K_skew = np.array([[0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0]])
            R_sim3 = (np.eye(3) + np.sin(angle) * K_skew
                      + (1 - np.cos(angle)) * K_skew @ K_skew)
        else:
            R_sim3 = np.eye(3)
        t_sim3 = gps_enu[0] - s * R_sim3 @ sfm_centers[0]
        print("[ALIGN] 2-camera alignment (rotation around baseline may be ambiguous)")
    else:
        s, R_sim3, t_sim3 = umeyama(sfm_centers, gps_enu)

    # Report alignment residuals
    aligned = np.array([s * R_sim3 @ sfm_centers[k] + t_sim3
                        for k in range(len(reg_with_gps))])
    errors = np.linalg.norm(aligned - gps_enu, axis=1)
    print(f"[ALIGN] {len(reg_with_gps)} cameras  "
          f"mean error={errors.mean():.2f} m  max={errors.max():.2f} m")

    return s, R_sim3, t_sim3, origin_lat, origin_lon, origin_alt


def sfm_to_gps(pts3d_sfm, s, R_sim3, t_sim3, origin_lat, origin_lon, origin_alt):
    """
    Transform SfM 3D points → GPS (lat, lon, alt) via Sim3 + ENU→GPS.
    Returns list of (lat, lon, alt) with same length as pts3d_sfm.
    """
    pts_enu = (s * R_sim3 @ pts3d_sfm.T).T + t_sim3
    return enu_to_gps(pts_enu, origin_lat, origin_lon, origin_alt)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FLAT-GROUND FALLBACK (single image or pixels with no nearby 3D pt)
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_gps_flat(pixel_uv, K, R, cam_lat, cam_lon, cam_height_m):
    """
    Ray-cast a pixel onto the flat ground plane (Z=0 in ENU).
    Returns (lat, lon) or None if the ray doesn't intersect the ground.
    Uses geo_core for a single, tested implementation.
    """
    u, v = pixel_uv
    return pixel_to_world_flat(u, v, K, R, cam_lat, cam_lon, cam_height_m)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: RAY–TERRAIN INTERSECTION (DEM)
# ─────────────────────────────────────────────────────────────────────────────

def load_dem(paths):
    """
    Load one or more GeoTIFF DEM files and return a combined
    get_elevation(lon, lat) -> float | None callable.

    Multiple tiles are queried in order; the first non-None result is returned.
    Requires rasterio: uv pip install rasterio
    """
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        raise ImportError("DEM support requires rasterio: uv pip install rasterio")

    sources = []
    for path in paths:
        src = rasterio.open(path)
        crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
        transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        nodata = src.nodata
        sources.append((src, transformer, nodata))
        print(f"[DEM] Loaded {Path(path).name}  "
              f"({src.width}×{src.height} px, res={src.res[0]:.2f} m, crs={crs_str})")

    def get_elevation(lon, lat):
        for src, transformer, nodata in sources:
            x, y = transformer.transform(lon, lat)
            # bounds check before indexing
            left, bottom, right, top = src.bounds
            if not (left <= x <= right and bottom <= y <= top):
                continue
            row, col = src.index(x, y)
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                continue
            window = rasterio.windows.Window(col, row, 1, 1)
            val = src.read(1, window=window).flat[0]
            if nodata is not None and val == nodata:
                continue
            if np.isnan(val):
                continue
            return float(val)
        return None

    return get_elevation


def ray_intersect_terrain(ray_dir_enu, cam_lat, cam_lon, cam_elev_m,
                           get_elevation, step_m=1.0, max_range_m=500.0):
    """
    March a unit ray in ENU from the camera until it hits the terrain, then
    refine the intersection with bisection to sub-step accuracy.

    cam_elev_m must be in the SAME vertical datum as the DEM (see
    pixel_to_gps_terrain which handles the WGS84→DEM datum correction).

    Returns (lat, lon, elev_m, dist_m) or None.
    """
    proj = Proj(proj="aeqd", lat_0=cam_lat, lon_0=cam_lon, datum="WGS84")

    def sample(t):
        east  = ray_dir_enu[0] * t
        north = ray_dir_enu[1] * t
        z_ray = ray_dir_enu[2] * t
        lon, lat = proj(east, north, inverse=True)
        elev = get_elevation(lon, lat)
        if elev is None:
            return None, east, north, z_ray, lon, lat
        return z_ray - (elev - cam_elev_m), east, north, z_ray, lon, lat

    # Coarse march
    t_prev = step_m
    gap_prev, *_ = sample(t_prev)
    t = step_m * 2
    while t <= max_range_m:
        gap, east, north, z_ray, lon, lat = sample(t)
        if gap is None:
            t += step_m
            t_prev = t
            gap_prev = None
            continue
        if gap_prev is not None and gap <= 0.0:
            # Crossed terrain — bisect between t_prev and t for sub-step accuracy
            lo, hi = t_prev, t
            for _ in range(8):   # 8 steps → step/256 ≈ 4 mm accuracy
                mid = (lo + hi) / 2.0
                g_mid, e_m, n_m, z_m, lon_m, lat_m = sample(mid)
                if g_mid is None:
                    break
                if g_mid <= 0.0:
                    hi = mid
                else:
                    lo = mid
            # Use the refined hi sample as the intersection point
            _, east, north, z_ray, lon, lat = sample(hi)
            elev = get_elevation(lon, lat)
            if elev is None:
                elev = cam_elev_m + z_ray
            dist_m = float(np.sqrt(east**2 + north**2 + z_ray**2))
            return lat, lon, elev, dist_m
        gap_prev = gap
        t_prev = t
        t += step_m
    return None


def pixel_to_gps_terrain(pixel_uv, K, R, cam_lat, cam_lon, cam_elev_gps_m,
                          get_elevation, cam_height_m=5.0,
                          step_m=1.0, max_range_m=500.0):
    """
    Ray-cast a pixel onto the DEM surface.

    Corrects for the WGS84 ellipsoid / DEM orthometric datum mismatch by
    looking up the DEM elevation directly beneath the camera and using
    (DEM ground elevation + cam_height_m) as the effective camera elevation.
    This avoids the ~25–35 m datum offset that causes large distance errors
    when GPS altitude is used directly with a NAVD88 DEM.

    Returns (lat, lon, elev_m, dist_m) or None.
    """
    u, v = pixel_uv
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = R.T @ ray_cam   # camera frame → ENU

    if ray_world[2] > 0:
        return None   # pointing up — no terrain intersection

    # Datum correction: replace GPS ellipsoidal altitude with
    # DEM orthometric elevation at camera footprint + mounting height.
    ground_elev = get_elevation(cam_lon, cam_lat)
    if ground_elev is not None:
        cam_elev_m = ground_elev + cam_height_m
    else:
        cam_elev_m = cam_elev_gps_m

    return ray_intersect_terrain(ray_world, cam_lat, cam_lon, cam_elev_m,
                                  get_elevation, step_m, max_range_m)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: INTERACTIVE CLICK-TO-GPS VIEWER
# ─────────────────────────────────────────────────────────────────────────────

class GeoReferenceTool3D:
    """
    Interactive click-to-GPS tool backed by a geo-aligned 3D point cloud.

    Displays one image at a time. The user clicks on any object to get its
    GPS coordinate. For each click:
      1. The nearest geo-aligned 3D point projected onto that image is found
         (within a configurable pixel radius).
      2. If no nearby 3D point exists, falls back to flat-ground ray casting
         using the camera's IMU and GPS from EXIF.

    The reconstructed 3D points are drawn as small coloured dots on the image
    so the user can see where reliable 3D coordinates are available.

    CONTROLS:
        Left click  — get GPS coordinate of clicked point
        Right click — label the last clicked point (prompts for name)
        N           — next image
        P           — previous image
        S           — save points to CSV (and GeoJSON if --geojson set)
        Q / ESC     — quit (auto-saves on exit)
    """

    # Max pixel distance to a projected 3D point for "3D mode" lookup.
    # Smaller = less risk of returning the wrong object's coordinates.
    # Larger = more clicks resolved via accurate 3D mode vs flat/terrain fallback.
    SNAP_RADIUS_PX = 30

    def __init__(self, images, features, poses, pts3d_sfm, pts3d_gps,
                 obs, K, camera_height_m,
                 get_elevation=None,
                 output_csv="georef_points.csv", output_geojson=None):
        self.images         = images
        self.features       = features
        self.poses          = poses
        self.pts3d_sfm      = pts3d_sfm   # (N, 3) in SfM frame
        self.pts3d_gps      = pts3d_gps   # list of (lat, lon, alt), len N; or []
        self.obs            = obs
        self.K              = K
        self.height         = camera_height_m
        self.get_elevation  = get_elevation   # DEM callable, or None
        self.output_csv     = output_csv
        self.output_geojson = output_geojson

        self.img_idx     = 0
        self.saved_pts   = []
        self.pending     = None
        self.window      = "Georeference 3D"
        self._datum_logged = set()   # images for which datum correction was printed

        # Per-image: KD-tree on 2D projections, mapping tree row → pt3d index
        self._trees    = {}   # img_idx -> cKDTree
        self._tree_ids = {}   # img_idx -> np.array of pt3d indices

        if len(self.pts3d_sfm) > 0 and len(self.pts3d_gps) > 0:
            self._build_projection_trees()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_projection_trees(self):
        """
        Project all SfM 3D points into each registered image.
        Builds a 2D KD-tree per image for O(log N) nearest-point lookup on click.
        """
        n_pts = len(self.pts3d_sfm)
        for img_idx, (R, t) in self.poses.items():
            img_h, img_w = self.images[img_idx]["img"].shape[:2]

            # All 3D points visible in this image according to obs
            visible_ids = np.array(
                [pt_idx for pt_idx in self.obs.get(img_idx, {}).keys()
                 if pt_idx < n_pts],
                dtype=int,
            )
            if len(visible_ids) == 0:
                continue

            pts = self.pts3d_sfm[visible_ids]   # (M, 3)

            # Project: camera coords = R @ pt + t
            pts_cam = (R @ pts.T).T + t          # (M, 3)
            in_front = pts_cam[:, 2] > 0

            # 2D pixel projection
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            u = pts_cam[:, 0] / pts_cam[:, 2] * fx + cx
            v = pts_cam[:, 1] / pts_cam[:, 2] * fy + cy

            in_image = (in_front &
                        (u >= 0) & (u < img_w) &
                        (v >= 0) & (v < img_h))

            u_ok = u[in_image]
            v_ok = v[in_image]
            ids_ok = visible_ids[in_image]

            if len(ids_ok) == 0:
                continue

            pts_2d = np.column_stack([u_ok, v_ok])
            self._trees[img_idx]    = cKDTree(pts_2d)
            self._tree_ids[img_idx] = ids_ok

    # ── GPS lookup ────────────────────────────────────────────────────────────

    def _query_gps(self, img_idx, u, v):
        """
        Return (lat, lon, dist_m, mode_str) for a click at (u, v) in image img_idx.

        dist_m is the straight-line distance from the camera to the point.
        mode is "3D" if we used a geo-aligned 3D point, "flat" otherwise.
        Returns (None, None, None, None) if no coordinate could be computed.
        """
        meta = self.images[img_idx]["meta"]

        # Try nearest 3D point
        tree = self._trees.get(img_idx)
        if tree is not None:
            dist, row = tree.query([u, v])
            if dist <= self.SNAP_RADIUS_PX:
                pt_idx = self._tree_ids[img_idx][row]
                if pt_idx < len(self.pts3d_gps):
                    lat, lon, pt_alt = self.pts3d_gps[pt_idx]
                    dist_m = None
                    if meta["lat"] is not None:
                        proj = Proj(proj="aeqd", lat_0=meta["lat"],
                                    lon_0=meta["lon"], datum="WGS84")
                        e, n = proj(lon, lat)
                        dz = pt_alt - (meta["alt"] or 0.0)
                        dist_m = float(np.sqrt(e**2 + n**2 + dz**2))
                    return lat, lon, dist_m, "3D"

        # Terrain and flat fallbacks both need IMU + GPS in EXIF
        if (meta["lat"] is not None and
                meta["yaw_deg"] is not None and
                meta["pitch_deg"] is not None):
            R_imu = build_rotation_matrix(
                meta["yaw_deg"],
                meta["pitch_deg"],
                meta["roll_deg"] or 0.0,
            )

            # Ray–DEM intersection (preferred over flat ground)
            if self.get_elevation is not None and meta["alt"] is not None:
                if img_idx not in self._datum_logged:
                    ground_elev = self.get_elevation(meta["lon"], meta["lat"])
                    if ground_elev is not None:
                        corrected = ground_elev + self.height
                        print(f"[DEM] {Path(self.images[img_idx]['path']).name}: "
                              f"GPS alt={meta['alt']:.1f} m (WGS84), "
                              f"DEM ground={ground_elev:.1f} m, "
                              f"cam elev={corrected:.1f} m (datum-corrected, "
                              f"offset={meta['alt']-corrected:+.1f} m)")
                    else:
                        print(f"[DEM] {Path(self.images[img_idx]['path']).name}: "
                              "no DEM coverage at camera position, using GPS alt")
                    self._datum_logged.add(img_idx)
                result = pixel_to_gps_terrain(
                    (u, v), self.K, R_imu,
                    meta["lat"], meta["lon"], meta["alt"],
                    self.get_elevation,
                    cam_height_m=self.height,
                )
                if result is not None:
                    lat, lon, _elev, dist_m = result
                    return lat, lon, dist_m, "terrain"

            # Flat-ground fallback
            result = pixel_to_gps_flat(
                (u, v), self.K, R_imu,
                meta["lat"], meta["lon"], self.height,
            )
            if result:
                lat, lon = result
                proj = Proj(proj="aeqd", lat_0=meta["lat"],
                            lon_0=meta["lon"], datum="WGS84")
                e, n = proj(lon, lat)
                dist_m = float(np.sqrt(e**2 + n**2 + self.height**2))
                return lat, lon, dist_m, "flat"

        return None, None, None, None

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_overlay(self):
        entry = self.images[self.img_idx]
        disp  = entry["img"].copy()
        img_h, img_w = disp.shape[:2]
        name  = Path(entry["path"]).name
        is_reg = self.img_idx in self.poses

        # Draw projected 3D points (grey dots so user knows coverage)
        tree = self._trees.get(self.img_idx)
        if tree is not None and len(tree.data) > 0:
            for u, v in tree.data:
                cv2.circle(disp, (int(u), int(v)), 3, (180, 180, 180), -1)

        # Draw saved points for this image
        for pt in self.saved_pts:
            if pt.get("img_idx") != self.img_idx:
                continue
            u, v = pt["pixel_u"], pt["pixel_v"]
            cv2.circle(disp, (u, v), 7, (0, 220, 0), -1)
            cv2.putText(disp,
                        f"{pt['label']} ({pt['lat']:.6f}, {pt['lon']:.6f})",
                        (u + 9, v - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        # Draw pending point
        if self.pending and self.pending.get("img_idx") == self.img_idx:
            u, v = self.pending["pixel_u"], self.pending["pixel_v"]
            mode = self.pending.get("mode", "?")
            color = (0, 140, 255) if mode == "3D" else (255, 180, 0)
            cv2.circle(disp, (u, v), 7, color, -1)
            dist_m = self.pending.get("dist_m")
            dist_s = f"  dist={dist_m:.1f} m" if dist_m is not None else ""
            info = (f"Lat: {self.pending['lat']:.7f}  "
                    f"Lon: {self.pending['lon']:.7f}{dist_s}  [{mode}]")
            cv2.putText(disp, info, (u + 9, v + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(disp, "Right-click to label | any key to skip",
                        (10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status bar
        reg_s = "REGISTERED" if is_reg else "no SfM"
        n3d   = len(self._trees.get(self.img_idx, {}).data) if self.img_idx in self._trees else 0
        cv2.putText(disp,
                    f"{name}  [{self.img_idx+1}/{len(self.images)}]  "
                    f"{reg_s}  {n3d} 3D pts  "
                    f"N=next P=prev S=save Q=quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window, disp)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            lat, lon, dist_m, mode = self._query_gps(self.img_idx, u, v)
            if lat is None:
                print(f"[CLICK] ({u},{v}) — no GPS result "
                      "(no nearby 3D point and no IMU/GPS in EXIF)")
                return
            dist_s = f"  dist={dist_m:.1f} m" if dist_m is not None else ""
            print(f"[CLICK] ({u},{v}) → lat={lat:.7f}  lon={lon:.7f}{dist_s}  [{mode}]")
            self.pending = {
                "img_idx": self.img_idx,
                "pixel_u": u, "pixel_v": v,
                "lat": lat, "lon": lon,
                "dist_m": dist_m,
                "mode": mode,
                "label": f"point_{len(self.saved_pts) + 1}",
                "image": Path(self.images[self.img_idx]["path"]).name,
            }
            self._draw_overlay()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pending:
                label = input(f"Label for ({u},{v}): ").strip()
                if label:
                    self.pending["label"] = label
                self.saved_pts.append(self.pending)
                self.pending = None
                self._draw_overlay()

    # ── Save / export ─────────────────────────────────────────────────────────

    def save_points(self):
        if not self.saved_pts:
            print("[SAVE] No points to save.")
            return

        fields = ["label", "pixel_u", "pixel_v", "lat", "lon", "dist_m", "mode", "image"]
        with open(self.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.saved_pts)
        print(f"[SAVE] {len(self.saved_pts)} points → {self.output_csv}")

        if self.output_geojson:
            features_gj = [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [pt["lon"], pt["lat"]],
                    },
                    "properties": {
                        k: pt[k]
                        for k in ("label", "pixel_u", "pixel_v", "dist_m", "mode", "image")
                    },
                }
                for pt in self.saved_pts
            ]
            gj = {"type": "FeatureCollection", "features": features_gj}
            with open(self.output_geojson, "w") as f:
                json.dump(gj, f, indent=2)
            print(f"[SAVE] GeoJSON → {self.output_geojson}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 800)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._draw_overlay()

        print("\n[TOOL] Click on objects to get their GPS coordinates.")
        print("  Left click  : get GPS coordinate")
        print("  Right click : label the last point (save to list)")
        print("  N / P       : next / previous image")
        print("  S           : save CSV (+ GeoJSON if --geojson)")
        print("  Q / ESC     : quit (auto-saves)\n")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                self.img_idx = (self.img_idx + 1) % len(self.images)
                self._draw_overlay()
            elif key == ord("p"):
                self.img_idx = (self.img_idx - 1) % len(self.images)
                self._draw_overlay()
            elif key == ord("s"):
                self.save_points()

        cv2.destroyAllWindows()
        self.save_points()
        return self.saved_pts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPS-anchored 3D georeferencing from photos with EXIF GPS+IMU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Image files or a directory of images")
    parser.add_argument("--calib",     default=None,
                        help="calibration.json (from camera_calibration.py)")
    parser.add_argument("--height",    type=float, default=5.0,
                        help="Camera height above ground in metres (default 5.0)")
    parser.add_argument("--output",    default="georef_points.csv",
                        help="Output CSV (default georef_points.csv)")
    parser.add_argument("--geojson",   default=None,
                        help="Output GeoJSON file (optional)")
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Image downscale for SfM processing (default 0.5)")
    parser.add_argument("--filter",    default="*NIR-OFF*",
                        help="Filename glob filter for images found inside "
                             "directories (default '*NIR-OFF*'). "
                             "Use '*' to include all image files.")
    parser.add_argument("--dem",       nargs="+", default=None, metavar="DEM",
                        help="One or more GeoTIFF DEM files for ray-terrain "
                             "intersection (more accurate than flat-ground). "
                             "Multiple tiles are automatically merged.")
    parser.add_argument("--no-sfm",    action="store_true",
                        help="Skip SfM; use flat-ground ray casting only")
    args = parser.parse_args()

    # ── Load DEM (optional) ───────────────────────────────────────────────────
    get_elevation = None
    if args.dem:
        missing = [p for p in args.dem if not os.path.exists(p)]
        if missing:
            print(f"[WARN] DEM file(s) not found: {missing}")
        dem_paths = [p for p in args.dem if os.path.exists(p)]
        if dem_paths:
            print("\n=== Loading DEM ===")
            get_elevation = load_dem(dem_paths)

    # ── Load images ───────────────────────────────────────────────────────────
    print("\n=== Loading images ===")
    print(f"[LOAD] Directory filter: {args.filter!r}")
    images = load_images(args.inputs, args.downscale, args.filter)
    if not images:
        sys.exit("[ERROR] No images loaded. Check paths/extensions.")

    img_h, img_w = images[0]["img"].shape[:2]

    # ── Intrinsics ────────────────────────────────────────────────────────────
    K, D, calib_size = load_intrinsics(args.calib, img_w, img_h)
    if calib_size is not None:
        K = scale_K(K, calib_size[0], calib_size[1], img_w, img_h)
    elif args.downscale != 1.0:
        # Nominal intrinsics were built at full res; scale to downscaled size
        K = scale_K(K, img_w / args.downscale, img_h / args.downscale, img_w, img_h)

    # ── Drop images without GPS ───────────────────────────────────────────────
    no_gps = [im for im in images if im["meta"]["lat"] is None]
    if no_gps:
        print(f"[LOAD] Skipping {len(no_gps)} image(s) with no GPS metadata: "
              + ", ".join(Path(im["path"]).name for im in no_gps))
        images = [im for im in images if im["meta"]["lat"] is not None]
    if not images:
        sys.exit("[ERROR] No images with GPS metadata found.")

    # ── Single-image / no-SfM mode ────────────────────────────────────────────
    if len(images) == 1 or args.no_sfm:
        mode_str = "single image" if len(images) == 1 else "--no-sfm"
        print(f"\n[MODE] {mode_str} — using flat-ground ray casting")
        has_gps = any(im["meta"]["lat"] is not None for im in images)
        if not has_gps:
            sys.exit("[ERROR] No GPS found in any image. Cannot georeference.")

        tool = GeoReferenceTool3D(
            images=images, features=[], poses={},
            pts3d_sfm=np.empty((0, 3)), pts3d_gps=[],
            obs={}, K=K,
            camera_height_m=args.height,
            get_elevation=get_elevation,
            output_csv=args.output,
            output_geojson=args.geojson,
        )
        tool.run()
        return

    # ── SfM ───────────────────────────────────────────────────────────────────
    print("\n=== Detecting features ===")
    features = detect_features(images)

    print("\n=== Running Structure from Motion ===")
    poses      = {}
    pts3d_sfm  = np.empty((0, 3))
    obs        = {}
    kp_to_pt   = {}
    try:
        poses, pts3d_sfm, obs, kp_to_pt = run_sfm(images, features, K)
    except RuntimeError as e:
        print(f"[WARN] SfM failed: {e}")
        print("[WARN] Falling back to flat-ground mode")

    # ── Geo-alignment ─────────────────────────────────────────────────────────
    pts3d_gps = []
    if len(poses) >= 2 and len(pts3d_sfm) > 0:
        print("\n=== Geo-aligning reconstruction to GPS ===")
        try:
            s, R_sim3, t_sim3, orig_lat, orig_lon, orig_alt = \
                align_to_gps(poses, pts3d_sfm, images)
            pts3d_gps = sfm_to_gps(pts3d_sfm, s, R_sim3, t_sim3,
                                    orig_lat, orig_lon, orig_alt)
            print(f"[ALIGN] {len(pts3d_gps)} 3D points geo-referenced")
        except RuntimeError as e:
            print(f"[WARN] Geo-alignment failed: {e}")
            print("[WARN] 3D point cloud will not be GPS-referenced; "
                  "flat-ground fallback will be used for all clicks.")
    else:
        n_gps = sum(1 for im in images if im["meta"]["lat"] is not None)
        print(f"\n[INFO] SfM registered {len(poses)} cameras, "
              f"{n_gps} images have GPS.")
        if n_gps > 0:
            print("[INFO] Using flat-ground mode (need ≥2 registered cameras "
                  "with GPS for 3D geo-alignment).")

    # ── Interactive tool ──────────────────────────────────────────────────────
    print("\n=== Starting interactive viewer ===")
    tool = GeoReferenceTool3D(
        images=images,
        features=features,
        poses=poses,
        pts3d_sfm=pts3d_sfm,
        pts3d_gps=pts3d_gps,
        obs=obs,
        K=K,
        camera_height_m=args.height,
        get_elevation=get_elevation,
        output_csv=args.output,
        output_geojson=args.geojson,
    )
    tool.run()


if __name__ == "__main__":
    main()
