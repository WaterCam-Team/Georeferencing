#!/usr/bin/env python3
"""
reconstruct3d.py — Incremental Structure from Motion (SfM)

Builds a sparse 3D point cloud from a directory of overlapping photos
using SIFT feature matching and incremental camera pose estimation.

USAGE:
    python reconstruct3d.py <image_dir> [--calib calibration.json]
                            [--output output.ply] [--max-images N]
                            [--downscale F] [--no-display]

EXAMPLE:
    python reconstruct3d.py Pixel3a/Pixel3a-BarryPark --calib calibration.json

DEPENDENCIES:
    uv pip install opencv-python matplotlib scipy

ALGORITHM:
    1. Load images (optionally downscaled for speed)
    2. Detect SIFT features in each image
    3. Initialise reconstruction from the best-matched image pair
       (Essential matrix → camera poses → triangulate seed points)
    4. Add each remaining image incrementally:
         - Match features to already-reconstructed 3D points (PnP)
         - Recover new camera pose
         - Triangulate new points visible in this image + any previous image
    5. Simple outlier filtering (reprojection error threshold)
    6. Display: 3D scatter plot of the point cloud + camera positions

OUTPUT:
    - Optional .ply point cloud file (loadable in Meshlab, CloudCompare, QGIS…)
    - Interactive matplotlib 3D view
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" / "GTK3Agg" if needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CAMERA INTRINSICS
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics(calib_path=None, img_w=None, img_h=None,
                    focal_mm=4.44, sensor_w_mm=4.614):
    """
    Return (K, D) numpy arrays.

    Priority:
      1. calibration.json if path given and file exists
      2. Nominal estimate from focal length / sensor width + image size
         (Pixel 3a: f=4.44 mm, sensor width ≈ 4.614 mm at 4032 px)
    """
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            d = json.load(f)
        K = np.array(d["K"], dtype=np.float64)
        D = np.array(d["D"], dtype=np.float64)
        print(f"[CALIB] Loaded from {calib_path}  (RMS={d.get('rms','?')} px)")
        return K, D

    if img_w is None or img_h is None:
        raise ValueError("Need image size to build nominal intrinsics")

    px_per_mm = img_w / sensor_w_mm
    fx = fy = focal_mm * px_per_mm
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros((1, 5), dtype=np.float64)
    print(f"[CALIB] Nominal  fx={fx:.1f} px  cx={cx:.0f},{cy:.0f}")
    return K, D


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_images(image_dir, max_images=None, downscale=1.0):
    """
    Load images from a directory, sorted by filename.
    Returns list of (path, image_bgr) tuples.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if max_images:
        paths = paths[:max_images]

    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [WARN] Could not read {p.name}")
            continue
        if downscale != 1.0:
            img = cv2.resize(img, None, fx=downscale, fy=downscale,
                             interpolation=cv2.INTER_AREA)
        images.append((str(p), img))
        print(f"  Loaded {p.name}  {img.shape[1]}×{img.shape[0]}")

    print(f"[LOAD] {len(images)} images loaded from {image_dir}")
    return images


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_features(images):
    """
    Detect SIFT keypoints and descriptors for all images.
    Returns list of (keypoints, descriptors) per image.
    """
    sift = cv2.SIFT_create(nfeatures=4000)
    features = []
    for path, img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)
        features.append((kps, descs))
        print(f"  {Path(path).name}: {len(kps)} keypoints")
    print(f"[SIFT] Done.")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURE MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def match_pair(descs_a, descs_b, ratio=0.75):
    """
    FLANN-based matching with Lowe's ratio test.
    Returns array of (idx_a, idx_b) matched descriptor indices.
    """
    if descs_a is None or descs_b is None:
        return np.empty((0, 2), dtype=int)

    index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(descs_a, descs_b, k=2)

    good = []
    for m_list in raw:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < ratio * n.distance:
                good.append((m.queryIdx, m.trainIdx))

    return np.array(good, dtype=int) if good else np.empty((0, 2), dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: INITIALISE RECONSTRUCTION FROM BEST PAIR
# ─────────────────────────────────────────────────────────────────────────────

def find_best_seed_pair(features, n_images):
    """
    Find the pair with the most inlier matches (after essential matrix RANSAC).
    Searches nearby pairs first (adjacent shots are most likely to overlap).
    Returns (i, j, matches_ij, E, mask).
    """
    best = (-1, -1, None, None, None, 0)  # i, j, matches, E, mask, n_inliers

    # Search within a window; fallback to all pairs for small datasets
    search_range = min(n_images, 8)

    for i in range(n_images - 1):
        for j in range(i + 1, min(i + search_range, n_images)):
            kps_a, des_a = features[i]
            kps_b, des_b = features[j]
            matches = match_pair(des_a, des_b)
            if len(matches) < 20:
                continue
            pts_a = np.float32([kps_a[m].pt for m in matches[:, 0]])
            pts_b = np.float32([kps_b[m].pt for m in matches[:, 1]])
            E, mask = cv2.findEssentialMat(
                pts_a, pts_b, focal=1.0, pp=(0, 0),   # normalised coords later
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if mask is None:
                continue
            n_inliers = int(mask.sum())
            if n_inliers > best[5]:
                best = (i, j, matches, E, mask, n_inliers)

    return best[:5]


def init_reconstruction(i, j, features, K, K_inv):
    """
    Bootstrap the reconstruction from image pair (i, j).
    Returns:
        poses    : {img_idx: (R, t)} world→camera for each registered image
        pts3d    : (N, 3) float64 array of triangulated 3D points
        pt_colors: (N, 3) uint8 RGB
        obs      : {img_idx: {pt3d_idx: kp_idx}} observations per image
        kp_to_pt : {img_idx: {kp_idx: pt3d_idx}}
    """
    kps_a, des_a = features[i]
    kps_b, des_b = features[j]
    matches = match_pair(des_a, des_b)
    if len(matches) < 20:
        raise RuntimeError(f"Not enough matches between images {i} and {j}")

    pts_a = np.float64([kps_a[m].pt for m in matches[:, 0]])
    pts_b = np.float64([kps_b[m].pt for m in matches[:, 1]])

    # Normalise to use findEssentialMat with the real K
    pts_an = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K,
                                  np.zeros(5)).reshape(-1, 2)
    pts_bn = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K,
                                  np.zeros(5)).reshape(-1, 2)

    E, mask_e = cv2.findEssentialMat(
        pts_an, pts_bn, np.eye(3),
        method=cv2.RANSAC, prob=0.999, threshold=1e-3
    )
    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    mask_e = mask_e.ravel().astype(bool)
    _, R, t, mask_p = cv2.recoverPose(E, pts_an[mask_e], pts_bn[mask_e], np.eye(3))
    mask_p = mask_p.ravel().astype(bool)

    # Build inlier index arrays
    inlier_idx = np.where(mask_e)[0]
    inlier_idx = inlier_idx[mask_p]
    pts_a_in = pts_a[inlier_idx]
    pts_b_in = pts_b[inlier_idx]
    match_in  = matches[inlier_idx]

    # Triangulate: camera i is origin
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, pts_a_in.T, pts_b_in.T)
    pts3d = (pts4d[:3] / pts4d[3]).T   # (N, 3)

    # Keep points in front of both cameras
    z1 = pts3d[:, 2]
    R2, t2 = R, t.ravel()
    z2 = (R2 @ pts3d.T + t2[:, None])[2]
    valid = (z1 > 0) & (z2 > 0)
    pts3d = pts3d[valid]
    match_in = match_in[valid]

    print(f"[INIT] Pair ({i},{j}): {len(pts3d)} 3D points from {len(inlier_idx)} inliers")

    poses = {
        i: (np.eye(3), np.zeros(3)),
        j: (R, t2),
    }

    # Build observation tables
    obs = {i: {}, j: {}}
    kp_to_pt = {i: {}, j: {}}
    for pt_idx, (ka, kb) in enumerate(match_in[:, :2]):
        obs[i][pt_idx] = int(ka)
        obs[j][pt_idx] = int(kb)
        kp_to_pt[i][int(ka)] = pt_idx
        kp_to_pt[j][int(kb)] = pt_idx

    return poses, pts3d, obs, kp_to_pt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ADD NEW IMAGE (PnP + TRIANGULATION)
# ─────────────────────────────────────────────────────────────────────────────

def add_image(new_idx, features, poses, pts3d, obs, kp_to_pt, K,
              reproj_thresh=4.0):
    """
    Register a new image into the existing reconstruction.

    1. For each already-registered image, match features against new_idx.
    2. Find 2D↔3D correspondences (matched kp already has a 3D point).
    3. Solve PnP to get new camera pose.
    4. Triangulate new 3D points (matched kp doesn't yet have a 3D point).

    Returns updated (pts3d, obs, kp_to_pt) or None if registration fails.
    """
    kps_new, des_new = features[new_idx]
    if des_new is None:
        return None

    # Collect 2D↔3D correspondences from ALL registered images
    pts3d_corr = []    # 3D world points
    pts2d_corr = []    # corresponding 2D pixels in new image

    for reg_idx in poses:
        kps_reg, des_reg = features[reg_idx]
        matches = match_pair(des_reg, des_new)
        if len(matches) < 8:
            continue
        for m_reg, m_new in matches:
            if int(m_reg) in kp_to_pt.get(reg_idx, {}):
                pt3d_idx = kp_to_pt[reg_idx][int(m_reg)]
                if pt3d_idx < len(pts3d):
                    pts3d_corr.append(pts3d[pt3d_idx])
                    pts2d_corr.append(kps_new[m_new].pt)

    if len(pts3d_corr) < 6:
        print(f"  [SKIP] Image {new_idx}: only {len(pts3d_corr)} 2D↔3D correspondences")
        return None

    pts3d_corr = np.float64(pts3d_corr)
    pts2d_corr = np.float64(pts2d_corr)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d_corr, pts2d_corr, K, np.zeros(5),
        iterationsCount=1000, reprojectionError=reproj_thresh,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success or inliers is None or len(inliers) < 6:
        print(f"  [SKIP] Image {new_idx}: PnP failed / too few inliers")
        return None

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec.ravel()
    poses[new_idx] = (R_new, t_new)

    print(f"  [PnP] Image {new_idx}: registered  ({len(inliers)} PnP inliers)")

    # Triangulate new points from this image + each registered neighbour
    obs.setdefault(new_idx, {})
    kp_to_pt.setdefault(new_idx, {})
    new_pts_count = 0

    P_new = K @ np.hstack([R_new, t_new[:, None]])

    for reg_idx in list(poses.keys()):
        if reg_idx == new_idx:
            continue
        R_reg, t_reg = poses[reg_idx]
        P_reg = K @ np.hstack([R_reg, t_reg[:, None]])

        kps_reg, des_reg = features[reg_idx]
        matches = match_pair(des_reg, des_new)
        if len(matches) < 8:
            continue

        # Only triangulate pairs where neither kp has a 3D point yet
        pts_reg_new, pts_new_new = [], []
        match_indices = []
        for m_reg, m_new in matches:
            if (int(m_reg) not in kp_to_pt.get(reg_idx, {}) and
                    int(m_new) not in kp_to_pt.get(new_idx, {})):
                pts_reg_new.append(kps_reg[m_reg].pt)
                pts_new_new.append(kps_new[m_new].pt)
                match_indices.append((int(m_reg), int(m_new)))

        if len(pts_reg_new) < 4:
            continue

        pts_reg_arr = np.float64(pts_reg_new)
        pts_new_arr = np.float64(pts_new_new)

        pts4d = cv2.triangulatePoints(P_reg, P_new,
                                       pts_reg_arr.T, pts_new_arr.T)
        new3d = (pts4d[:3] / pts4d[3]).T

        # Keep points in front of both cameras
        z_reg = (R_reg @ new3d.T + t_reg[:, None])[2]
        z_new = (R_new @ new3d.T + t_new[:, None])[2]

        # Reprojection error filter
        def reproj_err(P, pts3, pts2):
            ph = P @ np.vstack([pts3.T, np.ones(len(pts3))])
            ph = ph[:2] / ph[2]
            return np.linalg.norm(ph.T - pts2, axis=1)

        err_reg = reproj_err(P_reg, new3d, pts_reg_arr)
        err_new = reproj_err(P_new, new3d, pts_new_arr)

        valid = (z_reg > 0) & (z_new > 0) & \
                (err_reg < reproj_thresh) & (err_new < reproj_thresh)

        for vi, (good, (kr, kn)) in enumerate(zip(valid, match_indices)):
            if not good:
                continue
            pt_idx = len(pts3d)
            pts3d = np.vstack([pts3d, new3d[vi]])
            obs[reg_idx][pt_idx] = kr
            obs[new_idx][pt_idx] = kn
            kp_to_pt[reg_idx][kr] = pt_idx
            kp_to_pt[new_idx][kn] = pt_idx
            new_pts_count += 1

    print(f"         Triangulated {new_pts_count} new 3D points  "
          f"(total: {len(pts3d)})")
    return pts3d, obs, kp_to_pt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: POINT CLOUD COLOURING
# ─────────────────────────────────────────────────────────────────────────────

def colour_points(pts3d, poses, obs, images):
    """
    Assign each 3D point the average RGB colour from all images that observe it.
    Returns (N, 3) uint8 array.
    """
    colours = np.zeros((len(pts3d), 3), dtype=np.float64)
    counts  = np.zeros(len(pts3d), dtype=int)

    for img_idx, pt_obs in obs.items():
        _, img = images[img_idx]
        R, t = poses[img_idx]
        h, w = img.shape[:2]

        for pt_idx, kp_idx in pt_obs.items():
            if pt_idx >= len(pts3d):
                continue
            p3 = pts3d[pt_idx]
            p2 = R @ p3 + t
            if p2[2] <= 0:
                continue
            # Use 3D point projected back to image
            u = int(round(p2[0] / p2[2]))
            v = int(round(p2[1] / p2[2]))
            if 0 <= u < w and 0 <= v < h:
                bgr = img[v, u].astype(np.float64)
                colours[pt_idx] += bgr[::-1]   # BGR → RGB
                counts[pt_idx] += 1

    mask = counts > 0
    colours[mask] /= counts[mask, None]
    colours[~mask] = [180, 180, 180]
    return colours.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: PLY EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_ply(path, pts3d, colours):
    """Write a binary PLY point cloud (positions + RGB)."""
    n = len(pts3d)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(pts3d, colours):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    print(f"[PLY] Saved {n} points to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_camera_frustum(ax, R, t, K, img_w, img_h, scale=0.3, color="blue"):
    """
    Draw a small camera frustum in world coordinates.
    R, t : world→camera transform  (world pt → R @ pt + t)
    Camera centre in world = -R.T @ t
    """
    # Camera centre
    Rw = R.T
    tw = -Rw @ t
    cam_center = tw

    # Unproject image corners through K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_px = np.array([
        [0,     0,     1],
        [img_w, 0,     1],
        [img_w, img_h, 1],
        [0,     img_h, 1],
    ], dtype=np.float64)
    corners_px[:, 0] = (corners_px[:, 0] - cx) / fx
    corners_px[:, 1] = (corners_px[:, 1] - cy) / fy
    # Rotate to world frame
    dirs_world = (Rw @ corners_px.T).T   # (4, 3)
    corners_world = cam_center + scale * dirs_world

    # Draw four edges from camera centre to corners
    for c in corners_world:
        ax.plot([cam_center[0], c[0]],
                [cam_center[1], c[1]],
                [cam_center[2], c[2]],
                color=color, linewidth=0.8, alpha=0.6)
    # Draw the image rectangle
    rect = np.vstack([corners_world, corners_world[0]])
    ax.plot(rect[:, 0], rect[:, 1], rect[:, 2],
            color=color, linewidth=0.8, alpha=0.6)


def visualise(pts3d, colours, poses, K, img_w, img_h,
              title="3D Reconstruction"):
    """
    Interactive 3D scatter of the point cloud + camera frustums.
    """
    # Outlier removal: keep points within 3 std of median distance
    centre = np.median(pts3d, axis=0)
    dists  = np.linalg.norm(pts3d - centre, axis=1)
    thresh = np.median(dists) + 3 * dists.std()
    keep   = dists < thresh
    pts_v  = pts3d[keep]
    col_v  = colours[keep] / 255.0

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # Point cloud (subsample if large)
    step = max(1, len(pts_v) // 50_000)
    ax.scatter(pts_v[::step, 0], pts_v[::step, 1], pts_v[::step, 2],
               c=col_v[::step], s=1, linewidths=0)

    # Camera frustums
    cam_positions = []
    colors_cycle = plt.cm.plasma(np.linspace(0, 1, len(poses)))
    for (img_idx, (R, t)), col in zip(sorted(poses.items()), colors_cycle):
        draw_camera_frustum(ax, R, t, K, img_w, img_h,
                            color=col[:3], scale=0.3)
        cam_pos = -R.T @ t
        cam_positions.append(cam_pos)

    if cam_positions:
        cp = np.array(cam_positions)
        ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
                "r--", linewidth=1, alpha=0.5, label="Camera path")
        ax.legend(fontsize=8)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Equal-ish aspect ratio
    for start, end, set_lim in [
        (pts_v[:, 0], pts_v[:, 0], ax.set_xlim),
        (pts_v[:, 1], pts_v[:, 1], ax.set_ylim),
        (pts_v[:, 2], pts_v[:, 2], ax.set_zlim),
    ]:
        lo, hi = np.percentile(start, 2), np.percentile(end, 98)
        mid = (lo + hi) / 2
        half = max((hi - lo) / 2, 0.1)
        set_lim(mid - half, mid + half)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incremental SfM: build a 3D point cloud from photos."
    )
    parser.add_argument("image_dir", help="Folder containing overlapping photos")
    parser.add_argument("--calib", default=None,
                        help="Path to calibration.json (optional)")
    parser.add_argument("--output", default=None,
                        help="Save point cloud to this .ply file")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Use only the first N images (speed/memory)")
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Resize factor for images (default 0.5 for speed)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip the interactive 3D viewer")
    args = parser.parse_args()

    # ── Load images ──────────────────────────────────────────────────────────
    print(f"\n=== Loading images ===")
    images = load_images(args.image_dir, args.max_images, args.downscale)
    if len(images) < 2:
        sys.exit("Need at least 2 images.")

    _, sample = images[0]
    img_h, img_w = sample.shape[:2]

    # ── Intrinsics ───────────────────────────────────────────────────────────
    K, D = load_intrinsics(args.calib, img_w, img_h)
    # Scale K if images were downscaled
    if args.downscale != 1.0:
        K = K.copy()
        K[0] *= args.downscale
        K[1] *= args.downscale

    K_inv = np.linalg.inv(K)
    n = len(images)

    # ── Detect features ──────────────────────────────────────────────────────
    print(f"\n=== Detecting features ===")
    features = detect_features(images)

    # ── Initialise reconstruction from seed pair ─────────────────────────────
    print(f"\n=== Finding best seed pair ===")
    si, sj, _, _, _ = find_best_seed_pair(features, n)
    if si < 0:
        sys.exit("Could not find a valid seed pair. "
                 "Check that images overlap sufficiently.")
    print(f"[SEED] Best pair: images {si} and {sj}")

    print(f"\n=== Initialising reconstruction ===")
    poses, pts3d, obs, kp_to_pt = init_reconstruction(si, sj, features, K, K_inv)

    # ── Add remaining images incrementally ───────────────────────────────────
    print(f"\n=== Incremental registration ===")
    remaining = [i for i in range(n) if i not in poses]
    for idx in remaining:
        result = add_image(idx, features, poses, pts3d, obs, kp_to_pt, K)
        if result is not None:
            pts3d, obs, kp_to_pt = result

    print(f"\n=== Reconstruction complete ===")
    print(f"  Registered cameras : {len(poses)} / {n}")
    print(f"  3D points          : {len(pts3d)}")

    # ── Colour the point cloud ────────────────────────────────────────────────
    print(f"\n=== Colouring point cloud ===")
    colours = colour_points(pts3d, poses, obs, images)

    # ── Save PLY ─────────────────────────────────────────────────────────────
    if args.output:
        save_ply(args.output, pts3d, colours)

    # ── Visualise ────────────────────────────────────────────────────────────
    if not args.no_display:
        title = f"3D Reconstruction — {len(pts3d)} pts, {len(poses)} cameras"
        visualise(pts3d, colours, poses, K, img_w, img_h, title)


if __name__ == "__main__":
    main()
