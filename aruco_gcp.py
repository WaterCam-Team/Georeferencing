"""
ArUco-based ground control point workflow.

Two halves:
  1. detect_in_photo  — find ArUco markers in a UFONet photo (undistorted);
                        returns pixel (u, v) centre per marker ID.
  2. locate_in_pix4d  — detect the same markers across Pix4DCatch frames,
                        back-project through the depth map and camera pose,
                        and return a robust RTK-quality (lat, lon, elev_m)
                        per marker ID by averaging across views.

Combining both gives a complete GCP CSV with no manual pixel-clicking.

Marker recommendation
---------------------
Print 15×15 cm ArUco DICT_4X4_50 markers on weather-proof PVC/aluminium.
Black/white → high contrast in both visible and NIR bands (OV5647).
Label back of each board with its ArUco ID for field bookkeeping.

OPF rotation convention (verified against Pix4DCatch data)
----------------------------------------------------------
R = Rx(omega) @ Ry(phi) @ Rz(kappa)   camera → projected-world (ENU)
Camera frame: X=right, Y=up, Z=backward (OpenGL).
Forward ray: d_world = R @ [x/d, -y/d, -1] * depth   (pixel → ENU offset).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ArUco helpers
# ─────────────────────────────────────────────────────────────────────────────

_DICT_MAP = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
}


def _make_detector(dict_name: str):
    key = dict_name.upper()
    if key not in _DICT_MAP:
        raise ValueError(f"Unknown ArUco dict '{dict_name}'. Options: {list(_DICT_MAP)}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_MAP[key])
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _marker_centre(corners_px) -> tuple[float, float]:
    """Mean of four corners → (u, v)."""
    c = np.array(corners_px[0], dtype=np.float64)
    return float(c[:, 0].mean()), float(c[:, 1].mean())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Detect in UFONet photo
# ─────────────────────────────────────────────────────────────────────────────

def detect_in_photo(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    dict_name: str = "DICT_4X4_50",
) -> dict[int, dict]:
    """
    Detect ArUco markers in a (possibly distorted) image.

    Undistorts first, then detects.  Returns pixel coordinates in the
    undistorted image so they can be fed directly to pixel_to_gps_terrain.

    Returns:
        { marker_id: {"center_uv": (u, v), "corners_uv": list[tuple],
                      "K_undist": np.ndarray, "roi": (x, y, w, h)} }
    """
    h, w = image.shape[:2]
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    undist = cv2.undistort(image, K, D, None, K_new)
    x0, y0, cw, ch = roi
    undist = undist[y0:y0+ch, x0:x0+cw]
    K_crop = K_new.copy()
    K_crop[0, 2] -= x0
    K_crop[1, 2] -= y0

    detector = _make_detector(dict_name)
    corners, ids, _ = detector.detectMarkers(undist)

    result = {}
    if ids is None:
        return result
    for i, mid in enumerate(ids.flatten()):
        cx, cy = _marker_centre(corners[i])
        result[int(mid)] = {
            "center_uv": (cx, cy),
            "corners_uv": [(float(pt[0]), float(pt[1])) for pt in corners[i][0]],
            "K_undist": K_crop,
            "roi": roi,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Locate markers in Pix4DCatch session
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_opf(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    """
    OPF camera→world rotation: R = Rx(ω) @ Ry(φ) @ Rz(κ).
    Camera frame: X=right, Y=up, Z=backward.  World: ENU.
    """
    o, p, k = np.radians([omega_deg, phi_deg, kappa_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(o), -np.sin(o)],
                   [0, np.sin(o),  np.cos(o)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(k), -np.sin(k), 0],
                   [np.sin(k),  np.cos(k), 0],
                   [0,          0,          1]])
    return Rx @ Ry @ Rz


def _pixel_to_enu_offset(
    u: float, v: float,
    depth_m: float,
    fx: float, fy: float, cx: float, cy: float,
    R_c2w: np.ndarray,
) -> np.ndarray:
    """
    Convert image pixel + depth → ENU offset from camera origin (metres).

    Camera frame: X=right, Y=up, Z=backward.
    Scene point in camera frame (depth along -Z):
        x_c =  (u - cx) / fx * depth
        y_c = -(v - cy) / fy * depth   (image v↓ but camera Y↑)
        z_c = -depth                    (scene is in -Z direction)
    """
    x_c = (u - cx) / fx * depth_m
    y_c = -(v - cy) / fy * depth_m
    z_c = -depth_m
    return R_c2w @ np.array([x_c, y_c, z_c], dtype=np.float64)


def locate_in_pix4d(
    session_dir: str,
    dict_name: str = "DICT_4X4_50",
    every_n_frames: int = 5,
    min_depth_m: float = 0.3,
    max_depth_m: float = 20.0,
    min_views: int = 2,
) -> dict[int, dict]:
    """
    Detect ArUco markers across Pix4DCatch frames and return their
    RTK-quality world positions (averaged over multiple views).

    Parameters
    ----------
    session_dir    : Path to the Pix4DCatch session directory.
    dict_name      : ArUco dictionary (must match markers placed at site).
    every_n_frames : Sample every Nth frame (trade speed vs robustness).
    min_depth_m    : Ignore depth readings below this (likely invalid).
    max_depth_m    : Ignore depth readings above this (too far for accuracy).
    min_views      : Discard markers seen in fewer frames than this.

    Returns
    -------
    { marker_id: {"lat": float, "lon": float, "elev_m": float,
                  "n_views": int, "std_m": float} }
    """
    from pyproj import Transformer

    session_dir = Path(session_dir)
    images_dir  = session_dir / "images"
    opf_dir     = session_dir / "opf_files"

    # ── Load camera intrinsics ────────────────────────────────────────────────
    with open(opf_dir / "input_cameras.json") as f:
        inp = json.load(f)
    sensor = inp["sensors"][0]
    img_w, img_h = sensor["image_size_px"]        # 1920×1440
    fx_color = sensor["internals"]["focal_length_px"]
    cx_color, cy_color = sensor["internals"]["principal_point_px"]

    # ── Load capture list: geolocation (lat/lon/alt) + frame order ────────────
    captures_inp  = inp["captures"]               # same order as Image_XXXXXX.jpg
    with open(opf_dir / "projected_input_cameras.json") as f:
        proj = json.load(f)
    # projected captures keyed by id
    proj_by_id = {c["id"]: c for c in proj["captures"]}

    # ── Pix4DCatch depth map is 192×256 = 1/7.5 scale of 1440×1920 ──────────
    depth_scale = 256.0 / img_w   # 0.1333…
    fx_d = fx_color * depth_scale
    fy_d = fx_color * depth_scale
    cx_d = cx_color * depth_scale
    cy_d = cy_color * depth_scale

    # ── Coordinate transform: NAD83(2011) geographic → UTM 18N ───────────────
    to_utm = Transformer.from_crs("EPSG:6318", "EPSG:6347", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:6347", "EPSG:4326", always_xy=True)

    # ── Geoid separation for ellipsoidal → orthometric conversion ────────────
    # OPF geolocation altitudes are WGS84 ellipsoidal; elev_m output should be
    # NAVD88/EGM96 orthometric so it matches DEM-derived elevations in the rest
    # of the pipeline.  At Syracuse NY, N ≈ −34.5 m, so omitting this causes a
    # ~35 m error in all reported marker elevations.
    geoid_sep: Optional[float] = None
    rtk_path = session_dir / "geolocations" / "rtkGPS.csv"
    if rtk_path.exists():
        seps: list[float] = []
        with open(rtk_path) as _f:
            for _row in csv.DictReader(_f):
                try:
                    seps.append(float(_row["GeoidSeparation"]))
                except (KeyError, ValueError):
                    pass
        if seps:
            geoid_sep = sum(seps) / len(seps)
    if geoid_sep is not None:
        print(f"[aruco_gcp] Geoid separation N={geoid_sep:.3f} m → elev_m will be orthometric")
    else:
        print("[aruco_gcp] WARNING: rtkGPS.csv missing or no GeoidSeparation; elev_m will be ellipsoidal")

    detector = _make_detector(dict_name)
    observations: dict[int, list[np.ndarray]] = {}  # id → list of ENU world points

    n_frames = len(captures_inp)
    print(f"[aruco_gcp] {n_frames} frames, sampling every {every_n_frames} → "
          f"~{n_frames // every_n_frames} frames to check")

    for frame_idx in range(0, n_frames, every_n_frames):
        cap = captures_inp[frame_idx]
        frame_num = frame_idx + 1                   # files are 1-indexed

        img_path   = images_dir / f"Image_{frame_num:06d}.jpg"
        depth_path = images_dir / f"DepthMap_{frame_num:06d}.tiff"
        if not img_path.exists() or not depth_path.exists():
            continue

        # ── Colour image → ArUco detection ───────────────────────────────────
        color = cv2.imread(str(img_path))
        if color is None:
            continue
        corners, ids, _ = detector.detectMarkers(color)
        if ids is None:
            continue

        # ── Depth map ─────────────────────────────────────────────────────────
        try:
            import tifffile
            depth = tifffile.imread(str(depth_path)).astype(np.float32)
        except Exception:
            continue
        d_h, d_w = depth.shape[:2]

        # ── Camera pose ───────────────────────────────────────────────────────
        geo = cap["geolocation"]["coordinates"]  # [lat, lon, alt_ellipsoidal]
        cam_lat, cam_lon, cam_alt = geo[0], geo[1], geo[2]
        cam_e, cam_n = to_utm.transform(cam_lon, cam_lat)
        cam_pos_enu = np.array([cam_e, cam_n, cam_alt], dtype=np.float64)

        proj_cap = proj_by_id.get(cap["id"])
        if proj_cap is None:
            continue
        om, ph, ka = proj_cap["orientation"]["angles_deg"]
        R_c2w = _rotation_opf(om, ph, ka)

        # ── For each detected marker ──────────────────────────────────────────
        for i, mid in enumerate(ids.flatten()):
            u_color, v_color = _marker_centre(corners[i])

            # Scale colour pixel → depth map pixel
            u_d = u_color * depth_scale
            v_d = v_color * depth_scale
            # Nearest-neighbour sample (depth map is coarse)
            u_di = int(np.clip(round(u_d), 0, d_w - 1))
            v_di = int(np.clip(round(v_d), 0, d_h - 1))
            d_val = float(depth[v_di, u_di])

            if not (min_depth_m <= d_val <= max_depth_m):
                continue

            enu_offset = _pixel_to_enu_offset(
                u_d, v_d, d_val,
                fx_d, fy_d, cx_d, cy_d,
                R_c2w,
            )
            world_enu = cam_pos_enu + enu_offset   # East, North, Up (metres, UTM)

            mid_int = int(mid)
            observations.setdefault(mid_int, []).append(world_enu)

    # ── Average observations per marker ──────────────────────────────────────
    result: dict[int, dict] = {}
    for mid, pts in observations.items():
        if len(pts) < min_views:
            continue
        arr = np.array(pts)                         # (N, 3) ENU
        mean_enu = arr.mean(axis=0)
        std_m    = float(np.std(np.linalg.norm(arr - mean_enu, axis=1)))
        lon_out, lat_out = to_wgs84.transform(mean_enu[0], mean_enu[1])
        elev_ellip = float(mean_enu[2])
        elev_out = (elev_ellip - geoid_sep) if geoid_sep is not None else elev_ellip
        result[mid] = {
            "lat": float(lat_out),
            "lon": float(lon_out),
            "elev_m": elev_out,
            "n_views": len(pts),
            "std_m": std_m,
        }
        datum_tag = "ortho" if geoid_sep is not None else "ellip"
        print(f"  marker {mid:3d}: lat={lat_out:.7f}  lon={lon_out:.7f}  "
              f"elev={elev_out:.2f} m ({datum_tag})  n={len(pts)}  std={std_m:.3f} m")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Write GCP CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_gcp_csv(
    photo_detections: dict[int, dict],
    pix4d_locations: dict[int, dict],
    output_path: str,
    image_basename: str = "",
) -> list[dict]:
    """
    Merge photo pixel detections with Pix4DCatch world locations.

    Only markers present in both dicts are written.  Returns list of rows.
    """
    rows = []
    common = set(photo_detections) & set(pix4d_locations)
    if not common:
        print("[aruco_gcp] No markers found in both photo and Pix4DCatch session.")
        return rows

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "pixel_u", "pixel_v",
                                           "lat", "lon", "elev_m",
                                           "n_views", "std_m", "image"])
        w.writeheader()
        for mid in sorted(common):
            u, v = photo_detections[mid]["center_uv"]
            loc  = pix4d_locations[mid]
            row  = {
                "label":   f"aruco_{mid}",
                "pixel_u": round(u, 2),
                "pixel_v": round(v, 2),
                "lat":     round(loc["lat"],    8),
                "lon":     round(loc["lon"],    8),
                "elev_m":  round(loc["elev_m"], 3),
                "n_views": loc["n_views"],
                "std_m":   round(loc["std_m"],  4),
                "image":   image_basename,
            }
            w.writerow(row)
            rows.append(row)

    print(f"[aruco_gcp] Wrote {len(rows)} GCPs → {output_path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Extract ArUco GCPs from a UFONet photo + Pix4DCatch session",
        epilog=(
            "Workflow:\n"
            "  1. Place 4-6 ArUco DICT_4X4_50 markers (15x15 cm) within the\n"
            "     UFONet camera footprint before the session.\n"
            "  2. Run Pix4DCatch scan of the same area.\n"
            "  3. Run this script → gcps.csv ready for georeference_terrain.py\n"
            "     or gcp.py refine_pose_from_gcps.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image",    required=True, help="UFONet photo (.jpg)")
    p.add_argument("--pix4d",   required=True, help="Pix4DCatch session directory")
    p.add_argument("--calibration", "-c", default="./calibration.json")
    p.add_argument("--dict",    default="DICT_4X4_50",
                   help="ArUco dictionary (default DICT_4X4_50)")
    p.add_argument("--every-n", type=int, default=5,
                   help="Sample every N Pix4D frames (default 5)")
    p.add_argument("--min-views", type=int, default=2,
                   help="Min Pix4D frames per marker (default 2)")
    p.add_argument("--output",  "-o", default="./gcps.csv")
    args = p.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from georeference_tool import load_calibrated_intrinsics

    image = cv2.imread(args.image)
    if image is None:
        print(f"Cannot read image: {args.image}")
        sys.exit(1)

    K, D, calib_size, _ = load_calibrated_intrinsics(args.calibration)
    h, w = image.shape[:2]
    if calib_size and (calib_size[0], calib_size[1]) != (w, h):
        from georeference_tool import scale_intrinsics_for_resolution
        K = scale_intrinsics_for_resolution(K, calib_size[0], calib_size[1], w, h)

    print(f"\n[1] ArUco detection in UFONet photo: {os.path.basename(args.image)}")
    photo_det = detect_in_photo(image, K, D, args.dict)
    if not photo_det:
        print("  No markers detected. Check: correct dict? markers in frame? image readable?")
        sys.exit(1)
    for mid, det in photo_det.items():
        u, v = det["center_uv"]
        print(f"  marker {mid:3d}: pixel ({u:.1f}, {v:.1f})")

    print(f"\n[2] Locating markers in Pix4DCatch session: {args.pix4d}")
    pix4d_loc = locate_in_pix4d(
        args.pix4d, args.dict,
        every_n_frames=args.every_n,
        min_views=args.min_views,
    )
    if not pix4d_loc:
        print("  No markers located in Pix4DCatch session.")
        sys.exit(1)

    print(f"\n[3] Writing GCP CSV: {args.output}")
    rows = write_gcp_csv(photo_det, pix4d_loc, args.output,
                         os.path.basename(args.image))
    if not rows:
        sys.exit(1)

    print(f"\nDone. {len(rows)} GCPs written.")
    print("Next: pass --gcps", args.output, "to georeference_terrain.py")


if __name__ == "__main__":
    main()
