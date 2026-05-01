"""
Interactive RTK coordinate extractor for Pix4DCatch frames.

Opens a Pix4DCatch image frame in an OpenCV window.  Left-clicking any pixel
back-projects through the depth map and camera pose to produce an RTK-quality
(lat, lon, elev_m) for the clicked surface point.  Prints to terminal and
accumulates in a GCP CSV suitable for use with gcp.py / georeference_terrain.py.

Typical use — validate the UFO-006 georeferencing pipeline without ArUco markers:
  1. Open the closest Pix4DCatch frame to the camera footprint.
  2. Click identifiable features (road markings, curb edge, lamp post base).
  3. Find the same features in the UFO-006 photo and click them in
     georeference_terrain.py.
  4. Compare predicted vs RTK coordinates → residual in metres.

Usage:
    python gcp_from_pix4d.py <session_dir> --frame Image_001418.jpg [--out gcps_rtk.csv]

    # Or let the script find the closest frame to a given location:
    python gcp_from_pix4d.py <session_dir> --near-lat 43.039783 --near-lon -76.082819 --n 5

Keyboard shortcuts (OpenCV window):
    Left-click   : extract RTK coordinate at clicked pixel
    S            : save accumulated GCPs to CSV
    N / P        : next / previous frame (when --n frames are loaded)
    Q            : quit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pyproj import Transformer


# ─────────────────────────────────────────────────────────────────────────────
# OPF parsing helpers  (shared with aruco_gcp.py)
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_opf(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    o, p, k = np.radians([omega_deg, phi_deg, kappa_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(o), -np.sin(o)], [0, np.sin(o), np.cos(o)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(k), -np.sin(k), 0], [np.sin(k), np.cos(k), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _load_session_meta(session_dir: Path):
    """
    Return (captures_inp, proj_by_id, sensor_params, to_utm, to_wgs84, geoid_sep).
    """
    opf = session_dir / "opf_files"

    with open(opf / "input_cameras.json") as f:
        inp = json.load(f)

    sensor = inp["sensors"][0]
    img_w, img_h = sensor["image_size_px"]          # e.g. 1920×1440
    fx = sensor["internals"]["focal_length_px"]
    cx, cy = sensor["internals"]["principal_point_px"]

    with open(opf / "projected_input_cameras.json") as f:
        proj = json.load(f)
    proj_by_id = {c["id"]: c for c in proj["captures"]}

    # Depth map is 256 wide × 192 tall = 1/7.5 of 1920×1440
    depth_scale = 256.0 / img_w
    fx_d = fx * depth_scale
    fy_d = fx * depth_scale
    cx_d = cx * depth_scale
    cy_d = cy * depth_scale

    to_utm  = Transformer.from_crs("EPSG:6318", "EPSG:6347", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:6347", "EPSG:4326", always_xy=True)

    # Geoid separation from rtkGPS.csv for Z→orthometric conversion
    geoid_sep = None
    rtk_path = session_dir / "geolocations" / "rtkGPS.csv"
    if rtk_path.exists():
        seps = []
        with open(rtk_path) as f:
            for row in csv.DictReader(f):
                try:
                    seps.append(float(row["GeoidSeparation"]))
                except (KeyError, ValueError):
                    pass
        if seps:
            geoid_sep = float(np.mean(seps))

    return (
        inp["captures"], proj_by_id,
        dict(img_w=img_w, img_h=img_h, fx_d=fx_d, fy_d=fy_d, cx_d=cx_d, cy_d=cy_d,
             depth_scale=depth_scale),
        to_utm, to_wgs84, geoid_sep,
    )


def _build_cap_map(session_dir: Path):
    """Map capture_id → {image: 'Image_XXXXXX.jpg', depth: 'DepthMap_XXXXXX.tiff'}."""
    opf = session_dir / "opf_files"
    with open(opf / "camera_list.json") as f:
        cl = json.load(f)
    cam_id_to_uri = {c["id"]: c["uri"] for c in cl["cameras"]}

    with open(opf / "input_cameras.json") as f:
        ic = json.load(f)

    cap_map: dict[int, dict] = {}
    for cap in ic["captures"]:
        cid = cap["id"]
        for cam in cap.get("cameras", []):
            # Image file: camera's own ID
            sid = cam["id"]
            uri = cam_id_to_uri.get(sid, "")
            if "Image" in uri:
                cap_map.setdefault(cid, {})["image"] = uri.replace("../images/", "")
            # Depth map: ID stored in extensions
            ext = cam.get("extensions", {}).get("PIX4D_input_depth_map", {})
            dm_id = ext.get("id")
            if dm_id and dm_id in cam_id_to_uri:
                dm_uri = cam_id_to_uri[dm_id]
                cap_map.setdefault(cid, {})["depth"] = dm_uri.replace("../images/", "")
    return cap_map


# ─────────────────────────────────────────────────────────────────────────────
# Core: depth-map back-projection for one pixel
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_rtk(
    u: float, v: float,
    depth: np.ndarray,
    cam_lat: float, cam_lon: float, cam_alt_ellip: float,
    omega_deg: float, phi_deg: float, kappa_deg: float,
    sensor: dict,
    to_utm: Transformer,
    to_wgs84: Transformer,
    geoid_sep: Optional[float] = None,
    min_depth_m: float = 0.2,
    max_depth_m: float = 20.0,
) -> Optional[dict]:
    """
    Back-project image pixel (u, v) through the depth map to get world RTK coords.

    Returns dict with lat, lon, elev_m (orthometric if geoid_sep provided),
    elev_ellip, depth_m, and the camera position.  Returns None if depth is
    invalid or out of range.
    """
    fx_d = sensor["fx_d"]
    fy_d = sensor["fy_d"]
    cx_d = sensor["cx_d"]
    cy_d = sensor["cy_d"]
    ds   = sensor["depth_scale"]

    d_h, d_w = depth.shape[:2]
    u_d = u * ds
    v_d = v * ds
    u_di = int(np.clip(round(u_d), 0, d_w - 1))
    v_di = int(np.clip(round(v_d), 0, d_h - 1))
    d_val = float(depth[v_di, u_di])

    if not (min_depth_m <= d_val <= max_depth_m):
        return None

    cam_e, cam_n = to_utm.transform(cam_lon, cam_lat)
    R_c2w = _rotation_opf(omega_deg, phi_deg, kappa_deg)

    x_c = (u_d - cx_d) / fx_d * d_val
    y_c = -(v_d - cy_d) / fy_d * d_val
    z_c = -d_val
    enu_offset = R_c2w @ np.array([x_c, y_c, z_c])

    world_enu = np.array([cam_e, cam_n, cam_alt_ellip]) + enu_offset
    lon_out, lat_out = to_wgs84.transform(world_enu[0], world_enu[1])

    elev_ellip = float(world_enu[2])
    elev_ortho = (elev_ellip - geoid_sep) if geoid_sep is not None else None

    return {
        "lat": float(lat_out),
        "lon": float(lon_out),
        "elev_ellip_m": elev_ellip,
        "elev_ortho_m": elev_ortho,
        "depth_m": d_val,
        "cam_lat": cam_lat,
        "cam_lon": cam_lon,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Find frames closest to a target location
# ─────────────────────────────────────────────────────────────────────────────

def find_closest_frames(captures_inp, proj_by_id, to_utm, target_lat, target_lon, n=5):
    tx, ty = to_utm.transform(target_lon, target_lat)
    dists = []
    for cap in captures_inp:
        geo = cap["geolocation"]["coordinates"]
        clat, clon = geo[0], geo[1]
        cx2, cy2 = to_utm.transform(clon, clat)
        d = np.sqrt((cx2 - tx)**2 + (cy2 - ty)**2)
        dists.append((d, cap["id"]))
    dists.sort()
    return [cap_id for _, cap_id in dists[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

class FrameViewer:
    def __init__(self, session_dir: Path, frame_names: list[str], out_csv: Path,
                 captures_inp, proj_by_id, sensor, to_utm, to_wgs84, geoid_sep):
        self.session_dir = session_dir
        self.frame_names = frame_names
        self.out_csv     = out_csv
        self.captures_inp = captures_inp
        self.proj_by_id   = proj_by_id
        self.sensor       = sensor
        self.to_utm       = to_utm
        self.to_wgs84     = to_wgs84
        self.geoid_sep    = geoid_sep

        # Build frame_name → capture lookup
        self.name_to_cap: dict[str, dict] = {}
        cap_map = _build_cap_map(session_dir)
        for cap in captures_inp:
            cid = cap["id"]
            files = cap_map.get(cid, {})
            img_name = files.get("image", "")
            if img_name:
                self.name_to_cap[img_name] = cap

        self.frame_idx = 0
        self.gcps: list[dict] = []
        self.display = None
        self.depth   = None
        self.cur_cap = None

    def _load_frame(self):
        name = self.frame_names[self.frame_idx]
        img_path   = self.session_dir / "images" / name
        depth_name = name.replace("Image_", "DepthMap_").replace(".jpg", ".tiff")
        depth_path = self.session_dir / "images" / depth_name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Cannot read {img_path}")
            return False

        try:
            import tifffile
            self.depth = tifffile.imread(str(depth_path)).astype(np.float32)
        except Exception as e:
            print(f"[WARN] Depth map unavailable ({e})")
            self.depth = None

        self.cur_cap = self.name_to_cap.get(name)
        if self.cur_cap is None:
            print(f"[WARN] No pose found for {name}")

        # Scale image for display if very large
        h, w = img.shape[:2]
        max_dim = 1400
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        self._img_scale = scale
        self._img_orig  = img.copy()
        self.display    = img.copy()
        self._redraw_title(name)
        return True

    def _redraw_title(self, name):
        idx = self.frame_idx + 1
        n   = len(self.frame_names)
        cap = self.cur_cap
        pos_str = ""
        if cap:
            geo = cap["geolocation"]["coordinates"]
            pos_str = f"  cam=({geo[0]:.5f},{geo[1]:.5f})"
        title = f"[{idx}/{n}] {name}{pos_str}   GCPs:{len(self.gcps)}   S=save N=next P=prev Q=quit"
        cv2.setWindowTitle("Pix4D GCP Extractor", title)

    def _on_mouse(self, event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.cur_cap is None or self.depth is None:
            print("[WARN] No pose/depth — cannot back-project")
            return

        # Convert display pixel → original pixel
        s = self._img_scale
        u = x / s
        v = y / s

        geo  = self.cur_cap["geolocation"]["coordinates"]
        proj_cap = self.proj_by_id.get(self.cur_cap["id"])
        if proj_cap is None:
            print("[WARN] No projected pose for this frame")
            return
        om, ph, ka = proj_cap["orientation"]["angles_deg"]

        result = pixel_to_rtk(
            u, v, self.depth,
            geo[0], geo[1], geo[2],
            om, ph, ka,
            self.sensor, self.to_utm, self.to_wgs84,
            geoid_sep=self.geoid_sep,
        )
        if result is None:
            print(f"  ({x},{y}) → depth invalid or out of range")
            return

        gcp_id = len(self.gcps) + 1
        self.gcps.append({
            "id":         gcp_id,
            "frame":      self.frame_names[self.frame_idx],
            "pixel_u":    round(u, 1),
            "pixel_v":    round(v, 1),
            "lat":        result["lat"],
            "lon":        result["lon"],
            "elev_ortho_m": result["elev_ortho_m"],
            "elev_ellip_m": result["elev_ellip_m"],
            "depth_m":    result["depth_m"],
        })

        elev_str = (f"{result['elev_ortho_m']:.3f} m ortho"
                    if result["elev_ortho_m"] is not None
                    else f"{result['elev_ellip_m']:.3f} m ellip")
        print(f"  GCP {gcp_id}: lat={result['lat']:.6f}  lon={result['lon']:.6f}  "
              f"elev={elev_str}  depth={result['depth_m']:.2f} m")

        # Draw on display
        r = max(6, int(12 / self._img_scale))
        cv2.circle(self.display, (x, y), r, (0, 255, 0), 2)
        cv2.putText(self.display, str(gcp_id), (x + r + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 / self._img_scale, (0, 255, 0), 2)
        cv2.imshow("Pix4D GCP Extractor", self.display)
        self._redraw_title(self.frame_names[self.frame_idx])

    def _save(self):
        if not self.gcps:
            print("No GCPs to save.")
            return
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.gcps[0].keys()))
            w.writeheader()
            w.writerows(self.gcps)
        print(f"Saved {len(self.gcps)} GCPs → {self.out_csv}")

    def run(self):
        cv2.namedWindow("Pix4D GCP Extractor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Pix4D GCP Extractor", self._on_mouse)

        if not self._load_frame():
            return

        cv2.imshow("Pix4D GCP Extractor", self.display)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self._save()
            elif key == ord('n') and self.frame_idx < len(self.frame_names) - 1:
                self.frame_idx += 1
                if self._load_frame():
                    self.display = self._img_orig.copy()
                    cv2.imshow("Pix4D GCP Extractor", self.display)
            elif key == ord('p') and self.frame_idx > 0:
                self.frame_idx -= 1
                if self._load_frame():
                    self.display = self._img_orig.copy()
                    cv2.imshow("Pix4D GCP Extractor", self.display)

        cv2.destroyAllWindows()
        if self.gcps:
            ans = input(f"Save {len(self.gcps)} GCPs to {self.out_csv}? [y/N] ").strip().lower()
            if ans == 'y':
                self._save()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", help="Pix4DCatch session directory")
    p.add_argument("--frame", "-f", default=None,
                   help="Specific image filename (e.g. Image_001418.jpg)")
    p.add_argument("--near-lat", type=float, default=None,
                   help="Find N frames closest to this latitude")
    p.add_argument("--near-lon", type=float, default=None,
                   help="Find N frames closest to this longitude")
    p.add_argument("--n", type=int, default=5,
                   help="Number of closest frames to load (default 5)")
    p.add_argument("--out", "-o", default="gcps_rtk.csv",
                   help="Output GCP CSV (default: gcps_rtk.csv)")
    args = p.parse_args()

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        print(f"ERROR: {session_dir} is not a directory")
        sys.exit(1)

    captures_inp, proj_by_id, sensor, to_utm, to_wgs84, geoid_sep = \
        _load_session_meta(session_dir)

    if geoid_sep is not None:
        print(f"Geoid separation N={geoid_sep:.3f} m  (Z output = orthometric)")
    else:
        print("WARNING: geoid separation unknown; Z output will be ellipsoidal")

    # Determine which frames to open
    cap_map = _build_cap_map(session_dir)
    id_to_imgname = {cid: files.get("image", "") for cid, files in cap_map.items()}

    if args.frame:
        frame_names = [args.frame]
    elif args.near_lat is not None and args.near_lon is not None:
        closest_ids = find_closest_frames(
            captures_inp, proj_by_id, to_utm,
            args.near_lat, args.near_lon, n=args.n,
        )
        frame_names = [id_to_imgname[cid] for cid in closest_ids if cid in id_to_imgname]
        print(f"Closest {len(frame_names)} frames to ({args.near_lat}, {args.near_lon}):")
        for fn in frame_names:
            print(f"  {fn}")
    else:
        p.error("Provide --frame or both --near-lat and --near-lon")

    viewer = FrameViewer(
        session_dir, frame_names, Path(args.out),
        captures_inp, proj_by_id, sensor, to_utm, to_wgs84, geoid_sep,
    )
    viewer.run()


if __name__ == "__main__":
    main()
