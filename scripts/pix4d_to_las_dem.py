"""
Convert a Pix4DCatch OPF scan directory to LAS/LAZ, DEM GeoTIFF, and camera-pose CSV.

Outputs (written to --out-dir, named after the scan timestamp):
  <name>.laz                 -- point cloud, NAD83(2011) / UTM Zone 18N (EPSG:6347)
  <name>_dem.tif             -- gridded DEM GeoTIFF at --dem-res (default 0.05 m)
  <name>_camera_poses.csv    -- per-frame GPS + yaw/pitch/roll, GCP-ready

Usage:
  python scripts/pix4d_to_las_dem.py <scan_dir> [--out-dir DIR] [--dem-res M] [--source depth|cloud]

Point source (--source):
  depth (default) -- built from the frames' LiDAR depth maps and their OPF poses
                     (projected_input_cameras.json), the same geometry the frames and
                     the GCP tools use. Agrees with RTK-surveyed points to a few cm.
  cloud           -- the point cloud Pix4DCatch stores. Checked against the depth maps
                     on our scans: the newer opf_format clouds sit about 0.35-0.4 m low,
                     and a legacy-format cloud was tens of metres off (it is in the
                     phone's session frame, not the scene frame). Use only to compare.
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _read_geoid_separation(scan_dir: Path) -> float | None:
    """
    Return mean geoid undulation N (metres) from rtkGPS.csv, or None if unavailable.

    N = GPSAltitude - ReferenceAltitude (ellipsoidal minus orthometric).
    To convert point cloud Z from ellipsoidal to orthometric: Z_ortho = Z_ellip - N.
    For example, with N ≈ -34 m, orthometric = ellipsoidal + 34 m.
    """
    rtk_path = scan_dir / "geolocations" / "rtkGPS.csv"
    if not rtk_path.exists():
        return None
    seps = []
    with open(rtk_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                seps.append(float(row["GeoidSeparation"]))
            except (KeyError, ValueError):
                continue
    return float(np.mean(seps)) if seps else None


def _load_scene_reference_frame(scan_dir: Path):
    """Return (shift_xyz, crs_wkt) from scene_reference_frame.json."""
    candidates = [
        scan_dir / "opf_files" / "scene_reference_frame.json",
        scan_dir / "point_clouds" / "legacy" / "scene_reference_frame.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            shift = np.array(d["base_to_canonical"]["shift"], dtype=np.float64)
            crs_wkt = d["crs"]["definition"]
            return shift, crs_wkt
    raise FileNotFoundError(f"scene_reference_frame.json not found under {scan_dir}")


def _read_gltf_pointcloud(scan_dir: Path):
    """
    Parse the OPF point cloud. Supports two formats emitted by Pix4DCatch:

    New (2025+): opf_format/pcl.gltf + separate positions.glbin / colors.glbin
        POSITION accessor → VEC3 float32, COLOR_0 accessor → VEC4 uint8
    Legacy:      legacy/pointcloud.gltf + pointcloud.bin
        Single interleaved buffer: XYZRGB as 6 × float32 per point (stride 24)

    Returns:
        xyz_local : (N, 3) float64 -- local scene coordinates
        rgb_uint8 : (N, 3) uint8   -- colours 0-255
    """
    opf_gltf = scan_dir / "point_clouds" / "opf_format" / "pcl.gltf"
    leg_gltf  = scan_dir / "point_clouds" / "legacy" / "pointcloud.gltf"

    if opf_gltf.exists():
        return _read_opf_pointcloud(opf_gltf)
    if leg_gltf.exists():
        return _read_legacy_pointcloud(leg_gltf)
    raise FileNotFoundError(
        f"No point cloud found under {scan_dir / 'point_clouds'}. "
        "Expected opf_format/pcl.gltf or legacy/pointcloud.gltf."
    )


def _read_opf_pointcloud(gltf_path: Path):
    """New OPF format: positions.glbin (VEC3 float32) + colors.glbin (VEC4 uint8)."""
    with open(gltf_path) as f:
        gltf = json.load(f)

    pc_dir = gltf_path.parent
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]

    def _read_accessor(acc_idx: int, dtype) -> np.ndarray:
        acc = gltf["accessors"][acc_idx]
        n = acc["count"]
        bv = gltf["bufferViews"][acc["bufferView"]]
        uri = gltf["buffers"][bv["buffer"]]["uri"]
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        with open(pc_dir / uri, "rb") as f:
            f.seek(offset)
            raw = f.read(length)
        components = {"VEC2": 2, "VEC3": 3, "VEC4": 4}.get(acc["type"], 1)
        return np.frombuffer(raw, dtype=dtype).reshape(n, components)

    xyz_local = _read_accessor(attrs["POSITION"], np.float32).astype(np.float64)
    rgba      = _read_accessor(attrs["COLOR_0"],  np.uint8)
    rgb_uint8 = rgba[:, :3]
    return xyz_local, rgb_uint8


def _read_legacy_pointcloud(gltf_path: Path):
    """Legacy format: single interleaved XYZRGB buffer (6 × float32, stride 24)."""
    bin_path = gltf_path.parent / "pointcloud.bin"
    with open(gltf_path) as f:
        gltf = json.load(f)

    n_points = gltf["accessors"][0]["count"]
    stride = gltf["bufferViews"][0]["byteStride"]  # 24 bytes: 6 × float32
    expected_bytes = n_points * stride

    with open(bin_path, "rb") as f:
        raw = f.read(expected_bytes)

    if len(raw) < expected_bytes:
        raise ValueError(
            f"pointcloud.bin too small: got {len(raw)} bytes, expected {expected_bytes}"
        )

    arr = np.frombuffer(raw, dtype=np.float32).reshape(n_points, stride // 4)
    xyz_local = arr[:, :3].astype(np.float64)
    rgb_norm  = arr[:, 3:6]
    rgb_uint8 = (rgb_norm * 255).clip(0, 255).astype(np.uint8)
    return xyz_local, rgb_uint8


def _depth_pointcloud(scan_dir: Path, frame_step: int = 1, pixel_step: int = 4,
                      min_depth_m: float = 0.3, max_depth_m: float = 8.0):
    """
    Points from every frame_step-th frame's depth map (confidence 2 only, every
    pixel_step-th depth pixel), placed with the frame's OPF pose, in the scene's
    local frame (like the stored clouds, so _local_to_utm applies), with colours
    sampled from the frame. Frames are read in their stored orientation (EXIF ignored),
    which is the one the depth map and intrinsics refer to.
    """
    import cv2
    opf = scan_dir / "opf_files"
    inp = json.loads((opf / "input_cameras.json").read_text())
    sensor = inp["sensors"][0]
    f = float(sensor["internals"]["focal_length_px"])
    cx, cy = (float(v) for v in sensor["internals"]["principal_point_px"])
    img_w = int(sensor["image_size_px"][0])
    proj = {c["id"]: c for c in json.loads((opf / "projected_input_cameras.json").read_text())["captures"]}
    uri = {c["id"]: c["uri"].replace("../images/", "")
           for c in json.loads((opf / "camera_list.json").read_text())["cameras"]}

    xyz, rgb, used = [], [], 0
    for cap in inp["captures"][::max(1, frame_step)]:
        pc = proj.get(cap["id"])
        if pc is None or "geolocation" not in pc or "position" not in pc["geolocation"]:
            continue
        image = depth = None
        for cam in cap.get("cameras", []):
            if "Image" in uri.get(cam["id"], ""):
                image = uri[cam["id"]]
            dm = cam.get("extensions", {}).get("PIX4D_input_depth_map", {}).get("id")
            if dm in uri:
                depth = uri[dm]
        if image is None or depth is None:
            continue
        d = cv2.imread(str(scan_dir / "images" / depth), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(scan_dir / "images" / depth.replace("DepthMap", "Confidence")), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(str(scan_dir / "images" / image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if d is None or d.ndim != 2 or img is None:
            continue
        ds = d.shape[1] / img_w
        vv, uu = np.mgrid[0:d.shape[0]:pixel_step, 0:d.shape[1]:pixel_step]
        z = d[vv, uu].astype(np.float64)
        ok = (z >= min_depth_m) & (z <= max_depth_m)
        if conf is not None and conf.shape == d.shape:
            ok &= conf[vv, uu] >= 2
        u, v, z = uu[ok] / ds, vv[ok] / ds, z[ok]
        cam_pts = np.c_[(u - cx) / f * z, -(v - cy) / f * z, -z]       # OPF camera: x right, y up, z back
        R = _rotation_opf(*pc["orientation"]["angles_deg"])
        xyz.append(np.asarray(pc["geolocation"]["position"], dtype=np.float64) + cam_pts @ R.T)
        ui = np.clip(u.astype(int), 0, img.shape[1] - 1)
        vi = np.clip(v.astype(int), 0, img.shape[0] - 1)
        rgb.append(img[vi, ui, ::-1])
        used += 1
    if not xyz:
        raise RuntimeError(f"No readable depth maps with poses under {scan_dir}")
    print(f"  Depth maps used: {used}")
    return np.vstack(xyz), np.vstack(rgb).astype(np.uint8)


def _rotation_opf(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    """OPF camera->world rotation R = Rx(omega) Ry(phi) Rz(kappa) (as aruco_gcp/gcp_from_pix4d)."""
    o, p, k = np.radians([omega_deg, phi_deg, kappa_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(o), -np.sin(o)], [0, np.sin(o), np.cos(o)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(k), -np.sin(k), 0], [np.sin(k), np.cos(k), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _local_to_utm(xyz_local: np.ndarray, shift: np.ndarray,
                  geoid_sep: float | None = None) -> np.ndarray:
    """
    Convert local scene XYZ to UTM XYZ.

    utm_xyz = local_xyz - shift  (shift is base_to_canonical.shift).
    If geoid_sep (N, ellipsoidal minus orthometric) is provided, the Z column
    is corrected from ellipsoidal to NAVD88-approximate orthometric:
        Z_ortho = Z_ellip - N
    """
    xyz_utm = xyz_local - shift
    if geoid_sep is not None:
        xyz_utm = xyz_utm.copy()
        xyz_utm[:, 2] -= geoid_sep
    return xyz_utm


def _write_laz(xyz_utm: np.ndarray, rgb: np.ndarray, crs_wkt: str, out_path: Path):
    try:
        import laspy
    except ImportError:
        print("ERROR: laspy not installed. Run: pip install laspy[lazrs]", file=sys.stderr)
        sys.exit(1)

    header = laspy.LasHeader(point_format=2, version="1.4")
    header.offsets = xyz_utm.min(axis=0)
    header.scales = np.array([1e-4, 1e-4, 1e-4])

    las = laspy.LasData(header=header)
    las.x = xyz_utm[:, 0]
    las.y = xyz_utm[:, 1]
    las.z = xyz_utm[:, 2]
    las.red = (rgb[:, 0].astype(np.uint16) * 256)
    las.green = (rgb[:, 1].astype(np.uint16) * 256)
    las.blue = (rgb[:, 2].astype(np.uint16) * 256)

    try:
        from pyproj import CRS
        crs_obj = CRS.from_wkt(crs_wkt)
        las.header.add_crs(crs_obj)
    except Exception as exc:
        print(f"  Warning: could not embed CRS into LAZ ({exc})", file=sys.stderr)

    las.write(str(out_path))
    print(f"  Written: {out_path}  ({len(xyz_utm):,} points)")


def _grid_max_z(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                x_min: float, y_max: float, rows: int, cols: int,
                resolution: float) -> np.ndarray:
    """Bin points into cells and keep the maximum Z per cell."""
    col_idx = np.floor((x - x_min) / resolution).astype(np.int64)
    row_idx = np.floor((y_max - y) / resolution).astype(np.int64)

    # Clip to valid grid range (edge-padded points can land one cell outside)
    mask = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
    col_idx, row_idx, z = col_idx[mask], row_idx[mask], z[mask]

    dem = np.full((rows, cols), np.nan, dtype=np.float64)
    flat = row_idx * cols + col_idx
    order = np.argsort(flat)
    flat, z = flat[order], z[order]

    unique_cells, first = np.unique(flat, return_index=True)
    last = np.concatenate([first[1:], [len(flat)]])
    for u, f, l in zip(unique_cells, first, last):
        r, c = divmod(int(u), cols)
        dem[r, c] = z[f:l].max()

    return dem.astype(np.float32)


def _write_dem(xyz_utm: np.ndarray, crs_wkt: str, out_path: Path,
               resolution: float, method: str = "linear"):
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as e:
        print(f"ERROR: missing dependency for DEM export ({e}). "
              "Run: pip install rasterio", file=sys.stderr)
        sys.exit(1)

    x, y, z = xyz_utm[:, 0], xyz_utm[:, 1], xyz_utm[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Pad by one cell to avoid edge artefacts
    x_min -= resolution
    x_max += resolution
    y_min -= resolution
    y_max += resolution

    cols = int(np.ceil((x_max - x_min) / resolution)) + 1
    rows = int(np.ceil((y_max - y_min) / resolution)) + 1

    print(f"  Gridding DSM ({rows}×{cols} px at {resolution} m, method={method}) …")

    if method == "max":
        dem = _grid_max_z(x, y, z, x_min, y_max, rows, cols, resolution)
    else:
        try:
            from scipy.interpolate import griddata
        except ImportError as e:
            print(f"ERROR: scipy required for method=linear ({e}). "
                  "Run: pip install scipy  or use --dsm-method max", file=sys.stderr)
            sys.exit(1)
        gx = np.linspace(x_min, x_min + (cols - 1) * resolution, cols)
        gy = np.linspace(y_max, y_max - (rows - 1) * resolution, rows)
        grid_x, grid_y = np.meshgrid(gx, gy)
        dem = griddata(
            points=np.column_stack([x, y]),
            values=z,
            xi=(grid_x, grid_y),
            method=method,
        ).astype(np.float32)

    transform = from_origin(x_min, y_max, resolution, resolution)
    nodata = np.finfo(np.float32).min

    from pyproj import CRS
    crs_obj = CRS.from_wkt(crs_wkt)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=crs_obj,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dem_out = np.where(np.isnan(dem), nodata, dem)
        dst.write(dem_out, 1)

    print(f"  Written: {out_path}  ({rows}×{cols} px, {resolution} m/px)")


def _write_camera_poses(scan_dir: Path, out_path: Path):
    """
    Export per-frame GPS + orientation to CSV.

    Columns: frame_id, lat_deg, lon_deg, alt_ellipsoid_m,
             sigma_h_m, sigma_v_m,
             yaw_deg, pitch_deg, roll_deg,
             sigma_yaw_deg, sigma_pitch_deg, sigma_roll_deg
    """
    candidates = [
        scan_dir / "opf_files" / "input_cameras.json",
        scan_dir / "point_clouds" / "legacy" / "input_cameras.json",
    ]
    cam_path = None
    for p in candidates:
        if p.exists():
            cam_path = p
            break
    if cam_path is None:
        print("  Warning: input_cameras.json not found; skipping camera pose export.",
              file=sys.stderr)
        return

    with open(cam_path) as f:
        data = json.load(f)

    rows = []
    for cap in data.get("captures", []):
        geo = cap.get("geolocation", {})
        ori = cap.get("orientation", {})
        coords = geo.get("coordinates", [None, None, None])
        sigmas = geo.get("sigmas", [None, None, None])
        angles = ori.get("angles_deg", [None, None, None])
        angle_sigmas = ori.get("sigmas_deg", [None, None, None])
        # Only captures that have a reference camera also have meaningful pose
        rows.append([
            cap.get("id", ""),
            coords[0], coords[1], coords[2],
            sigmas[0], sigmas[2],
            angles[0], angles[1], angles[2],
            angle_sigmas[0], angle_sigmas[1], angle_sigmas[2],
        ])

    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id",
            "lat_deg", "lon_deg", "alt_ellipsoid_m",
            "sigma_h_m", "sigma_v_m",
            "yaw_deg", "pitch_deg", "roll_deg",
            "sigma_yaw_deg", "sigma_pitch_deg", "sigma_roll_deg",
        ])
        w.writerows(rows)

    print(f"  Written: {out_path}  ({len(rows)} frames)")


def process_scan(scan_dir: Path, out_dir: Path, dem_res: float, dsm_method: str = "linear",
                 source: str = "depth", frame_step: int = 1, pixel_step: int = 4):
    scan_name = scan_dir.name
    print(f"\nProcessing: {scan_name}")

    out_dir.mkdir(parents=True, exist_ok=True)

    shift, crs_wkt = _load_scene_reference_frame(scan_dir)
    print(f"  CRS: NAD83(2011)/UTM18N  shift: {shift.tolist()}")

    geoid_sep = _read_geoid_separation(scan_dir)
    if geoid_sep is not None:
        print(f"  Geoid separation N={geoid_sep:.3f} m  → Z corrected to orthometric")
    else:
        print("  Warning: rtkGPS.csv not found; Z remains ellipsoidal", file=sys.stderr)

    if source == "depth":
        xyz_local, rgb = _depth_pointcloud(scan_dir, frame_step=frame_step, pixel_step=pixel_step)
        print(f"  Points from depth maps: {len(xyz_local):,}")
    else:
        xyz_local, rgb = _read_gltf_pointcloud(scan_dir)
        print(f"  Points read from the stored cloud: {len(xyz_local):,} "
              "(may be offset; see --source in the module docstring)")

    xyz_utm = _local_to_utm(xyz_local, shift, geoid_sep=geoid_sep)

    laz_path = out_dir / f"{scan_name}.laz"
    _write_laz(xyz_utm, rgb, crs_wkt, laz_path)

    dem_path = out_dir / f"{scan_name}_dem.tif"
    _write_dem(xyz_utm, crs_wkt, dem_path, dem_res, method=dsm_method)

    pose_path = out_dir / f"{scan_name}_camera_poses.csv"
    _write_camera_poses(scan_dir, pose_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan_dir", help="Path to a Pix4DCatch scan directory")
    parser.add_argument("--out-dir", default="output/pix4d",
                        help="Output directory (default: output/pix4d)")
    parser.add_argument("--dem-res", type=float, default=0.05,
                        help="DEM grid resolution in metres (default: 0.05)")
    parser.add_argument("--dsm-method", default="linear",
                        choices=["linear", "nearest", "cubic", "max"],
                        help="Gridding method: linear/nearest/cubic use scipy griddata "
                             "(interpolates gaps); max bins points and keeps highest Z "
                             "per cell — true DSM, no gap-filling (default: linear)")
    parser.add_argument("--source", default="depth", choices=["depth", "cloud"],
                        help="depth: build points from the frames' depth maps and poses (default); "
                             "cloud: use the point cloud Pix4DCatch stored (may be offset)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="with --source depth: use every Nth frame (default 1)")
    parser.add_argument("--pixel-step", type=int, default=4,
                        help="with --source depth: use every Nth depth-map pixel (default 4)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: {scan_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    process_scan(scan_dir, Path(args.out_dir).resolve(), args.dem_res, args.dsm_method,
                 source=args.source, frame_step=args.frame_step, pixel_step=args.pixel_step)
    print("\nDone.")


if __name__ == "__main__":
    main()
