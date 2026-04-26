"""
Convert a Pix4DCatch OPF scan directory to LAS/LAZ, DEM GeoTIFF, and camera-pose CSV.

Outputs (written to --out-dir, named after the scan timestamp):
  <name>.laz                 -- point cloud, NAD83(2011) / UTM Zone 18N (EPSG:6347)
  <name>_dem.tif             -- gridded DEM GeoTIFF at --dem-res (default 0.05 m)
  <name>_camera_poses.csv    -- per-frame GPS + yaw/pitch/roll, GCP-ready

Usage:
  python scripts/pix4d_to_las_dem.py <scan_dir> [--out-dir DIR] [--dem-res M]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


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
    Parse pointcloud.gltf + pointcloud.bin.

    Returns:
        xyz_local : (N, 3) float64 -- local scene coordinates
        rgb_uint8 : (N, 3) uint8   -- colours 0-255
    """
    legacy = scan_dir / "point_clouds" / "legacy"
    gltf_path = legacy / "pointcloud.gltf"
    bin_path = legacy / "pointcloud.bin"

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
    rgb_norm = arr[:, 3:6]
    rgb_uint8 = (rgb_norm * 255).clip(0, 255).astype(np.uint8)

    return xyz_local, rgb_uint8


def _local_to_utm(xyz_local: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """utm_xyz = local_xyz - shift  (shift is base_to_canonical.shift)."""
    return xyz_local - shift


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


def process_scan(scan_dir: Path, out_dir: Path, dem_res: float, dsm_method: str = "linear"):
    scan_name = scan_dir.name
    print(f"\nProcessing: {scan_name}")

    out_dir.mkdir(parents=True, exist_ok=True)

    shift, crs_wkt = _load_scene_reference_frame(scan_dir)
    print(f"  CRS: NAD83(2011)/UTM18N  shift: {shift.tolist()}")

    xyz_local, rgb = _read_gltf_pointcloud(scan_dir)
    print(f"  Points read: {len(xyz_local):,}")

    xyz_utm = _local_to_utm(xyz_local, shift)

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
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: {scan_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    process_scan(scan_dir, Path(args.out_dir).resolve(), args.dem_res, args.dsm_method)
    print("\nDone.")


if __name__ == "__main__":
    main()
