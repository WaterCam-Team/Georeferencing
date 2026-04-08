#!/usr/bin/env python3
"""
Interactive end-to-end workflow for terrain georeferencing and water-boundary export.

Orchestrates (where possible):
  - Camera calibration via camera_calibration.calibrate_camera (or reuse calibration.json)
  - Terrain setup consistent with georeference_terrain.py
  - Optional notes/paths for RTK control and Planet/Pix4D GCP scripts
  - Water outline from a binary mask (e.g. SegFormer output) or manual polygon on the field image

SegFormer: run inference separately in ../segformer_5band (e.g. segment_tiff_5band.py for 5-band
GeoTIFF), then pass the exported mask image (same pixel size as the field RGB if possible).

Usage:
  python water_georef_workflow.py
  python water_georef_workflow.py --output-dir ./session_out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Repo root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from camera_geometry import build_rotation_matrix
from geo_core import camera_elev_from_dem

from georeference_terrain import (
    VERTICAL_EGM96,
    VERTICAL_ELLIPSOID,
    TerrainGeoreferencer,
    convert_camera_elev_to_terrain_datum,
    make_terrain_provider,
    pixel_to_gps_terrain,
    _load_intrinsics,
)

try:
    from georeference_tool import read_gps_from_exif
except ImportError:
    read_gps_from_exif = None  # type: ignore[misc, assignment]

try:
    from camera_calibration import calibrate_camera
except ImportError:
    calibrate_camera = None  # type: ignore[misc, assignment]


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default is not None:
        s = input(f"{msg} [{default}]: ").strip()
        return s if s else default
    return input(f"{msg}: ").strip()


def _prompt_yes(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    s = input(f"{msg} ({d}): ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true")


def _prompt_float(msg: str, default: Optional[float] = None) -> Optional[float]:
    s = _prompt(msg, str(default) if default is not None else "")
    if not s:
        return default
    return float(s)


@dataclass
class WorkflowConfig:
    output_dir: Path
    calibration_path: Path
    field_image: Path
    dem_path: Optional[str]
    las_path: Optional[str]
    las_crs: Optional[int]
    lat: float
    lon: float
    height_above_ground: Optional[float]
    elev: Optional[float]
    camera_elev_datum: str
    terrain_vertical_datum: Optional[str]
    heading: float
    pitch: float
    roll: float
    gcp_csv_path: Optional[str] = None
    rtk_notes_path: Optional[str] = None
    segformer_mask: Optional[str] = None
    boundary_mode: str = "mask"  # mask | manual | skip
    extras: Dict[str, Any] = field(default_factory=dict)


def _save_manifest(cfg: WorkflowConfig, path: Path) -> None:
    d = asdict(cfg)
    d["output_dir"] = str(cfg.output_dir)
    d["calibration_path"] = str(cfg.calibration_path)
    d["field_image"] = str(cfg.field_image)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def _step_calibration(out_dir: Path) -> Path:
    print("\n=== 1. Camera calibration ===\n")
    if _prompt_yes("Is calibration already available (calibration.json)?", default=True):
        p = Path(_prompt("Path to calibration.json", str(out_dir / "calibration.json")))
        if not p.exists():
            raise FileNotFoundError(f"Calibration not found: {p}")
        return p

    if calibrate_camera is None:
        raise RuntimeError("camera_calibration could not be imported.")

    print(
        "You need a folder of checkerboard images (see docs/GEOREFERENCING_PROCEDURE.md).\n"
        "Enter interior corner counts (not squares) and square size in metres."
    )
    img_dir = _prompt("Folder with calibration .jpg/.png images", "./calib_images")
    save_path = out_dir / "calibration.json"
    bw = int(_prompt("Interior corners horizontally (BOARD_W)", "8"))
    bh = int(_prompt("Interior corners vertically (BOARD_H)", "6"))
    sq = float(_prompt("Square size (m)", "0.025"))
    h_str = _prompt("Camera height above ground for georef (m, optional)", "")
    cam_h = float(h_str) if h_str else None

    intrinsics = calibrate_camera(
        image_dir=img_dir,
        save_path=str(save_path),
        show_corners=True,
        camera_height_m=cam_h,
        board_w=bw,
        board_h=bh,
        square_size_m=sq,
    )
    print(f"[OK] Calibration saved: {save_path}  RMS={intrinsics.rms:.4f} px")
    return save_path


def _step_sensor_and_terrain(out_dir: Path) -> Tuple[Path, Optional[str], Optional[str], Optional[int]]:
    print("\n=== 2. Field image and terrain data ===\n")
    field = Path(_prompt("Path to field image (RGB/JPG)", "./field_image.jpg"))
    if not field.exists():
        raise FileNotFoundError(field)

    dem = _prompt("Path to DEM GeoTIFF (or Enter to skip)", "") or None
    las = _prompt("Path to LAS/LAZ (or Enter to skip)", "") or None
    las_crs = None
    if las:
        c = _prompt("LAS EPSG if not in file (or Enter)", "")
        las_crs = int(c) if c.strip() else None

    if not dem and not las:
        raise ValueError("Need at least one of DEM or LAS for terrain georeferencing.")

    return field, dem, las, las_crs


def _step_pose(field_image: Path) -> Tuple[float, float, Optional[float], Optional[float], float, float, float, str, Optional[str]]:
    print("\n=== 3. Camera pose (coarse GPS/IMU — improve later with GCP/RTK) ===\n")
    lat = lon = None
    if read_gps_from_exif and _prompt_yes("Read lat/lon from EXIF (if present)?", default=True):
        gps = read_gps_from_exif(str(field_image))
        lat = gps.get("lat")
        lon = gps.get("lon")
        hd = gps.get("heading")
        if lat is not None and lon is not None:
            print(f"  EXIF: lat={lat}, lon={lon}, heading={hd}")
    if lat is None:
        lat = float(_prompt("Camera latitude (deg)", ""))
    if lon is None:
        lon = float(_prompt("Camera longitude (deg)", ""))

    h_agl = _prompt_float("Height above ground (m) — strongly recommended for terrain", None)
    elev_manual = None
    if h_agl is None and _prompt_yes("Enter absolute camera elevation (m) instead?", default=False):
        elev_manual = _prompt_float("Camera elevation (m)", 0.0)

    heading = float(_prompt("Heading deg (0=N)", "0") or "0")
    pitch = float(_prompt("Pitch deg (negative=down)", "-15") or "-15")
    roll = float(_prompt("Roll deg", "0") or "0")

    print("\nVertical datums (see vertical_datum.py). Common: camera=wgs84_ellipsoid, terrain=egm96 or navd88.")
    cam_datum = _prompt("Camera elev datum", VERTICAL_ELLIPSOID)
    tv = _prompt("Terrain vertical datum (Enter to infer/default)", "").strip() or None

    return lat, lon, h_agl, elev_manual, heading, pitch, roll, cam_datum, tv


def _step_control_extras(out_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    print("\n=== 4. Control / accuracy (optional) ===\n")
    print(
        "Use RTK or surveyed points for true camera position and/or GCP lat/lon.\n"
        "You can save a text/CSV path here for your records; this script does not parse all formats."
    )
    rtk = _prompt("Path to RTK/control notes file (or Enter to skip)", "") or None
    gcp = _prompt("Path to existing GCP CSV from Planet/Pix4D tools (or Enter)", "") or None

    if gcp and _prompt_yes("Open georeference_tool.py for pose refinement from GCPs? (flat model)", default=False):
        print("  Run manually: python georeference_tool.py  (set GCP_CSV in script or press L to load)")
        print(f"  Suggested GCP file: {gcp}")

    if _prompt_yes("Run planet_gcp_match.py now (needs Planet ortho GeoTIFF)?", default=False):
        ortho = _prompt("Planet/Pix4D ortho GeoTIFF path", "")
        out_csv = out_dir / "planet_gcp.csv"
        cmd = [
            sys.executable,
            str(_ROOT / "planet_gcp_match.py"),
            "--field-image",
            str(out_dir / "field_image.jpg"),
            "--planet-tif",
            ortho,
            "--output-csv",
            str(out_csv),
            "--calibration",
            str(out_dir / "calibration.json"),
        ]
        print("  Command:", " ".join(cmd))
        if _prompt_yes("Execute this command?", default=False):
            subprocess.run(cmd, cwd=str(_ROOT), check=False)

    return rtk, gcp if gcp else None


def _step_boundary() -> Tuple[str, Optional[str]]:
    print("\n=== 5. Water boundary ===\n")
    print("  (a) Mask image from SegFormer or other segmentation (same size as field image)")
    print("  (b) Manual polygon: click vertices on the field image, press Enter to finish")
    print("  (c) Skip boundary export (terrain tool only)")
    choice = _prompt("Choose a / b / c", "a").lower()
    if choice == "c":
        return "skip", None
    if choice == "b":
        return "manual", None
    mask_path = _prompt("Path to mask image (nonzero = water)", "")
    return "mask", mask_path or None


def _manual_polygon_filled_mask(image: np.ndarray) -> np.ndarray:
    pts: List[Tuple[int, int]] = []
    done = [False]

    def mouse(e, x, y, *_):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    win = "Water polygon — click vertices, Enter when done, Esc cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse)
    disp = image.copy()
    while not done[0]:
        d = disp.copy()
        if len(pts) > 0:
            cv2.polylines(d, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(d, p, 4, (0, 0, 255), -1)
        cv2.putText(
            d, "Clicks add vertices. Press Enter to close polygon, Esc to cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow(win, d)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(pts) >= 3:
            done[0] = True
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")

    cv2.destroyAllWindows()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def _undistort_mask_like_terrain_georeferencer(mask: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Same undistort + ROI crop as TerrainGeoreferencer.__init__, so (u,v) match tg.K_undist
    and pixel_to_gps_terrain. Uses georeference_tool.undistort when available.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    try:
        from georeference_tool import undistort as _undistort_gt

        und, _ = _undistort_gt(mask, K, D)
        return und
    except ImportError:
        h, w = mask.shape[:2]
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        out = cv2.undistort(mask, K, D, None, K_new)
        x, y, cw, ch = roi
        return out[y : y + ch, x : x + cw]


def _contour_to_latlon(
    binary_undist: np.ndarray,
    tg: TerrainGeoreferencer,
    step: int = 2,
) -> List[Tuple[float, float, float]]:
    """binary_undist: water=255 foreground on undistorted image grid."""
    _, binary = cv2.threshold(binary_undist, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.squeeze(1)
    if pts.ndim != 2:
        return []
    out: List[Tuple[float, float, float]] = []
    for i in range(0, len(pts), step):
        u, v = float(pts[i][0]), float(pts[i][1])
        r = pixel_to_gps_terrain(
            (u, v),
            tg.K_undist,
            tg.R,
            tg.camera_lat,
            tg.camera_lon,
            tg.camera_elev_m,
            tg.get_elevation,
        )
        if r is None:
            continue
        lat, lon, elev, _ = r
        out.append((lon, lat, elev))
    return out


def _build_terrain_georeferencer(
    image: np.ndarray,
    calib_path: str,
    dem_path: Optional[str],
    las_path: Optional[str],
    las_crs: Optional[int],
    lat: float,
    lon: float,
    height_above_ground: Optional[float],
    elev: Optional[float],
    camera_elev_datum: str,
    terrain_vertical_datum_opt: Optional[str],
    heading: float,
    pitch: float,
    roll: float,
) -> TerrainGeoreferencer:
    get_elev, dem_bounds, las_bounds, inferred_td = make_terrain_provider(
        dem_path=dem_path,
        las_path=las_path,
        las_crs_epsg=las_crs,
        las_resolution_m=1.0,
    )
    terrain_vertical_datum = terrain_vertical_datum_opt or inferred_td or VERTICAL_EGM96

    h, w = image.shape[:2]
    K, D = _load_intrinsics(calib_path, w, h)
    R = build_rotation_matrix(heading, pitch, roll)

    elev_from_dem = False
    final_elev: float
    if height_above_ground is not None:
        final_elev = camera_elev_from_dem(get_elev, lat, lon, height_above_ground)
        elev_from_dem = True
    elif elev is not None:
        final_elev = float(elev)
    else:
        raise ValueError("Need --height-above-ground or camera elevation.")

    if not elev_from_dem and terrain_vertical_datum != camera_elev_datum:
        conv, ok = convert_camera_elev_to_terrain_datum(
            lon, lat, final_elev, camera_elev_datum, terrain_vertical_datum
        )
        if ok:
            final_elev = float(conv)

    return TerrainGeoreferencer(
        image=image,
        K=K,
        D=D,
        R=R,
        camera_lat=lat,
        camera_lon=lon,
        camera_elev_m=final_elev,
        get_elevation=get_elev,
    )


def _write_geojson(path: Path, ring: Sequence[Tuple[float, float]]) -> None:
    """ring: list of (lon, lat), closed ring."""
    if len(ring) < 3:
        return
    coords = [list(p) for p in ring]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    feat = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "water_boundary"},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feat, f, indent=2)


def _write_boundary_csv(path: Path, rows: Sequence[Tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "elev_m"])
        for lon, lat, el in rows:
            w.writerow([lon, lat, el])


def run_workflow(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_path = _step_calibration(output_dir)
    field_image, dem, las, las_crs = _step_sensor_and_terrain(output_dir)

    # Copy field image to session for reproducibility
    session_field = output_dir / "field_image.jpg"
    import shutil

    shutil.copy2(field_image, session_field)

    lat, lon, h_agl, elev_m, heading, pitch, roll, cam_datum, tv = _step_pose(session_field)
    rtk_path, gcp_path = _step_control_extras(output_dir)
    mode, mask_path = _step_boundary()

    cfg = WorkflowConfig(
        output_dir=output_dir,
        calibration_path=calib_path,
        field_image=session_field,
        dem_path=dem,
        las_path=las,
        las_crs=las_crs,
        lat=lat,
        lon=lon,
        height_above_ground=h_agl,
        elev=elev_m,
        camera_elev_datum=cam_datum,
        terrain_vertical_datum=tv,
        heading=heading,
        pitch=pitch,
        roll=roll,
        gcp_csv_path=gcp_path,
        rtk_notes_path=rtk_path,
        segformer_mask=mask_path,
        boundary_mode=mode,
        extras={"segformer_hint": "Optional: ../segformer_5band/segment_tiff_5band.py for 5-band TIFF"},
    )
    _save_manifest(cfg, output_dir / "workflow_manifest.json")

    image = cv2.imread(str(session_field))
    if image is None:
        raise RuntimeError("Could not read field image")

    tg = _build_terrain_georeferencer(
        image,
        str(calib_path),
        dem,
        las,
        las_crs,
        lat,
        lon,
        h_agl,
        elev_m,
        cam_datum,
        tv,
        heading,
        pitch,
        roll,
    )

    if mode == "skip":
        print("\n[OK] Manifest saved. Run interactive terrain georeferencing:")
        print(f"  python georeference_terrain.py {session_field} --dem {dem or ''} --las {las or ''} ...")
        if _prompt_yes("Launch georeference_terrain.py interactively now (subprocess)?", default=False):
            cmd = [sys.executable, str(_ROOT / "georeference_terrain.py"), str(session_field)]
            if dem:
                cmd.extend(["--dem", dem])
            if las:
                cmd.extend(["--las", las])
            if las_crs:
                cmd.extend(["--las-crs", str(las_crs)])
            cmd.extend(["--calibration", str(calib_path), "--lat", str(lat), "--lon", str(lon)])
            cmd.extend(["--heading", str(heading), "--pitch", str(pitch), "--roll", str(roll)])
            if h_agl is not None:
                cmd.extend(["--height-above-ground", str(h_agl)])
            elif elev_m is not None:
                cmd.extend(["--elev", str(elev_m)])
            subprocess.run(cmd, cwd=str(_ROOT))
        return 0

    if mode == "manual":
        water_mask_orig = _manual_polygon_filled_mask(image)
    else:
        if not mask_path or not Path(mask_path).exists():
            raise FileNotFoundError("Mask path invalid")
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m_color = cv2.imread(mask_path)
            if m_color is None:
                raise FileNotFoundError(f"Could not read mask image: {mask_path}")
            m = cv2.cvtColor(m_color, cv2.COLOR_BGR2GRAY)
        if m.shape[:2] != image.shape[:2]:
            raise ValueError("Mask size must match field image size")
        _, water_mask_orig = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    K, D = _load_intrinsics(str(calib_path), image.shape[1], image.shape[0])
    und_binary = _undistort_mask_like_terrain_georeferencer(water_mask_orig, K, D)
    ring_ll = _contour_to_latlon(und_binary, tg)
    if len(ring_ll) < 3:
        print("[WARN] Too few valid terrain hits on boundary. Check pose, DEM coverage, and mask.")
        return 1

    geojson_path = output_dir / "water_boundary.geojson"
    csv_path = output_dir / "water_boundary.csv"
    _write_geojson(geojson_path, [(a[0], a[1]) for a in ring_ll])
    _write_boundary_csv(csv_path, ring_ll)
    print(f"\n[OK] Wrote {geojson_path} and {csv_path} ({len(ring_ll)} points).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive water georeferencing workflow (terrain-first)")
    ap.add_argument(
        "--output-dir",
        default="./water_georef_session",
        help="Session directory for calibration copy, manifest, and outputs",
    )
    args = ap.parse_args()
    try:
        return run_workflow(Path(args.output_dir))
    except (KeyboardInterrupt, EOFError):
        print("\n[exit]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
