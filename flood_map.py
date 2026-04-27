#!/usr/bin/env python3
"""
flood_map.py — Flood boundary GPS extraction from field photographs

Segments flood water in a photo using SAM (Segment Anything Model), then
projects the flood boundary to GPS coordinates using DEM ray casting.

USAGE:
    python flood_map.py photo.jpg --dem dem1.tif dem2.tif
    python flood_map.py photos/   --dem dem1.tif dem2.tif --output flood.geojson
    python flood_map.py photo.jpg --no-dem --height 4.0

CONTROLS:
    Left click        — mark flood water (foreground)
    Shift+left click  — mark dry land / background (improves SAM accuracy)
    Space / Enter     — run segmentation from current clicks
    A                 — accept mask and project boundary to GPS coordinates
    C                 — clear clicks and mask, start over
    N / P             — next / previous image
    S                 — save GeoJSON + CSV now
    Q / ESC           — quit (auto-saves on exit)

SEGMENTATION BACKENDS (tried in order, first available is used):
    SAM 2   — best quality   uv pip install sam2
    SAM 1   — good quality   uv pip install segment-anything
    GrabCut — basic, no model download needed (always available via OpenCV)

    SAM 2 downloads model weights (~350 MB) on first run via HuggingFace.
    Use --sam-model to select size: tiny / small (default) / base-plus / large
    Use --sam-checkpoint to load a local SAM 1 .pth file instead.

OUTPUT:
    GeoJSON FeatureCollection — one Polygon per image, load directly in QGIS
    CSV                       — one row per boundary GPS point

DEPENDENCIES:
    uv pip install opencv-python numpy pillow piexif pyproj rasterio sam2
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

from camera_geometry import build_rotation_matrix
from exif_imu import read_gps_imu_from_exif

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: EXIF METADATA
# ─────────────────────────────────────────────────────────────────────────────

def read_metadata(path):
    """Read GPS lat/lon/alt and IMU roll/pitch/yaw from EXIF. Missing → None."""
    return read_gps_imu_from_exif(path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CAMERA INTRINSICS
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics(calib_path, img_w, img_h,
                    focal_mm=4.44, sensor_w_mm=4.614):
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            d = json.load(f)
        K = np.array(d["K"], dtype=np.float64)
        D = np.array(d["D"], dtype=np.float64)
        size = d.get("img_size")
        print(f"[CALIB] Loaded {calib_path}  RMS={d.get('rms','?')} px")
        if size and (size[0] != img_w or size[1] != img_h):
            K = _scale_K(K, size[0], size[1], img_w, img_h)
        return K, D
    px_per_mm = img_w / sensor_w_mm
    fx = focal_mm * px_per_mm
    K = np.array([[fx, 0, img_w / 2],
                  [0, fx, img_h / 2],
                  [0,  0,         1]], dtype=np.float64)
    print(f"[CALIB] Nominal  fx={fx:.1f} px")
    return K, np.zeros((1, 5), dtype=np.float64)


def _scale_K(K, sw, sh, dw, dh):
    K = K.copy()
    K[0, 0] *= dw / sw; K[0, 2] *= dw / sw
    K[1, 1] *= dh / sh; K[1, 2] *= dh / sh
    return K


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DEM LOADING AND RAY–TERRAIN INTERSECTION
# ─────────────────────────────────────────────────────────────────────────────

def load_dem(paths):
    """
    Load one or more GeoTIFF DEM tiles. Returns get_elevation(lon, lat) callable.
    Multiple tiles are queried in order; first non-None result wins.
    """
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        raise ImportError("DEM support requires rasterio: uv pip install rasterio")

    sources = []
    for path in paths:
        src = rasterio.open(path)
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        tf  = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        sources.append((src, tf, src.nodata))
        print(f"[DEM] {Path(path).name}  {src.width}×{src.height} px  "
              f"res={src.res[0]:.1f} m  crs={crs}")

    def get_elevation(lon, lat):
        for src, tf, nodata in sources:
            x, y = tf.transform(lon, lat)
            left, bottom, right, top = src.bounds
            if not (left <= x <= right and bottom <= y <= top):
                continue
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                continue
            val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1)).flat[0]
            if (nodata is not None and val == nodata) or np.isnan(val):
                continue
            return float(val)
        return None

    return get_elevation


def _ray_to_terrain(ray_dir, cam_lat, cam_lon, cam_elev_m, get_elevation,
                    step_m=1.0, max_range_m=500.0):
    """March a unit ENU ray from the camera until it crosses the DEM surface.
    Returns (lat, lon, elev_m, dist_m) or None."""
    proj = Proj(proj="aeqd", lat_0=cam_lat, lon_0=cam_lon, datum="WGS84")

    def gap(t):
        e, n, z = ray_dir[0]*t, ray_dir[1]*t, ray_dir[2]*t
        lon, lat = proj(e, n, inverse=True)
        elev = get_elevation(lon, lat)
        return (z - (elev - cam_elev_m), e, n, z, lon, lat) if elev is not None \
               else (None, e, n, z, lon, lat)

    g_prev, t_prev = None, step_m
    t = step_m
    while t <= max_range_m:
        g, e, n, z, lon, lat = gap(t)
        if g is not None and g_prev is not None and g <= 0.0:
            # Bisect for sub-step accuracy
            lo, hi = t_prev, t
            for _ in range(8):
                mid = (lo + hi) / 2.0
                gm, em, nm, zm, lonm, latm = gap(mid)
                if gm is None:
                    break
                if gm <= 0.0:
                    hi, e, n, z, lon, lat = mid, em, nm, zm, lonm, latm
                else:
                    lo = mid
            elev = get_elevation(lon, lat) or (cam_elev_m + z)
            return lat, lon, elev, float(np.sqrt(e**2 + n**2 + z**2))
        if g is not None:
            g_prev = g
        t_prev = t
        t += step_m
    return None


def pixel_to_gps(pixel_uv, K, R, cam_lat, cam_lon, cam_alt_gps,
                  cam_height_m, get_elevation=None):
    """
    Project a pixel to GPS.

    Uses DEM ray casting if get_elevation is provided (with WGS84→DEM datum
    correction: replace GPS altitude with DEM ground elevation + mount height).
    Falls back to flat-ground plane otherwise.

    Returns (lat, lon, dist_m, mode) or None.
    """
    u, v = pixel_uv
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = R.T @ ray_cam   # ENU direction

    if ray_world[2] > 0:
        return None   # pointing up

    if get_elevation is not None:
        # Datum correction: use DEM ground elevation under camera, not GPS altitude
        ground = get_elevation(cam_lon, cam_lat)
        cam_elev = (ground + cam_height_m) if ground is not None else cam_alt_gps
        result = _ray_to_terrain(ray_world, cam_lat, cam_lon, cam_elev,
                                  get_elevation)
        if result is not None:
            lat, lon, _elev, dist_m = result
            return lat, lon, dist_m, "terrain"

    # Flat-ground fallback
    if abs(ray_world[2]) < 1e-9:
        return None
    lam = -cam_height_m / ray_world[2]
    if lam < 0:
        return None
    ground_enu = np.array([0.0, 0.0, cam_height_m]) + lam * ray_world
    proj = Proj(proj="aeqd", lat_0=cam_lat, lon_0=cam_lon, datum="WGS84")
    lon_g, lat_g = proj(ground_enu[0], ground_enu[1], inverse=True)
    dist_m = float(np.sqrt(ground_enu[0]**2 + ground_enu[1]**2 + cam_height_m**2))
    return lat_g, lon_g, dist_m, "flat"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SEGMENTATION BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

class _SAM2Segmenter:
    """SAM 2 backend — loads from HuggingFace on first use."""

    MODEL_IDS = {
        "tiny":      "facebook/sam2.1-hiera-tiny",
        "small":     "facebook/sam2.1-hiera-small",
        "base-plus": "facebook/sam2.1-hiera-base-plus",
        "large":     "facebook/sam2.1-hiera-large",
    }

    def __init__(self, model_size="small"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model_id = self.MODEL_IDS.get(model_size, self.MODEL_IDS["small"])
        print(f"[SAM2] Loading {model_id} (downloads ~350 MB on first use)…")
        self._predictor = SAM2ImagePredictor.from_pretrained(model_id)
        self._current_image = None
        print("[SAM2] Ready")

    def set_image(self, image_rgb):
        import torch
        with torch.inference_mode():
            self._predictor.set_image(image_rgb)
        self._current_image = image_rgb

    def segment(self, fg_points, bg_points=None):
        """Returns best binary mask (H×W bool) from the given click points."""
        import torch
        pts   = list(fg_points) + (list(bg_points) if bg_points else [])
        labels = [1]*len(fg_points) + [0]*(len(bg_points) if bg_points else 0)
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(pts, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=True,
            )
        return masks[int(np.argmax(scores))].astype(bool)


class _SAM1Segmenter:
    """SAM 1 backend — requires a local checkpoint file."""

    def __init__(self, checkpoint, model_type="vit_h"):
        from segment_anything import sam_model_registry, SamPredictor
        print(f"[SAM1] Loading {checkpoint}…")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self._predictor = SamPredictor(sam)
        self._current_image = None
        print("[SAM1] Ready")

    def set_image(self, image_rgb):
        self._predictor.set_image(image_rgb)
        self._current_image = image_rgb

    def segment(self, fg_points, bg_points=None):
        pts    = list(fg_points) + (list(bg_points) if bg_points else [])
        labels = [1]*len(fg_points) + [0]*(len(bg_points) if bg_points else 0)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array(pts, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True,
        )
        return masks[int(np.argmax(scores))].astype(bool)


class _GrabCutSegmenter:
    """OpenCV GrabCut fallback — no model download required."""

    def set_image(self, image_rgb):
        self._image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def segment(self, fg_points, bg_points=None):
        h, w = self._image_bgr.shape[:2]
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        for x, y in fg_points:
            cv2.circle(mask, (int(x), int(y)), max(w//40, 10), cv2.GC_FGD, -1)
        if bg_points:
            for x, y in bg_points:
                cv2.circle(mask, (int(x), int(y)), max(w//40, 10), cv2.GC_BGD, -1)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (0, 0, w-1, h-1)
        cv2.grabCut(self._image_bgr, mask, rect, bgd, fgd, 5,
                    cv2.GC_INIT_WITH_MASK)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                        True, False)


def load_segmenter(sam_checkpoint=None, sam_model_size="small"):
    """
    Load the best available segmentation backend.
    Returns (segmenter, backend_name).
    """
    # SAM 2
    try:
        seg = _SAM2Segmenter(model_size=sam_model_size)
        return seg, "SAM2"
    except Exception as e:
        print(f"[SEG] SAM2 not available ({e}); trying SAM1…")

    # SAM 1
    if sam_checkpoint and os.path.exists(sam_checkpoint):
        try:
            seg = _SAM1Segmenter(sam_checkpoint)
            return seg, "SAM1"
        except Exception as e:
            print(f"[SEG] SAM1 failed ({e}); falling back to GrabCut")
    elif sam_checkpoint:
        print(f"[SEG] --sam-checkpoint {sam_checkpoint!r} not found; "
              "falling back to GrabCut")
    else:
        print("[SEG] SAM not available. Falling back to GrabCut.\n"
              "      Install SAM 2 for better results: uv pip install sam2")

    return _GrabCutSegmenter(), "GrabCut"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MASK → GPS BOUNDARY
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_contour(mask, epsilon_px=3.0, min_area_px=1000):
    """
    Extract the largest contour from a binary mask and simplify it.

    epsilon_px  : Douglas-Peucker simplification tolerance in pixels.
    min_area_px : ignore contours smaller than this (noise rejection).

    Returns simplified contour as (N, 2) int array [x, y], or None.
    """
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # Keep the largest contour
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None
    # Simplify with Douglas-Peucker
    simplified = cv2.approxPolyDP(contour, epsilon=epsilon_px, closed=True)
    return simplified.reshape(-1, 2)   # (N, 2) [x, y]


def contour_to_gps(contour, K, R, cam_lat, cam_lon, cam_alt,
                    cam_height_m, get_elevation=None):
    """
    Project every contour point (pixel) to a GPS coordinate.

    Returns list of dicts: {pixel_x, pixel_y, lat, lon, dist_m, mode}.
    Points that don't intersect the ground are silently dropped.
    """
    results = []
    n = len(contour)
    for i, (x, y) in enumerate(contour):
        if i % max(1, n // 20) == 0:
            pct = 100 * i // n
            print(f"\r[PROJECT] {pct:3d}%  ({i}/{n} points)", end="", flush=True)
        result = pixel_to_gps((float(x), float(y)), K, R,
                               cam_lat, cam_lon, cam_alt,
                               cam_height_m, get_elevation)
        if result is None:
            continue
        lat, lon, dist_m, mode = result
        results.append(dict(pixel_x=int(x), pixel_y=int(y),
                            lat=lat, lon=lon, dist_m=dist_m, mode=mode))
    print(f"\r[PROJECT] 100%  ({n}/{n} points) — "
          f"{len(results)} GPS coordinates computed")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def build_geojson_feature(gps_points, meta, image_name, mode_summary):
    """Build a GeoJSON Polygon feature from an ordered list of GPS points."""
    if len(gps_points) < 3:
        return None
    coords = [[p["lon"], p["lat"]] for p in gps_points]
    coords.append(coords[0])   # close the ring
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {
            "image":        image_name,
            "camera_lat":   meta.get("lat"),
            "camera_lon":   meta.get("lon"),
            "camera_alt_m": meta.get("alt"),
            "n_points":     len(gps_points),
            "mode":         mode_summary,
        },
    }


def save_outputs(features, output_geojson, output_csv):
    if not features:
        print("[SAVE] No flood boundaries to save.")
        return

    # GeoJSON
    gj = {"type": "FeatureCollection", "features": features}
    with open(output_geojson, "w") as f:
        json.dump(gj, f, indent=2)
    print(f"[SAVE] GeoJSON → {output_geojson}  ({len(features)} polygon(s))")

    # CSV (all boundary points, labelled by image)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pixel_x", "pixel_y",
                                           "lat", "lon", "dist_m", "mode"])
        w.writeheader()
        for feat in features:
            img_name = feat["properties"]["image"]
            coords   = feat["geometry"]["coordinates"][0][:-1]   # drop closing pt
            props    = feat["properties"]
            for lon, lat in coords:
                w.writerow({"image": img_name,
                             "pixel_x": "", "pixel_y": "",
                             "lat": lat, "lon": lon,
                             "dist_m": "", "mode": props["mode"]})
    print(f"[SAVE] CSV    → {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: INTERACTIVE TOOL
# ─────────────────────────────────────────────────────────────────────────────

class FloodMapTool:
    """
    Interactive flood boundary mapper.

    Workflow per image:
      1. Click flood water pixels (left click) and optionally dry land
         (Shift+left click) to guide segmentation.
      2. Press Space to run the segmentation model.
      3. Inspect the mask overlay. Press C to redo, or A to accept.
      4. On accept, each boundary pixel is projected to GPS via DEM ray
         casting. The result is overlaid on the image.
      5. Navigate to the next image with N, or save and quit with S/Q.
    """

    OVERLAY_ALPHA = 0.35   # flood mask opacity
    CONTOUR_COLOR = (0, 200, 255)
    FG_COLOR      = (0, 200, 255)   # foreground click dots
    BG_COLOR      = (50, 50, 220)   # background click dots

    def __init__(self, images, K, D, get_elevation, cam_height_m,
                 segmenter, backend_name,
                 output_geojson="flood_boundary.geojson",
                 output_csv="flood_boundary.csv"):
        self.images        = images
        self.K             = K
        self.D             = D
        self.get_elevation = get_elevation
        self.height        = cam_height_m
        self.segmenter     = segmenter
        self.backend       = backend_name
        self.out_geojson   = output_geojson
        self.out_csv       = output_csv

        self.img_idx    = 0
        self.features   = []          # completed GeoJSON features
        self.window     = "Flood Boundary Mapper"

        self._reset_state()

    def _reset_state(self):
        self.fg_pts  = []    # foreground click coords [(x, y), …]
        self.bg_pts  = []    # background click coords
        self.mask    = None  # current segmentation mask (H×W bool)
        self.contour = None  # simplified pixel contour (N, 2)
        self.gps_pts = []    # projected GPS boundary points
        self.phase   = "clicking"   # clicking → segmented → projected

    # ── Image preparation ─────────────────────────────────────────────────────

    def _current_entry(self):
        return self.images[self.img_idx]

    def _display_image(self):
        """Return a clean undistorted copy of the current image."""
        img = self._current_entry()["img"]
        h, w = img.shape[:2]
        K_new, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), alpha=0)
        undist = cv2.undistort(img, self.K, self.D, None, K_new)
        x, y, cw, ch = roi
        return undist[y:y+ch, x:x+cw].copy(), K_new

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        disp, _ = self._display_image()
        img_h, img_w = disp.shape[:2]
        entry = self._current_entry()
        name  = Path(entry["path"]).name
        meta  = entry["meta"]

        # Flood mask overlay
        if self.mask is not None:
            overlay = disp.copy()
            overlay[self.mask] = (
                overlay[self.mask].astype(np.float32) * (1 - self.OVERLAY_ALPHA)
                + np.array([255, 140, 0], dtype=np.float32) * self.OVERLAY_ALPHA
            ).astype(np.uint8)
            disp = overlay

        # Boundary contour
        if self.contour is not None:
            cv2.polylines(disp, [self.contour.reshape(-1, 1, 2)],
                          isClosed=True, color=self.CONTOUR_COLOR, thickness=2)

        # GPS boundary points (projected)
        for pt in self.gps_pts:
            px, py = pt["pixel_x"], pt["pixel_y"]
            cv2.circle(disp, (px, py), 3, (0, 255, 80), -1)

        # Click points
        for x, y in self.fg_pts:
            cv2.drawMarker(disp, (x, y), self.FG_COLOR,
                           cv2.MARKER_CROSS, 14, 2)
        for x, y in self.bg_pts:
            cv2.drawMarker(disp, (x, y), self.BG_COLOR,
                           cv2.MARKER_TILTED_CROSS, 14, 2)

        # GPS summary if projected
        if self.phase == "projected" and self.gps_pts:
            lats = [p["lat"] for p in self.gps_pts]
            lons = [p["lon"] for p in self.gps_pts]
            summary = (f"{len(self.gps_pts)} boundary pts  "
                       f"lat [{min(lats):.5f}, {max(lats):.5f}]  "
                       f"lon [{min(lons):.5f}, {max(lons):.5f}]")
            cv2.putText(disp, summary, (10, img_h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

        # Status bar
        gps_s  = f"{meta['lat']:.5f},{meta['lon']:.5f}" \
                 if meta["lat"] else "no GPS"
        imu_s  = (f"Y={meta['yaw_deg']:.1f}° P={meta['pitch_deg']:.1f}°"
                  if meta["yaw_deg"] is not None else "no IMU")
        dem_s  = "DEM✓" if self.get_elevation else "no DEM"
        phase_hints = {
            "clicking":  "L-click=water  Shift+click=land  Space=segment",
            "segmented": "A=accept→GPS  C=redo  Space=re-segment",
            "projected": "S=save  N=next  C=redo  Q=quit",
        }
        line1 = (f"{name}  [{self.img_idx+1}/{len(self.images)}]  "
                 f"{gps_s}  {imu_s}  {dem_s}  [{self.backend}]")
        line2 = phase_hints[self.phase]
        cv2.putText(disp, line1, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(disp, line2, (10, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 220, 255), 1)

        cv2.imshow(self.window, disp)

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _run_segmentation(self):
        if not self.fg_pts:
            print("[SEG] Add at least one foreground click first.")
            return
        entry = self._current_entry()
        img_rgb = cv2.cvtColor(entry["img"], cv2.COLOR_BGR2RGB)
        print(f"[SEG] Segmenting with {self.backend}…")
        self.segmenter.set_image(img_rgb)
        self.mask    = self.segmenter.segment(self.fg_pts, self.bg_pts or None)
        self.contour = mask_to_contour(self.mask)
        self.phase   = "segmented"
        n_px = int(self.mask.sum())
        n_ct = len(self.contour) if self.contour is not None else 0
        print(f"[SEG] Mask: {n_px} px  contour: {n_ct} pts")
        self._draw()

    # ── GPS projection ─────────────────────────────────────────────────────────

    def _project_boundary(self):
        if self.contour is None:
            print("[PROJECT] No contour to project. Run segmentation first.")
            return
        entry = self._current_entry()
        meta  = entry["meta"]
        if meta["lat"] is None:
            print("[PROJECT] No GPS in this image — cannot project to GPS.")
            return
        if meta["yaw_deg"] is None:
            print("[PROJECT] No IMU in this image — cannot project to GPS.")
            return

        R = build_rotation_matrix(meta["yaw_deg"],
                                   meta["pitch_deg"] or 0.0,
                                   meta["roll_deg"]  or 0.0)
        # Use undistorted K for projection
        _, K_undist = self._display_image()

        self.gps_pts = contour_to_gps(
            self.contour, K_undist, R,
            meta["lat"], meta["lon"], meta["alt"] or 0.0,
            self.height, self.get_elevation,
        )

        if not self.gps_pts:
            print("[PROJECT] No boundary points projected successfully.")
            return

        modes = set(p["mode"] for p in self.gps_pts)
        mode_s = "+".join(sorted(modes))
        feat = build_geojson_feature(self.gps_pts, meta,
                                      Path(entry["path"]).name, mode_s)
        if feat:
            # Replace any previous feature for this image
            self.features = [f for f in self.features
                              if f["properties"]["image"]
                              != Path(entry["path"]).name]
            self.features.append(feat)

        self.phase = "projected"
        self._draw()

    # ── Navigation and save ───────────────────────────────────────────────────

    def _go_to(self, idx):
        self.img_idx = idx % len(self.images)
        self._reset_state()
        self._draw()

    def save(self):
        save_outputs(self.features, self.out_geojson, self.out_csv)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, _param):
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            if shift:
                self.bg_pts.append((x, y))
                print(f"[CLICK] Background ({x},{y})")
            else:
                self.fg_pts.append((x, y))
                print(f"[CLICK] Foreground ({x},{y})")
            self._draw()

    def run(self):
        print(f"\n[TOOL] Flood Boundary Mapper  ({self.backend})")
        print("  L-click          : mark flood water")
        print("  Shift+L-click    : mark dry land (optional, improves accuracy)")
        print("  Space / Enter    : run segmentation")
        print("  A                : accept mask → project to GPS")
        print("  C                : clear and start over")
        print("  N / P            : next / previous image")
        print("  S                : save GeoJSON + CSV")
        print("  Q / ESC          : quit (auto-saves)\n")

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 800)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._draw()

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord(" "), 13):   # Space or Enter
                self._run_segmentation()
            elif key == ord("a") and self.phase == "segmented":
                self._project_boundary()
            elif key == ord("c"):
                self._reset_state()
                self._draw()
            elif key == ord("n"):
                self._go_to(self.img_idx + 1)
            elif key == ord("p"):
                self._go_to(self.img_idx - 1)
            elif key == ord("s"):
                self.save()

        cv2.destroyAllWindows()
        self.save()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(paths, name_filter="*NIR-OFF*"):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    found = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  [WARN] Not found: {p}  (quote paths with spaces)")
            continue
        if p.is_dir():
            matches = sorted(
                x for x in p.rglob("*")
                if x.suffix.lower() in exts
                and fnmatch.fnmatch(x.name, name_filter)
            )
            if not matches:
                print(f"  [WARN] No files matching {name_filter!r} under {p}")
            found.extend(matches)
        elif p.suffix.lower() in exts:
            found.append(p)
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Flood boundary GPS extraction using SAM + DEM ray casting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Image files or directories to process")
    parser.add_argument("--dem",   nargs="+", default=None, metavar="DEM",
                        help="GeoTIFF DEM file(s) for terrain-accurate projection")
    parser.add_argument("--calib", default=None,
                        help="calibration.json from camera_calibration.py")
    parser.add_argument("--height", type=float, default=5.0,
                        help="Camera mounting height above ground in metres (default 5.0)")
    parser.add_argument("--filter", default="*NIR-OFF*",
                        help="Filename glob for directory search (default '*NIR-OFF*')")
    parser.add_argument("--output", default="flood_boundary.geojson",
                        help="Output GeoJSON file (default flood_boundary.geojson)")
    parser.add_argument("--sam-model", default="small",
                        choices=["tiny", "small", "base-plus", "large"],
                        help="SAM 2 model size (default small)")
    parser.add_argument("--sam-checkpoint", default=None,
                        help="Path to a local SAM 1 .pth checkpoint file")
    parser.add_argument("--no-dem", action="store_true",
                        help="Skip DEM; use flat-ground projection")
    args = parser.parse_args()

    # ── DEM ───────────────────────────────────────────────────────────────────
    get_elevation = None
    if not args.no_dem and args.dem:
        missing = [p for p in args.dem if not os.path.exists(p)]
        if missing:
            print(f"[WARN] DEM file(s) not found: {missing}")
        dem_paths = [p for p in args.dem if os.path.exists(p)]
        if dem_paths:
            print("\n=== Loading DEM ===")
            get_elevation = load_dem(dem_paths)
    elif not args.no_dem and not args.dem:
        print("[INFO] No --dem provided. Using flat-ground projection.\n"
              "       Pass --dem path/to/dem.tif for terrain-accurate results.")

    # ── Images ────────────────────────────────────────────────────────────────
    print(f"\n=== Loading images (filter: {args.filter!r}) ===")
    paths = _collect_images(args.inputs, args.filter)
    if not paths:
        sys.exit("[ERROR] No images found.")

    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [WARN] Could not read {p.name}")
            continue
        meta = read_metadata(str(p))
        if meta["lat"] is None:
            print(f"  [SKIP] {p.name} — no GPS")
            continue
        if meta["yaw_deg"] is None:
            print(f"  [SKIP] {p.name} — no IMU orientation")
            continue
        images.append({"path": str(p), "img": img, "meta": meta})
        print(f"  {p.name}  GPS=({meta['lat']:.5f},{meta['lon']:.5f})  "
              f"alt={meta['alt']:.1f}m  "
              f"Y={meta['yaw_deg']:.1f}° P={meta['pitch_deg']:.1f}°")

    if not images:
        sys.exit("[ERROR] No images with both GPS and IMU metadata found.")
    print(f"[LOAD] {len(images)} usable image(s)")

    # ── Intrinsics ────────────────────────────────────────────────────────────
    h, w = images[0]["img"].shape[:2]
    K, D = load_intrinsics(args.calib, w, h)

    # ── Segmenter ─────────────────────────────────────────────────────────────
    print("\n=== Loading segmentation model ===")
    segmenter, backend = load_segmenter(args.sam_checkpoint, args.sam_model)

    # ── Run tool ──────────────────────────────────────────────────────────────
    output_csv = args.output.replace(".geojson", ".csv")
    tool = FloodMapTool(
        images=images,
        K=K, D=D,
        get_elevation=get_elevation,
        cam_height_m=args.height,
        segmenter=segmenter,
        backend_name=backend,
        output_geojson=args.output,
        output_csv=output_csv,
    )
    tool.run()


if __name__ == "__main__":
    main()
