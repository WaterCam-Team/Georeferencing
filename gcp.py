"""
Ground control point (GCP) system for georeferencing.

Allows the user to provide known (lat, lon [, elev]) for points identified in
the image (pixel u, v). These GCPs are used to:
  1. Refine camera pose (lat, lon, height, heading, pitch, roll) by minimizing
     reprojection error in geographic space.
  2. Optionally fit a thin-plate spline (TPS) warp from pixel (u, v) to (lat, lon)
     so that any click uses the warp when enough GCPs are available, improving
     local accuracy beyond the camera model.

GCP CSV format: label,pixel_u,pixel_v,lat,lon,elev_m
  - elev_m is optional (empty for flat-ground only).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# GCP data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundControlPoint:
    """One GCP: pixel location and known GPS (and optional elevation)."""
    label: str
    pixel_u: float
    pixel_v: float
    lat: float
    lon: float
    elev_m: Optional[float] = None

    def to_dict(self):
        return {
            "label": self.label,
            "pixel_u": self.pixel_u,
            "pixel_v": self.pixel_v,
            "lat": self.lat,
            "lon": self.lon,
            "elev_m": self.elev_m,
        }


def load_gcps(path: str | Path) -> List[GroundControlPoint]:
    """Load GCPs from CSV. Columns: label, pixel_u, pixel_v, lat, lon [, elev_m]."""
    path = Path(path)
    gcps = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                elev = row.get("elev_m", "").strip()
                gcps.append(GroundControlPoint(
                    label=row["label"].strip(),
                    pixel_u=float(row["pixel_u"]),
                    pixel_v=float(row["pixel_v"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    elev_m=float(elev) if elev else None,
                ))
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid GCP row in {path}: {row}") from e
    return gcps


def save_gcps(path: str | Path, gcps: List[GroundControlPoint]) -> None:
    """Save GCPs to CSV."""
    path = Path(path)
    fieldnames = ["label", "pixel_u", "pixel_v", "lat", "lon", "elev_m"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in gcps:
            d = g.to_dict()
            d["elev_m"] = "" if d["elev_m"] is None else d["elev_m"]
            writer.writerow(d)


# ─────────────────────────────────────────────────────────────────────────────
# Pose refinement (minimize lat/lon residuals at GCPs)
# ─────────────────────────────────────────────────────────────────────────────

def refine_pose_from_gcps(
    K: np.ndarray,
    gcps: List[GroundControlPoint],
    cam_lat_0: float,
    cam_lon_0: float,
    height_0: float,
    heading_0: float,
    pitch_0: float,
    roll_0: float,
    *,
    bounds: bool = True,
) -> Tuple[float, float, float, np.ndarray, float, float, float, float]:
    """
    Refine camera pose (lat, lon, height, heading, pitch, roll) by minimizing
    sum of squared (predicted_lat - known_lat, predicted_lon - known_lon) over GCPs.

    Returns (cam_lat, cam_lon, height_m, R_3x3, rms_deg, heading_deg, pitch_deg, roll_deg).
    If refinement fails, returns initial pose and NaN rms.
    """
    from scipy.optimize import least_squares
    from camera_geometry import build_rotation_matrix
    from geo_core import pixel_to_world_flat

    if len(gcps) < 3:
        R0 = build_rotation_matrix(heading_0, pitch_0, roll_0)
        return cam_lat_0, cam_lon_0, height_0, R0, float("nan"), heading_0, pitch_0, roll_0

    def residual(x: np.ndarray) -> np.ndarray:
        cam_lon, cam_lat, height_m, heading_deg, pitch_deg, roll_deg = x
        R = build_rotation_matrix(heading_deg, pitch_deg, roll_deg)
        out = []
        for g in gcps:
            pred = pixel_to_world_flat(
                g.pixel_u, g.pixel_v, K, R, cam_lat, cam_lon, height_m
            )
            if pred is None:
                out.append(1e2)  # large residual if ray misses ground
                out.append(1e2)
            else:
                out.append(pred[0] - g.lat)
                out.append(pred[1] - g.lon)
        return np.array(out, dtype=np.float64)

    x0 = np.array([
        cam_lon_0, cam_lat_0, height_0,
        heading_0, pitch_0, roll_0,
    ], dtype=np.float64)

    # Bounds: position ±~0.002 deg (~200 m); height 0.1–200 m; pitch [-90,0]; heading/roll wide
    if bounds:
        lb = [cam_lon_0 - 0.002, cam_lat_0 - 0.002, 0.1, -360, -90, -180]
        ub = [cam_lon_0 + 0.002, cam_lat_0 + 0.002, 200, 360, 0, 180]
        res = least_squares(
            residual, x0, bounds=(lb, ub),
            loss="soft_l1", f_scale=0.1,
            max_nfev=500,
        )
    else:
        res = least_squares(residual, x0, loss="soft_l1", f_scale=0.1, max_nfev=500)

    if not res.success:
        R0 = build_rotation_matrix(heading_0, pitch_0, roll_0)
        return cam_lat_0, cam_lon_0, height_0, R0, float("nan"), heading_0, pitch_0, roll_0

    cam_lon, cam_lat, height_m, heading_deg, pitch_deg, roll_deg = res.x
    R = build_rotation_matrix(heading_deg, pitch_deg, roll_deg)
    rms_deg = np.sqrt(np.mean(res.fun ** 2))
    return (float(cam_lat), float(cam_lon), float(height_m), R, float(rms_deg),
            float(heading_deg), float(pitch_deg), float(roll_deg))


# ─────────────────────────────────────────────────────────────────────────────
# TPS warp: (u, v) -> (lat, lon) from GCPs
# ─────────────────────────────────────────────────────────────────────────────

def fit_tps_warp(gcps: List[GroundControlPoint]) -> Optional[Callable[[float, float], Optional[Tuple[float, float]]]]:
    """
    Fit a thin-plate spline from pixel (u, v) to (lat, lon) using GCPs.

    Requires at least 6 GCPs for stability. Returns a callable
    warp(u, v) -> (lat, lon) or None if outside support.
    """
    if len(gcps) < 6:
        return None
    try:
        from scipy.interpolate import RBFInterpolator
    except ImportError:
        return None

    pts = np.array([[g.pixel_u, g.pixel_v] for g in gcps], dtype=np.float64)
    lats = np.array([g.lat for g in gcps], dtype=np.float64)
    lons = np.array([g.lon for g in gcps], dtype=np.float64)

    # thin_plate is smooth and passes near the control points
    rbf_lat = RBFInterpolator(pts, lats, kernel="thin_plate_spline", smoothing=0.0)
    rbf_lon = RBFInterpolator(pts, lons, kernel="thin_plate_spline", smoothing=0.0)

    def warp(u: float, v: float) -> Optional[Tuple[float, float]]:
        p = np.array([[u, v]], dtype=np.float64)
        lat = float(rbf_lat(p)[0])
        lon = float(rbf_lon(p)[0])
        return (lat, lon)

    return warp


def gcp_residuals(
    K: np.ndarray,
    R: np.ndarray,
    cam_lat: float,
    cam_lon: float,
    camera_height_m: float,
    gcps: List[GroundControlPoint],
) -> List[Tuple[float, float, float]]:
    """
    Compute per-GCP residuals: (error_lat_deg, error_lon_deg, error_m).
    error_m is approximate horizontal error in metres at that location.
    """
    from pyproj import Proj
    from geo_core import pixel_to_world_flat

    proj = Proj(proj="aeqd", lat_0=cam_lat, lon_0=cam_lon, datum="WGS84")
    out = []
    for g in gcps:
        pred = pixel_to_world_flat(
            g.pixel_u, g.pixel_v, K, R, cam_lat, cam_lon, camera_height_m
        )
        if pred is None:
            out.append((float("nan"), float("nan"), float("nan")))
            continue
        lat_err = pred[0] - g.lat
        lon_err = pred[1] - g.lon
        e, n = proj(g.lon, g.lat)
        e2, n2 = proj(pred[1], pred[0])
        dist_m = np.sqrt((e2 - e) ** 2 + (n2 - n) ** 2)
        out.append((lat_err, lon_err, float(dist_m)))
    return out
