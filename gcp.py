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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Minimum GCPs for pose refinement. The pose has 6 unknowns and each GCP gives
# 2 equations, so 3 points fit exactly with zero redundancy: a wrong point is
# absorbed into the pose and cannot be detected. 4+ gives redundancy.
MIN_GCPS = 4

# Robust-loss knee (soft-L1) in METRES of ground error. The residuals being
# minimised are ground distances in metres, so this is the error size above
# which a point starts to be down-weighted (roughly 3 px at a few hundred m
# range for a typical camera).
GCP_F_SCALE_M = 0.5

# Residual (m) assigned when a GCP's ray misses the ground / is rejected.
MISS_RESIDUAL_M = 1.0e3

# Leave-one-out: flag a GCP if removing it moves any projected GCP by more
# than this many metres.
LOO_SHIFT_FLAG_M = 2.0
# ...and only if it is at least this many times larger than for every other
# GCP (see _flag_leave_one_out).
LOO_DOMINANCE = 2.0

M_PER_DEG_LAT = 111320.0

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

@dataclass
class LeaveOneOut:
    """Effect of removing one GCP and re-solving the pose."""
    label: str
    shift_m: float          # max ground displacement of any GCP pixel between the full and reduced solutions
    heldout_error_m: float  # error at the removed GCP under the reduced solution
    rms_without_m: float    # RMS residual (all remaining GCPs) of the reduced solution
    flagged: bool


@dataclass
class PoseRefinement:
    """Result of refine_pose_with_diagnostics (all distances in metres)."""
    ok: bool
    cam_lat: float
    cam_lon: float
    height_m: float
    R: np.ndarray
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    rms_m: float = float("nan")
    residuals_m: List[float] = field(default_factory=list)  # per GCP, ground distance
    leave_one_out: List[LeaveOneOut] = field(default_factory=list)

    @property
    def flagged_labels(self) -> List[str]:
        return [l.label for l in self.leave_one_out if l.flagged]


def _ground_point_enu(g, K, R, proj, cam_e, cam_n, height_m):
    """Predicted ground (east, north) in the shared aeqd frame, or None."""
    from geo_core import pixel_ray, intersect_flat
    off = intersect_flat(R.T @ pixel_ray(K, g.pixel_u, g.pixel_v), height_m)
    if off is None:
        return None
    return cam_e + off[0], cam_n + off[1]


def _solve(K, gcps, x0, proj, bounds):
    """Robust least squares on ground-distance residuals in METRES."""
    from scipy.optimize import least_squares
    from camera_geometry import build_rotation_matrix

    targets = [proj(g.lon, g.lat) for g in gcps]

    def residual(x: np.ndarray) -> np.ndarray:
        cam_lon, cam_lat, height_m, heading_deg, pitch_deg, roll_deg = x
        R = build_rotation_matrix(heading_deg, pitch_deg, roll_deg)
        cam_e, cam_n = proj(cam_lon, cam_lat)
        out = []
        for g, (te, tn) in zip(gcps, targets):
            pred = _ground_point_enu(g, K, R, proj, cam_e, cam_n, height_m)
            if pred is None:
                out += [MISS_RESIDUAL_M, MISS_RESIDUAL_M]  # ray misses ground
            else:
                out += [pred[0] - te, pred[1] - tn]
        return np.array(out, dtype=np.float64)

    kw = dict(loss="soft_l1", f_scale=GCP_F_SCALE_M, max_nfev=500)
    if bounds:
        # position +-~0.002 deg (~200 m); height 0.1-200 m; pitch [-90,0]; heading/roll wide
        lb = [x0[0] - 0.002, x0[1] - 0.002, 0.1, -360, -90, -180]
        ub = [x0[0] + 0.002, x0[1] + 0.002, 200, 360, 0, 180]
        return least_squares(residual, x0, bounds=(lb, ub), **kw)
    return least_squares(residual, x0, **kw)


def _per_point_m(K, gcps, x, proj):
    from camera_geometry import build_rotation_matrix
    cam_lon, cam_lat, height_m, hd, pt, rl = x
    R = build_rotation_matrix(hd, pt, rl)
    cam_e, cam_n = proj(cam_lon, cam_lat)
    out = []
    for g in gcps:
        te, tn = proj(g.lon, g.lat)
        pred = _ground_point_enu(g, K, R, proj, cam_e, cam_n, height_m)
        out.append(float("nan") if pred is None
                   else float(np.hypot(pred[0] - te, pred[1] - tn)))
    return out


def _rms(vals):
    v = np.asarray(vals, dtype=np.float64)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def _dominant(vals: List[float], i: int) -> bool:
    """vals[i] is finite and at least LOO_DOMINANCE x every other finite value."""
    v = vals[i]
    if not np.isfinite(v):
        return False
    others = [x for j, x in enumerate(vals) if j != i and np.isfinite(x)]
    return all(v >= LOO_DOMINANCE * x for x in others)


def _flag_leave_one_out(ref: PoseRefinement) -> None:
    """
    Set .flagged on the leave-one-out entries. A GCP is suspect when it
    clearly stands out from the others in one of two ways:
      * MISFIT: leaving it out, it is predicted worse than LOO_SHIFT_FLAG_M,
        and the remaining points then fit at under half the full-fit RMS and
        better (by >= LOO_DOMINANCE x) than after removing any other point,
        i.e. this one point explains the misfit.
      * LEVERAGE: removing it moves the solution (max ground shift of any
        GCP pixel) by more than LOO_SHIFT_FLAG_M, dominating the others'.
    Dominance matters because with few GCPs every leave-one-out solution is
    poorly constrained and all points look somewhat bad. With exactly
    MIN_GCPS points each reduced fit is exactly determined, so detection is
    weak; 5-6+ well-spread GCPs are recommended.
    """
    loo = ref.leave_one_out
    shift = [l.shift_m for l in loo]
    rms_wo = [l.rms_without_m for l in loo]
    for i, l in enumerate(loo):
        others = [x for j, x in enumerate(rms_wo) if j != i and np.isfinite(x)]
        misfit = (l.heldout_error_m > LOO_SHIFT_FLAG_M
                  and l.rms_without_m < 0.5 * ref.rms_m
                  and all(l.rms_without_m * LOO_DOMINANCE <= x for x in others))
        leverage = l.shift_m > LOO_SHIFT_FLAG_M and _dominant(shift, i)
        l.flagged = bool(misfit or leverage)


def refine_pose_with_diagnostics(
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
    leave_one_out: bool = True,
) -> PoseRefinement:
    """
    Refine camera pose (lat, lon, height, heading, pitch, roll) from GCPs.

    Minimises ground-distance residuals in METRES (east/north difference
    between each projected GCP pixel and its surveyed position, in a local
    azimuthal-equidistant frame), with a soft-L1 robust loss whose knee is
    GCP_F_SCALE_M metres. Needs at least MIN_GCPS (4) points; with fewer the
    pose is not redundantly determined and an outlier cannot be detected, so
    the initial pose is returned with ok=False.

    Reports the per-GCP residual in metres and a leave-one-out check: each
    GCP is removed in turn, the pose re-solved, and the point is FLAGGED (not
    dropped) when it stands out by misfit or leverage (see
    _flag_leave_one_out). Flat ground is assumed; elev_m is
    ignored (see docs/GEOREFERENCING_PROCEDURE.md).
    """
    from pyproj import Proj
    from camera_geometry import build_rotation_matrix

    R0 = build_rotation_matrix(heading_0, pitch_0, roll_0)
    fail = PoseRefinement(False, cam_lat_0, cam_lon_0, height_0, R0,
                          heading_0, pitch_0, roll_0)
    if len(gcps) < MIN_GCPS:
        log.warning("Pose refinement needs >= %d GCPs (have %d): 3 points give "
                    "no redundancy, so a wrong point cannot be detected.",
                    MIN_GCPS, len(gcps))
        return fail

    proj = Proj(proj="aeqd", lat_0=cam_lat_0, lon_0=cam_lon_0, datum="WGS84")
    x0 = np.array([cam_lon_0, cam_lat_0, height_0, heading_0, pitch_0, roll_0],
                  dtype=np.float64)
    res = _solve(K, gcps, x0, proj, bounds)
    if not res.success:
        return fail

    x = res.x
    per_pt = _per_point_m(K, gcps, x, proj)
    out = PoseRefinement(
        True, float(x[1]), float(x[0]), float(x[2]),
        build_rotation_matrix(x[3], x[4], x[5]),
        float(x[3]), float(x[4]), float(x[5]),
        rms_m=_rms(per_pt), residuals_m=per_pt,
    )

    if leave_one_out:
        from camera_geometry import build_rotation_matrix as brm
        def _grounds(xx):
            Rm = brm(xx[3], xx[4], xx[5])
            ce, cn = proj(xx[0], xx[1])
            return [_ground_point_enu(g, K, Rm, proj, ce, cn, xx[2]) for g in gcps]
        full = _grounds(x)
        for i, g in enumerate(gcps):
            rest = gcps[:i] + gcps[i + 1:]
            r = _solve(K, rest, x, proj, bounds)
            if not r.success:
                continue
            pts = _grounds(r.x)
            shifts = [np.hypot(a[0] - b[0], a[1] - b[1])
                      for a, b in zip(full, pts) if a is not None and b is not None]
            shift = float(max(shifts)) if shifts else float("nan")
            held = _per_point_m(K, [g], r.x, proj)[0]
            rms_wo = _rms(_per_point_m(K, rest, r.x, proj))
            out.leave_one_out.append(
                LeaveOneOut(g.label, shift, held, rms_wo, False))
        _flag_leave_one_out(out)
        for l in out.leave_one_out:
            if l.flagged:
                log.warning(
                    "GCP %s is suspect: removing it moves the solution by up to "
                    "%.2f m, its error when left out is %.2f m and the rest fit "
                    "to %.2f m RMS (was %.2f m). Check the point (not dropped).",
                    l.label, l.shift_m, l.heldout_error_m, l.rms_without_m,
                    out.rms_m)
    return out


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
    Backwards-compatible wrapper around refine_pose_with_diagnostics.

    Returns (cam_lat, cam_lon, height_m, R_3x3, rms_deg, heading_deg,
    pitch_deg, roll_deg). rms_deg is the RMS ground residual converted to
    degrees of latitude (rms_m / 111320) for existing callers; use
    refine_pose_with_diagnostics for metres, per-point residuals and the
    leave-one-out report. If refinement fails (including fewer than MIN_GCPS
    points), returns the initial pose and NaN rms.
    """
    r = refine_pose_with_diagnostics(
        K, gcps, cam_lat_0, cam_lon_0, height_0,
        heading_0, pitch_0, roll_0, bounds=bounds,
    )
    rms_deg = r.rms_m / M_PER_DEG_LAT if r.ok else float("nan")
    return (r.cam_lat, r.cam_lon, r.height_m, r.R, rms_deg,
            r.heading_deg, r.pitch_deg, r.roll_deg)


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
