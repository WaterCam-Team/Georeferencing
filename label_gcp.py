"""
label_gcp.py
============
Interactive manual pixel-labeling tool for ArUco/GCP markers.

Formalizes the manual labeling done in field validation sessions: automatic
ArUco detection (`aruco_gcp.detect_in_photo`) has repeatedly failed on real
field photos (oblique angle, foreshortened markers), so pixel coordinates were
estimated by eye/zoomed re-inspection and typed into a table by hand. This
tool replaces that with click-to-label against a magnified cursor view, driven
directly off RTK ground truth (a Pix4DCatch `input-control-points.json`, or a
plain label/lat/lon/elev_m CSV), and writes a `gcp.py`-compatible GCP CSV.

Why a magnifier instead of pan/zoom: precision at the marker (sub-pixel-ish
placement) is what limited manual labeling (some markers stayed at large
residuals even after a cropped, zoomed re-inspection pass). A fixed
always-on magnified inset next to the cursor gives that precision on every
click, with no separate zoom/pan state to manage.

Usage:
    python label_gcp.py --image photo.jpg --control-points Points/input-control-points.json
    python label_gcp.py --image photo.jpg --control-points gcps_truth.csv --resume

Output CSV columns: label,pixel_u,pixel_v,lat,lon,elev_m (see gcp.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gcp import GroundControlPoint, save_gcps, load_gcps

try:
    from georeference_tool import (
        load_calibrated_intrinsics,
        scale_intrinsics_for_resolution,
        undistort,
    )
    _HAS_GEOREF_TOOL = True
except Exception:
    _HAS_GEOREF_TOOL = False


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth control point loading (pure functions — unit testable)
# ─────────────────────────────────────────────────────────────────────────────

class ControlPoint:
    __slots__ = ("label", "lat", "lon", "elev_m")

    def __init__(self, label: str, lat: float, lon: float, elev_m: Optional[float]):
        self.label = label
        self.lat = lat
        self.lon = lon
        self.elev_m = elev_m


_TRAILING_LABEL_RE = re.compile(r"([A-Za-z]+_\d+)$")


def clean_control_point_label(raw_id: str) -> str:
    """
    PIX4D GCP ids are prefixed with the project name, e.g.
    "site 1_global_5" -> "global_5". Strip down to the trailing
    "<word>_<number>" token when present; otherwise keep the id as-is.
    """
    m = _TRAILING_LABEL_RE.search(raw_id.strip())
    return m.group(1) if m else raw_id.strip()


def load_control_points_pix4d_json(path: str | Path) -> List[ControlPoint]:
    """Parse a Pix4DCatch `input-control-points.json` GCP-collection file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points: List[ControlPoint] = []
    for gcp in data.get("gcps", []):
        raw_id = gcp.get("id", "")
        geo = gcp.get("geolocation", {})
        coords = geo.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lat, lon = float(coords[0]), float(coords[1])
        elev_m = float(coords[2]) if len(coords) > 2 else None
        points.append(ControlPoint(clean_control_point_label(raw_id), lat, lon, elev_m))
    return points


def load_control_points_csv(path: str | Path) -> List[ControlPoint]:
    """Parse a plain label,lat,lon[,elev_m] CSV of RTK ground truth."""
    points: List[ControlPoint] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elev = (row.get("elev_m") or "").strip()
            points.append(ControlPoint(
                row["label"].strip(),
                float(row["lat"]),
                float(row["lon"]),
                float(elev) if elev else None,
            ))
    return points


def load_control_points(path: str | Path) -> List[ControlPoint]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_control_points_pix4d_json(path)
    return load_control_points_csv(path)


def natural_sort_key(label: str):
    parts = re.split(r"(\d+)", label)
    return [int(p) if p.isdigit() else p for p in parts]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive labeling session
# ─────────────────────────────────────────────────────────────────────────────

# Keys '1'..'9' then 'a'..'z' (minus reserved letters below) assign labels.
_RESERVED_KEYS = {ord(c) for c in "sSuUcCqQzZxX"} | {27}


def _assignable_keys(n: int) -> List[int]:
    keys = [ord(str(d)) for d in range(1, 10)]
    for c in "abdefghijklmnoprtvwy":  # skip reserved: s,u,c,q,z,x
        keys.append(ord(c))
    return keys[:n]


class LabelSession:
    def __init__(
        self,
        image: np.ndarray,
        control_points: List[ControlPoint],
        display_max_dim: int = 1400,
        magnifier_zoom: int = 6,
        magnifier_box: int = 220,
    ):
        self.image = image
        self.img_h, self.img_w = image.shape[:2]
        self.control_points = sorted(control_points, key=lambda p: natural_sort_key(p.label))
        self.keys = _assignable_keys(len(self.control_points))
        self.key_to_label = {k: cp.label for k, cp in zip(self.keys, self.control_points)}
        self.label_to_cp = {cp.label: cp for cp in self.control_points}

        # label -> (pixel_u, pixel_v)
        self.assignments: Dict[str, Tuple[float, float]] = {}
        self.history: List[str] = []  # order of assignment, for undo

        scale = min(1.0, display_max_dim / max(self.img_w, self.img_h))
        self.disp_scale = scale
        self.disp_w = max(1, int(round(self.img_w * scale)))
        self.disp_h = max(1, int(round(self.img_h * scale)))
        self.display_base = cv2.resize(image, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        self.mouse_disp = (0, 0)
        self.pending_pos: Optional[Tuple[float, float]] = None  # full-image (u, v) awaiting a label key
        self.magnifier_zoom = magnifier_zoom
        self.magnifier_box = magnifier_box

        self.window = "label_gcp - see terminal for key legend"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # The Qt highgui backend doesn't materialize a real window handle
        # until the first imshow()+waitKey() cycle runs; setMouseCallback()
        # called any earlier raises "NULL window handler".
        cv2.imshow(self.window, self.display_base)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window, self._on_mouse)

    # -- coordinate helpers --------------------------------------------------
    def _disp_to_full(self, x: int, y: int) -> Tuple[float, float]:
        return x / self.disp_scale, y / self.disp_scale

    def _full_to_disp(self, u: float, v: float) -> Tuple[int, int]:
        return int(round(u * self.disp_scale)), int(round(v * self.disp_scale))

    def _on_mouse(self, event, x, y, *_):
        # cv2 invokes this from a C callback; an uncaught Python exception here
        # can be swallowed instead of surfacing, which looks like "the click
        # did nothing." Guard defensively and always report what we saw.
        try:
            self.mouse_disp = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                # NOTE: left-click no longer guesses which label you meant by
                # assuming clicks happen in ascending label order (global_1,
                # then global_2, ...). That guess silently mislabeled markers
                # clicked in a different order (confirmed 2026-08-05: 4/5
                # markers came out shifted by one label, and the 5th landed on
                # an already-labeled spot instead of the intended marker).
                # A click now only marks a *pending* position; you must press
                # the marker's own number/letter key to commit it to a label.
                u, v = self._disp_to_full(x, y)
                self.pending_pos = (min(max(u, 0.0), self.img_w - 1.0),
                                     min(max(v, 0.0), self.img_h - 1.0))
                print(f"[CLICK] pending at pixel ({self.pending_pos[0]:.1f}, "
                      f"{self.pending_pos[1]:.1f}) — press the marker's key to assign it.",
                      flush=True)
        except Exception:
            import traceback
            print("[ERROR] exception in mouse callback:", flush=True)
            traceback.print_exc()

    def _assign(self, label: str):
        # Prefer the last deliberate click (pending_pos); fall back to the
        # live hover position for users who just hover-and-press without
        # clicking first. Either way this is always an explicit, single-label
        # commit -- never inferred from click order.
        if self.pending_pos is not None:
            u, v = self.pending_pos
        else:
            u, v = self._disp_to_full(*self.mouse_disp)
            u = min(max(u, 0.0), self.img_w - 1.0)
            v = min(max(v, 0.0), self.img_h - 1.0)
        self.assignments[label] = (u, v)
        if label in self.history:
            self.history.remove(label)
        self.history.append(label)
        self.pending_pos = None
        print(f"[ASSIGN] {label} -> pixel ({u:.1f}, {v:.1f})", flush=True)

    def _undo(self):
        if not self.history:
            print("[UNDO] Nothing to undo.")
            return
        label = self.history.pop()
        del self.assignments[label]
        print(f"[UNDO] Cleared {label}")

    def _clear_nearest(self):
        if not self.assignments:
            print("[CLEAR] No assignments yet.")
            return
        mu, mv = self._disp_to_full(*self.mouse_disp)
        nearest = min(
            self.assignments.items(),
            key=lambda kv: (kv[1][0] - mu) ** 2 + (kv[1][1] - mv) ** 2,
        )
        label = nearest[0]
        del self.assignments[label]
        if label in self.history:
            self.history.remove(label)
        print(f"[CLEAR] Cleared {label} (nearest to cursor)")

    def _to_gcps(self) -> List[GroundControlPoint]:
        gcps = []
        for label, (u, v) in self.assignments.items():
            cp = self.label_to_cp[label]
            gcps.append(GroundControlPoint(label, u, v, cp.lat, cp.lon, cp.elev_m))
        return gcps

    # -- drawing --------------------------------------------------------------
    def _draw_magnifier(self, canvas: np.ndarray):
        u, v = self._disp_to_full(*self.mouse_disp)
        u_i, v_i = int(round(u)), int(round(v))
        half = self.magnifier_box // (2 * self.magnifier_zoom) or 1
        x0, x1 = u_i - half, u_i + half
        y0, y1 = v_i - half, v_i + half
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(self.img_w, x1), min(self.img_h, y1)
        crop = self.image[y0c:y1c, x0c:x1c]
        if crop.size == 0:
            return
        zoomed = cv2.resize(
            crop, None, fx=self.magnifier_zoom, fy=self.magnifier_zoom,
            interpolation=cv2.INTER_NEAREST,
        )
        box = self.magnifier_box
        mag = np.full((box, box, 3), 40, dtype=np.uint8)
        oy = int(round(pad_t * self.magnifier_zoom))
        ox = int(round(pad_l * self.magnifier_zoom))
        zh, zw = zoomed.shape[:2]
        zh, zw = min(zh, box - oy), min(zw, box - ox)
        if zh > 0 and zw > 0:
            mag[oy:oy + zh, ox:ox + zw] = zoomed[:zh, :zw]
        cy, cx = box // 2, box // 2
        cv2.line(mag, (cx - 8, cy), (cx + 8, cy), (0, 255, 255), 1)
        cv2.line(mag, (cx, cy - 8), (cx, cy + 8), (0, 255, 255), 1)
        cv2.rectangle(mag, (0, 0), (box - 1, box - 1), (200, 200, 200), 1)

        mx0, my0 = canvas.shape[1] - box - 10, 10
        canvas[my0:my0 + box, mx0:mx0 + box] = mag
        cv2.putText(canvas, f"{self.magnifier_zoom}x  px=({u:.1f},{v:.1f})",
                    (mx0, my0 + box + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)

    def _draw_overlay(self):
        canvas = self.display_base.copy()
        for label, (u, v) in self.assignments.items():
            x, y = self._full_to_disp(u, v)
            cv2.drawMarker(canvas, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(canvas, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if self.pending_pos is not None:
            x, y = self._full_to_disp(*self.pending_pos)
            cv2.drawMarker(canvas, (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
            cv2.putText(canvas, "press marker's key to assign", (x + 10, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y = 20
        for k, label in self.key_to_label.items():
            done = label in self.assignments
            text = f"[{chr(k)}] {label}" + (" DONE" if done else "")
            color = (120, 255, 120) if done else (255, 255, 255)
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 18

        n_done = len(self.assignments)
        n_total = len(self.control_points)
        cv2.putText(canvas, f"{n_done}/{n_total} labeled  |  click then press key  |  S save  U undo  C clear-nearest  Q quit",
                    (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        self._draw_magnifier(canvas)
        cv2.imshow(self.window, canvas)

    # -- main loop --------------------------------------------------------------
    def run(self, output_csv: Path) -> List[GroundControlPoint]:
        print("[label_gcp] Key legend:")
        for k, label in self.key_to_label.items():
            print(f"    [{chr(k)}] -> {label}")
        print("    Click on a marker (or just hover over it), then press its key to")
        print("    assign that exact label -- there is no click-order guessing.")
        print("    S save   U undo   C clear nearest-to-cursor   Z/X magnifier zoom -/+   Q/ESC quit")

        while True:
            self._draw_overlay()
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                gcps = self._to_gcps()
                if gcps:
                    save_gcps(output_csv, gcps)
                    print(f"[SAVE] {len(gcps)} GCPs -> {output_csv}")
                else:
                    print("[SAVE] Nothing labeled yet.")
            elif key in (ord("u"), ord("U")):
                self._undo()
            elif key in (ord("c"), ord("C")):
                self._clear_nearest()
            elif key in (ord("z"), ord("Z")):
                self.magnifier_zoom = max(1, self.magnifier_zoom - 1)
            elif key in (ord("x"), ord("X")):
                self.magnifier_zoom = min(20, self.magnifier_zoom + 1)
            elif key in self.key_to_label:
                self._assign(self.key_to_label[key])

        cv2.destroyAllWindows()
        return self._to_gcps()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Interactively label ArUco/GCP marker pixel positions against RTK ground truth."
    )
    p.add_argument("--image", required=True, help="Field photo to label")
    p.add_argument("--control-points", required=True,
                    help="RTK ground truth: Pix4DCatch input-control-points.json, "
                         "or a label,lat,lon[,elev_m] CSV")
    p.add_argument("--output-csv", default=None,
                    help="Output GCP CSV (default: <image_stem>_gcp.csv)")
    p.add_argument("--pixel-space", choices=["original", "undistorted"], default="undistorted")
    p.add_argument("--calibration", default="./calibration.json")
    p.add_argument("--resume", action="store_true",
                    help="Pre-load existing --output-csv assignments (by label) if present")
    p.add_argument("--display-max-dim", type=int, default=1400)
    p.add_argument("--magnifier-zoom", type=int, default=6)
    p.add_argument("--magnifier-box", type=int, default=220)
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERR] Image not found: {image_path}")
        return 2
    cp_path = Path(args.control_points)
    if not cp_path.exists():
        print(f"[ERR] Control points file not found: {cp_path}")
        return 2

    control_points = load_control_points(cp_path)
    if not control_points:
        print(f"[ERR] No control points parsed from {cp_path}")
        return 2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[ERR] Could not read image: {image_path}")
        return 2

    if args.pixel_space == "undistorted":
        if not _HAS_GEOREF_TOOL:
            print("[ERR] georeference_tool import failed; cannot undistort. Use --pixel-space original.")
            return 2
        if not Path(args.calibration).exists():
            print(f"[ERR] Calibration JSON not found: {args.calibration}")
            return 2
        h, w = image.shape[:2]
        K, D, calib_img_size, _ = load_calibrated_intrinsics(args.calibration)
        if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (w, h):
            K = scale_intrinsics_for_resolution(K, calib_img_size[0], calib_img_size[1], w, h)
        image, _K_new = undistort(image, K, D)  # noqa: F841
        print(f"[label_gcp] Working in undistorted pixel space "
              f"({image.shape[1]}x{image.shape[0]}) using {args.calibration}")
    else:
        print("[label_gcp] Working in original (distorted) pixel space.")

    out_path = Path(args.output_csv) if args.output_csv else image_path.with_name(image_path.stem + "_gcp.csv")

    session = LabelSession(
        image, control_points,
        display_max_dim=args.display_max_dim,
        magnifier_zoom=args.magnifier_zoom,
        magnifier_box=args.magnifier_box,
    )

    if args.resume and out_path.exists():
        existing = load_gcps(out_path)
        for g in existing:
            if g.label in session.label_to_cp:
                session.assignments[g.label] = (g.pixel_u, g.pixel_v)
                session.history.append(g.label)
        print(f"[RESUME] Loaded {len(existing)} existing assignments from {out_path}")

    gcps = session.run(out_path)
    if gcps:
        save_gcps(out_path, gcps)
        print(f"[SAVE] {len(gcps)} GCPs -> {out_path}")
    else:
        print("[label_gcp] No GCPs labeled; nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
