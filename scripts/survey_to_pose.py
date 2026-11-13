#!/usr/bin/env python3
"""RTK survey + one photo with ArUco targets -> the camera's surveyed pose.

For a site visit (see docs/GEOREFERENCING_PROCEDURE.md): lay ArUco targets in the
camera's view, take a photo with the node, survey each target and the camera with
RTK, then run this. It

1. finds the targets in the photo (aruco_gcp.detect_in_photo, undistorted frame);
2. matches them to the RTK points by label (marker id N <-> RTK point "<prefix>N");
3. writes the GCP CSV (gcp.save_gcps format);
4. solves heading, pitch and roll (>= 4 GCPs) with gcp.refine_pose_with_diagnostics,
   the camera held at its RTK point (fix_position) and at the taped mount height
   (--fit-height solves the height too). A free 6-parameter fit can trade a small
   camera shift for wrong angles when the targets lie on near-flat ground close to
   the camera, so the surveyed position is not refitted. It tries several starting
   headings, since an IMU/compass guess can be far off, keeps the best, and flags
   suspect GCPs with the library's leave-one-out check;
5. writes a unit config (unit_config.example.json layout) and the API pose config
   (PUT /devices/{id}/pose-config), and prints the residuals and leave-one-out flags;
6. writes an API survey import (survey_manifest_<unit>_<date>.json + the photo as
   reference.jpg) for the API's scripts/import_survey.py: an open survey from
   --valid-from (default now) with the lens calibration, the solved pose as its base
   pose, the method and an accuracy line from the residuals and held-out errors, and
   the node's IMU orientation (median over the photo and any --imu-photos) that later
   captures' tilt is measured against. On site:

       python scripts/import_survey.py <out-dir>/survey_manifest_<unit>_<date>.json \
           --api http://<server>:8000 --email <admin> --close-previous

Site data stays local: point --out-dir inside private/ (git-ignored). The script
refuses to write elsewhere unless --allow-outside-private is given.

    python3 scripts/survey_to_pose.py \\
        --photo private/survey-006-2026-11/capture.jpg \\
        --calibration private/calibrations/<unit>.json \\
        --rtk private/survey-006-2026-11/rtk_export.csv \\
        --unit-id UFO006 --device-id watercam-006 --mount-height 1.20 --heading 70 \\
        --pitch -15 --imu-photos private/survey-006-2026-11/imu/*.jpg \\
        --out-dir private/survey-006-2026-11/

RTK CSV: one row per point, with a name column and lat/lon/height columns. Column
names differ between receivers; the defaults match common exports (Name, Latitude,
Longitude, Ellipsoidal height) and --col-* options override them. The camera point
is the one named --camera-point (default CAM); targets are <--target-prefix><id>
(default "M", so marker 7 is "M7"). Heights are taken as given (ellipsoidal by
default; say which in --height-datum).
"""
import argparse
import csv
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def read_rtk(path, col_name, col_lat, col_lon, col_h):
    pts = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.strip(): c for c in (rd.fieldnames or [])}
        missing = [c for c in (col_name, col_lat, col_lon) if c not in cols]
        if missing:
            raise SystemExit(f"{path}: missing column(s) {missing}; columns are {list(cols)} "
                             "(use --col-name/--col-lat/--col-lon/--col-height)")
        for r in rd:
            name = (r[cols[col_name]] or "").strip()
            if not name:
                continue
            h = r.get(cols.get(col_h, ""), "") if col_h in cols else ""
            pts[name] = (float(r[cols[col_lat]]), float(r[cols[col_lon]]),
                         float(h) if str(h).strip() else None)
    return pts


def load_calibration(path):
    import numpy as np
    d = json.load(open(path))
    K = np.array(d.get("K") or d.get("camera_matrix"), dtype=float)
    D = np.array(d.get("D") or d.get("dist_coeffs") or [[0.0] * 5], dtype=float)
    return K, D, d


def solve_angles(K, gcps, cam_lat, cam_lon, height, heading0, pitch0, roll0, fit_height=False):
    """Heading, pitch and roll (and the height with fit_height), the camera held at its RTK point.

    gcp.refine_pose_with_diagnostics with fix_position (and fix_height unless fit_height),
    from several starting headings since an IMU or compass guess can be far off. Keeps the
    lowest ground RMS and reruns it with the leave-one-out check. Returns the PoseRefinement.
    """
    from gcp import refine_pose_with_diagnostics

    def fit(hd, loo):
        return refine_pose_with_diagnostics(
            K, gcps, cam_lat, cam_lon, height, ((hd + 180.0) % 360.0) - 180.0, pitch0, roll0,
            fix_position=True, fix_height=not fit_height, leave_one_out=loo)

    best = None
    for dh in (0, -30, 30, -60, 60, -90, 90, 180):
        r = fit(heading0 + dh, False)
        if r.ok and (best is None or r.rms_m < best[1].rms_m):
            best = (heading0 + dh, r)
    if best is None:
        raise SystemExit("pose solve failed from every starting heading: check the GCPs")
    return fit(best[1].heading_deg, True)


def imu_median(paths):
    """The node's IMU orientation over several of its photos (EXIF UserComment): the median
    gravity vector and roll/pitch, leaving out garbled readings (gravity far from 9.81) and
    a not-yet-started IMU (roll and pitch both 0). None if no photo has a usable reading."""
    import numpy as np
    from exif_imu import read_gps_imu_from_exif
    grav, rp = [], []
    for path in paths:
        try:
            e = read_gps_imu_from_exif(path)
        except Exception:
            continue
        g = e.get("gravity")
        if g and abs(float(np.linalg.norm(g)) - 9.81) <= 0.8:
            grav.append([float(v) for v in g])
        r, p = e.get("roll_deg"), e.get("pitch_deg")
        if r is not None and p is not None and not (r == 0 and p == 0):
            rp.append([float(r), float(p)])
    if not grav and not rp:
        return None
    out = {"source": f"median of {max(len(grav), len(rp))} survey photo(s), EXIF"}
    if grav:
        out["gravity"] = [round(float(v), 4) for v in np.median(grav, axis=0)]
    if rp:
        out["roll_deg"], out["pitch_deg"] = (round(float(v), 3) for v in np.median(rp, axis=0))
    return out


def survey_manifest(a, calib, img_size, pose_config, height, cam_h, gcps, rms, loo, flagged, imu, stamp):
    """The API survey import (scripts/import_survey.py in the API repo) for this survey."""
    import statistics
    from datetime import datetime
    base_pose = dict(pose_config)
    if cam_h is not None and a.height_datum.lower() in ("orthometric", "navd88"):
        base_pose["camera_alt_m"] = cam_h          # the 3D viewer's datum is orthometric
    errs = list(loo.values())
    accuracy = (f"Pose: ground residual RMS {rms:.2f} m over {len(gcps)} RTK-surveyed targets; "
                f"held-out (leave-one-out) error median {statistics.median(errs):.2f} m, "
                f"max {max(errs):.2f} m" + (f"; flagged: {', '.join(flagged)}" if flagged else "")
                + ". Position: the RTK camera point.") if errs else f"Pose: ground residual RMS {rms:.2f} m."
    method = (f"Site survey {stamp}: RTK points for the camera ({a.camera_point}) and {len(gcps)} ArUco "
              f"targets in the node's own photo; heading, pitch and roll solved from the targets with the "
              f"camera held at its RTK point, mount height {'solved' if a.fit_height else 'taped'} "
              f"({height:.2f} m). RTK heights: {a.height_datum}. scripts/survey_to_pose.py.")
    survey = {
        "name": a.survey_name or f"{a.unit_id} site survey {stamp}",
        "valid_from": a.valid_from or datetime.now().astimezone().isoformat(timespec="seconds"),
        "valid_to": None,
        "camera_calibration": {"K": calib.get("K") or calib.get("camera_matrix"),
                               "D": calib.get("D") or calib.get("dist_coeffs"), "img_size": img_size},
        "base_pose": base_pose,
        "method": method,
        "accuracy": accuracy,
    }
    if imu:
        survey["imu_orientation"] = imu
    return {"device_id": a.device_id, "survey": survey, "reference_photo": "reference.jpg", "water_maps": []}


def inside_private(path):
    p = os.path.realpath(path)
    priv = os.path.realpath(os.path.join(ROOT, "private"))
    return p == priv or p.startswith(priv + os.sep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--photo", required=True, help="the node's photo with the targets in view")
    ap.add_argument("--calibration", required=True, help="the unit's camera calibration JSON (K, D)")
    ap.add_argument("--rtk", required=True, help="RTK export CSV")
    ap.add_argument("--unit-id", required=True)
    ap.add_argument("--device-id", required=True, help="the unit's device id in the API (for the survey import)")
    ap.add_argument("--survey-name", help="the survey's name in the API (default: '<unit> site survey <date>')")
    ap.add_argument("--valid-from", help="when the survey starts applying, ISO 8601 with zone (default: now)")
    ap.add_argument("--imu-photos", nargs="*", default=[],
                    help="more node photos taken over a few minutes at the survey: their IMU readings' median "
                         "is the survey's orientation (one reading alone can be off)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mount-height", type=float, required=True,
                    help="camera height above the ground below it, taped (m)")
    ap.add_argument("--heading", type=float, required=True, help="rough heading for the solver (deg, from IMU/compass)")
    ap.add_argument("--pitch", type=float, default=-15.0, help="rough pitch for the solver (deg)")
    ap.add_argument("--roll", type=float, default=0.0, help="rough roll for the solver (deg)")
    ap.add_argument("--camera-point", default="CAM", help="RTK point name of the camera")
    ap.add_argument("--target-prefix", default="M", help="RTK name prefix of the ArUco targets")
    ap.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary")
    ap.add_argument("--col-name", default="Name")
    ap.add_argument("--col-lat", default="Latitude")
    ap.add_argument("--col-lon", default="Longitude")
    ap.add_argument("--col-height", default="Ellipsoidal height")
    ap.add_argument("--height-datum", default="ellipsoid", help="what the RTK heights are (for the notes)")
    ap.add_argument("--max-range", type=float, default=None, help="max_range_m for the API pose config")
    ap.add_argument("--fit-height", action="store_true",
                    help="solve the mount height too (default: keep the taped --mount-height)")
    ap.add_argument("--allow-outside-private", action="store_true")
    a = ap.parse_args()

    if not inside_private(a.out_dir) and not a.allow_outside_private:
        raise SystemExit(f"--out-dir {a.out_dir} is outside private/: site data must stay local "
                         "(use --allow-outside-private to override)")
    os.makedirs(a.out_dir, exist_ok=True)

    import cv2
    from aruco_gcp import detect_in_photo
    from gcp import GroundControlPoint, save_gcps

    K, D, calib = load_calibration(a.calibration)
    img = cv2.imread(a.photo)
    if img is None:
        raise SystemExit(f"can't read {a.photo}")
    found = detect_in_photo(img, K, D, a.dict)
    if not found:
        raise SystemExit("no ArUco markers found in the photo")
    K_undist = next(iter(found.values()))["K_undist"]

    pts = read_rtk(a.rtk, a.col_name, a.col_lat, a.col_lon, a.col_height)
    if a.camera_point not in pts:
        raise SystemExit(f"RTK export has no camera point {a.camera_point!r} (names: {sorted(pts)[:12]}...)")
    cam_lat, cam_lon, cam_h = pts[a.camera_point]

    gcps, unmatched = [], []
    for mid, m in sorted(found.items()):
        name = f"{a.target_prefix}{mid}"
        if name not in pts:
            unmatched.append(name)
            continue
        lat, lon, h = pts[name]
        u, v = m["center_uv"]
        gcps.append(GroundControlPoint(label=name, pixel_u=round(u, 2), pixel_v=round(v, 2),
                                       lat=lat, lon=lon, elev_m=h))
    surveyed_not_seen = sorted(n for n in pts if n.startswith(a.target_prefix)
                               and n[len(a.target_prefix):].isdigit()
                               and int(n[len(a.target_prefix):]) not in found)
    print(f"markers in the photo: {len(found)}; matched to RTK: {len(gcps)}")
    if unmatched:
        print(f"  seen but not surveyed: {', '.join(unmatched)}")
    if surveyed_not_seen:
        print(f"  surveyed but not seen: {', '.join(surveyed_not_seen)}")
    stamp = date.today().isoformat()
    gcp_path = os.path.join(a.out_dir, f"gcps_{a.unit_id}_{stamp}.csv")
    save_gcps(gcp_path, gcps)
    print(f"wrote {gcp_path}")
    if len(gcps) < 4:
        raise SystemExit("need at least 4 matched GCPs to solve the pose")

    ref = solve_angles(K_undist, gcps, cam_lat, cam_lon, a.mount_height,
                       a.heading, a.pitch, a.roll, a.fit_height)
    heading, pitch, roll, height = ref.heading_deg % 360.0, ref.pitch_deg, ref.roll_deg, ref.height_m
    per, rms = ref.residuals_m, ref.rms_m
    loo = {l.label: l.heldout_error_m for l in ref.leave_one_out}
    flagged = [l.label for l in ref.leave_one_out if l.flagged]
    print("solved pose (camera fixed at the RTK point):")
    print(f"  heading {heading:.2f} deg, pitch {pitch:.2f} deg, roll {roll:.2f} deg, "
          f"height {height:.2f} m ({'solved' if a.fit_height else 'taped'})")
    print(f"  ground residual RMS {rms:.2f} m; per GCP: "
          + ", ".join(f"{g.label} {r:.2f}" for g, r in zip(gcps, per)))
    print("  leave-one-out error per GCP: " + ", ".join(f"{l} {e:.2f}" for l, e in loo.items()))
    if flagged:
        print(f"  FLAGGED: {', '.join(flagged)} (check those targets; rerun without them if they're wrong)")

    # Position: the RTK camera point is the survey; the solve checks it and gives the angles
    unit_config = {
        "unit_id": a.unit_id,
        "calibration": os.path.basename(a.calibration),
        "camera_lat": cam_lat,
        "camera_lon": cam_lon,
        "camera_alt_ellipsoid_m": cam_h if a.height_datum == "ellipsoid" else None,
        "mount_height_m": round(height, 3),
        "heading_deg": round(heading, 2),
        "pitch_deg": round(pitch, 2),
        "roll_deg": round(roll, 2),
        "camera_elev_datum": a.height_datum,
        "notes": (f"Surveyed {stamp}: RTK camera point {a.camera_point}, {len(gcps)} ArUco GCPs; "
                  f"angles solved with the camera fixed at the RTK point (RMS {rms:.2f} m"
                  f"{', height solved' if a.fit_height else ''}). scripts/survey_to_pose.py."),
    }
    pose_config = {
        "mount_height_m": round(height, 3),
        "camera_lat": cam_lat,
        "camera_lon": cam_lon,
        "heading_deg": round(heading, 2),
        "pitch_deg": round(pitch, 2),
        "roll_deg": round(roll, 2),
        "mask_frame": "undistorted",
    }
    if a.max_range is not None:
        pose_config["max_range_m"] = a.max_range
    uc_path = os.path.join(a.out_dir, f"unit_config_{a.unit_id}_surveyed_{stamp}.json")
    pc_path = os.path.join(a.out_dir, f"pose_config_{a.unit_id}_{stamp}.json")
    json.dump(unit_config, open(uc_path, "w"), indent=2)
    json.dump(pose_config, open(pc_path, "w"), indent=2)
    report = {"rms_m": rms, "height_m": height, "height_solved": a.fit_height,
              "residuals_m": dict(zip([g.label for g in gcps], per)), "leave_one_out_m": loo,
              "flagged": flagged, "markers_seen": sorted(found), "unmatched": unmatched,
              "surveyed_not_seen": surveyed_not_seen}
    json.dump(report, open(os.path.join(a.out_dir, f"survey_report_{a.unit_id}_{stamp}.json"), "w"), indent=2)
    print(f"wrote {uc_path}\nwrote {pc_path}")

    import shutil
    shutil.copyfile(a.photo, os.path.join(a.out_dir, "reference.jpg"))
    img_size = calib.get("img_size") or [int(img.shape[1]), int(img.shape[0])]
    imu = imu_median([a.photo] + list(a.imu_photos))
    manifest = survey_manifest(a, calib, img_size, pose_config, height, cam_h, gcps, rms, loo, flagged, imu, stamp)
    mf_path = os.path.join(a.out_dir, f"survey_manifest_{a.unit_id}_{stamp}.json")
    json.dump(manifest, open(mf_path, "w"), indent=2)
    print(f"wrote {mf_path} (+ reference.jpg)")
    if imu is None:
        print("  no usable IMU reading in the photos: the survey has no orientation to measure tilt against")
    print("\nTo put it on the server as the unit's open survey (closing the previous one), from the API repo:")
    print(f"  python scripts/import_survey.py {mf_path} --api http://<server>:8000 --email <admin> "
          "--close-previous")
    return 0


if __name__ == "__main__":
    sys.exit(main())
