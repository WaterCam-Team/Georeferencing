"""Pix4DCatch scan + one node photo -> the camera's pose, without surveyed targets.

When no ArUco targets were surveyed (so survey_to_pose.py can't run), an RTK-fixed
Pix4DCatch scan of the camera's view still gives 3D positions for whatever the camera
sees. This script:

  1. finds SIFT features in the node photo and in every --frame-step-th scan frame
     (read in their stored orientation, which the depth maps and intrinsics use);
  2. keeps the ratio-test matches that pass a fundamental-matrix RANSAC per frame;
  3. gives each matched scan pixel a 3D position from that frame's depth map
     (confidence 2 only) and OPF pose; a photo feature seen in several frames gets the
     median, and is dropped if the frames disagree by more than 10 cm;
  4. solves the full 6-DoF pose with PnP RANSAC + Levenberg-Marquardt, using the node's
     calibration (K, D). Nothing is held fixed, so a surveyed camera point
     (--check-csv/--check-point) and any phone readings stay independent checks;
  5. converts the camera to lat/lon and ellipsoidal height, and the rotation to the
     heading (true north) / pitch / roll of ufonet_geo.geometry.build_rotation_matrix;
  6. takes mount_height_m as the camera's height above the 20th-percentile depth-cloud
     height 0.3-0.8 m around it (the ground at the pole base; check it);
  7. reruns PnP on 200 random halves of the inliers (spread = noise only; the scan's own
     frame-orientation sigma, about 1 deg, limits the real accuracy).

It writes pose_config_<unit>_<date>.json (the API pose config fields),
scan_pose_report_<unit>_<date>.json, and scan_pose_matches_<unit>_<date>.npz to --out-dir,
which must be under private/ (site coordinates). Check the result with
scan_pose_overlay.py before uploading it. See docs/SCAN_TO_POSE.md.

Example:
    ../.venv-geo-api/bin/python -I scripts/scan_to_pose.py \\
        --scan private/survey-006-<date>/scan --photo private/survey-006-<date>/photo.jpg \\
        --calibration calibration.json --unit-id UFO006 --out-dir private/survey-006-<date> \\
        --check-csv private/survey-006-<date>/rtk.csv --check-point MOUNT_CENTRE
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from pyproj import CRS, Proj, Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ufonet_geo.geometry import build_rotation_matrix, latlon_to_enu  # noqa: E402
from pix4d_to_las_dem import _depth_pointcloud, _rotation_opf  # noqa: E402


def load_scan(scan, frame_step):
    opf = scan / "opf_files"
    inp = json.loads((opf / "input_cameras.json").read_text())
    sen = inp["sensors"][0]
    intr = (float(sen["internals"]["focal_length_px"]), *map(float, sen["internals"]["principal_point_px"]),
            int(sen["image_size_px"][0]))
    proj = {c["id"]: c for c in json.loads((opf / "projected_input_cameras.json").read_text())["captures"]}
    uri = {c["id"]: c["uri"].replace("../images/", "")
           for c in json.loads((opf / "camera_list.json").read_text())["cameras"]}
    srf = json.loads((opf / "scene_reference_frame.json").read_text())
    frames = []
    for cap in inp["captures"][::max(1, frame_step)]:
        pc = proj.get(cap["id"])
        image = depth = None
        for cam in cap.get("cameras", []):
            if "Image" in uri.get(cam["id"], ""):
                image = uri[cam["id"]]
            dm = cam.get("extensions", {}).get("PIX4D_input_depth_map", {}).get("id")
            if dm in uri:
                depth = uri[dm]
        if pc and "position" in pc.get("geolocation", {}) and image and depth:
            frames.append((image, depth, np.array(pc["geolocation"]["position"], float),
                           _rotation_opf(*pc["orientation"]["angles_deg"])))
    shift = np.array(srf["base_to_canonical"]["shift"], float)
    return frames, intr, shift, CRS.from_wkt(srf["crs"]["definition"])


def correspondences(photo_gray, K, D, scan, frames, intr, ratio=0.75, max_depth=8.0):
    f, cx, cy, img_w = intr
    clahe = cv2.createCLAHE(2.0, (8, 8))
    kp0, de0 = cv2.SIFT_create(nfeatures=20000).detectAndCompute(clahe.apply(photo_gray), None)
    sift = cv2.SIFT_create(nfeatures=8000)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    obs, per_frame = {}, []
    for image, depth, pos, R in frames:
        g = cv2.imread(str(scan / "images" / image), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        dm = cv2.imread(str(scan / "images" / depth), cv2.IMREAD_UNCHANGED)
        cf = cv2.imread(str(scan / "images" / depth.replace("DepthMap", "Confidence")), cv2.IMREAD_UNCHANGED)
        if g is None or dm is None:
            continue
        kp1, de1 = sift.detectAndCompute(clahe.apply(g), None)
        if de1 is None or len(kp1) < 50:
            continue
        m = [a for a, b in flann.knnMatch(de0, de1, k=2) if a.distance < ratio * b.distance]
        if len(m) < 25:
            continue
        p0 = np.float32([kp0[x.queryIdx].pt for x in m])
        p1 = np.float32([kp1[x.trainIdx].pt for x in m])
        p0u = cv2.undistortPoints(p0.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        F, inl = cv2.findFundamentalMat(p0u, p1, cv2.FM_RANSAC, 3.0, 0.999)
        if F is None or inl.sum() < 20:
            continue
        inl = inl.ravel().astype(bool)
        ds = dm.shape[1] / img_w
        n = 0
        for x, (u, v) in zip(np.array(m)[inl], p1[inl]):
            ud, vd = int(round(u * ds)), int(round(v * ds))
            if not (0 <= ud < dm.shape[1] and 0 <= vd < dm.shape[0]):
                continue
            z = float(dm[vd, ud])
            if (cf is not None and cf[vd, ud] < 2) or not (0.3 <= z <= max_depth):
                continue
            cam = np.array([(u - cx) / f * z, -(v - cy) / f * z, -z])   # OPF camera: x right, y up, z back
            obs.setdefault(x.queryIdx, []).append(pos + R @ cam)
            n += 1
        per_frame.append((image, int(inl.sum()), n))
    pts2, pts3 = [], []
    for i, L in obs.items():
        L = np.array(L)
        med = np.median(L, axis=0)
        if len(L) >= 2 and np.median(np.linalg.norm(L - med, axis=1)) > 0.10:
            continue
        pts2.append(kp0[i].pt)
        pts3.append(med)
    return np.float64(pts2), np.float64(pts3), per_frame


def pnp(P, Q, K, D, ransac=True):
    m = P.mean(0)
    if ransac:
        ok, rv, tv, inl = cv2.solvePnPRansac(P - m, Q, K, D, iterationsCount=20000, reprojectionError=6.0,
                                             confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise SystemExit("PnP RANSAC failed: too few consistent matches")
        inl = inl.ravel()
    else:
        ok, rv, tv = cv2.solvePnP(P - m, Q, K, D, flags=cv2.SOLVEPNP_EPNP)
        inl = np.arange(len(P))
    rv, tv = cv2.solvePnPRefineLM(P[inl] - m, Q[inl], K, D, rv, tv)
    Rw2c = cv2.Rodrigues(rv)[0]
    return Rw2c, m - Rw2c.T @ tv.ravel(), inl, (rv, tv, m)


def grid_to_enu(crs, e, n, lat, lon, step=10.0):
    """3x3 matrix taking scene (grid E, N, up) directions to the local ENU of ufonet_geo at (lat, lon).

    Built numerically through the same UTM -> lat/lon -> latlon_to_enu route the API's
    projection uses, so the meridian convergence (here about 0.7 deg) needs no sign convention."""
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    cols = []
    for de, dn in ((step, 0.0), (0.0, step)):
        lo, la = to_geo.transform(e + de, n + dn)
        x, y = latlon_to_enu(la, lo, lat, lon)
        cols.append([x / step, y / step, 0.0])
    M = np.array([cols[0], cols[1], [0.0, 0.0, 1.0]]).T
    u, _, vt = np.linalg.svd(M)          # drop the tiny scale factor, keep the rotation
    return u @ vt


def angles(Rw2c, G2E):
    """Scene -> camera rotation to true-north heading/pitch/roll (build_rotation_matrix convention)."""
    R = Rw2c @ G2E.T
    fwd = R[2]
    heading = math.degrees(math.atan2(fwd[0], fwd[1])) % 360
    pitch = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    rolls = np.arange(-60, 60, 0.01)
    roll = float(rolls[np.argmin([np.linalg.norm(build_rotation_matrix(heading, pitch, r) - R) for r in rolls])])
    resid = np.linalg.norm(build_rotation_matrix(heading, pitch, roll) - R)
    return heading, pitch, roll, resid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--scan", required=True, type=Path, help="Pix4DCatch project folder (opf_files/, images/)")
    ap.add_argument("--photo", required=True, type=Path, help="the node's photo, taken in the pose to solve")
    ap.add_argument("--calibration", required=True, type=Path, help="the camera calibration JSON (K, D)")
    ap.add_argument("--unit-id", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frame-step", type=int, default=2, help="use every n-th scan frame (default 2)")
    ap.add_argument("--max-depth", type=float, default=8.0, help="ignore depth beyond this (m)")
    ap.add_argument("--check-csv", type=Path, help="RTK CSV (Name, Latitude, Longitude, Ellipsoidal height)")
    ap.add_argument("--check-point", default="CAM", help="point in --check-csv to compare the camera with")
    ap.add_argument("--date", default=date.today().isoformat(), help="survey date for the file names")
    a = ap.parse_args()

    out = a.out_dir.resolve()
    if not str(out).startswith(str((ROOT / "private").resolve()) + os.sep):
        raise SystemExit("--out-dir must be under private/: the outputs hold site coordinates")
    out.mkdir(parents=True, exist_ok=True)

    cal = json.loads(a.calibration.read_text())
    K = np.array(cal.get("K") or cal.get("camera_matrix"), float)
    D = np.array(cal.get("D") or cal.get("dist_coeffs"), float).ravel()
    photo = cv2.imread(str(a.photo), cv2.IMREAD_GRAYSCALE)
    if photo is None:
        raise SystemExit(f"can't read {a.photo}")
    if cal.get("img_size") and list(photo.shape[::-1]) != list(cal["img_size"]):
        raise SystemExit(f"photo is {photo.shape[::-1]} but the calibration is for {cal['img_size']}")

    frames, intr, shift, crs = load_scan(a.scan, a.frame_step)
    print(f"{len(frames)} scan frames; matching ...")
    Q, P, per_frame = correspondences(photo, K, D, a.scan, frames, intr, max_depth=a.max_depth)
    print(f"{len(Q)} photo-scan correspondences from {len(per_frame)} frames")
    if len(Q) < 12:
        raise SystemExit("too few matches: is the photo's view in the scan?")
    Rw2c, C, inl, (rv, tv, m) = pnp(P, Q, K, D)
    pr, _ = cv2.projectPoints(P[inl] - m, rv, tv, K, D)
    res = np.linalg.norm(pr.reshape(-1, 2) - Q[inl], axis=1)

    C_utm = C - shift
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(C_utm[0], C_utm[1])
    gamma = Proj(crs).get_factors(lon, lat).meridian_convergence   # reported only
    G2E = grid_to_enu(crs, C_utm[0], C_utm[1], lat, lon)
    heading, pitch, roll, rresid = angles(Rw2c, G2E)

    # bootstrap: noise of the solve
    rng = np.random.default_rng(0)
    Pi, Qi, boot, held = P[inl], Q[inl], [], []
    for _ in range(200):
        idx = rng.permutation(len(Pi))
        h1, h2 = idx[:len(Pi) // 2], idx[len(Pi) // 2:]
        R_b, C_b, _, (rvb, tvb, mb) = pnp(Pi[h1], Qi[h1], K, D, ransac=False)
        prb, _ = cv2.projectPoints(Pi[h2] - mb, rvb, tvb, K, D)
        held.append(np.median(np.linalg.norm(prb.reshape(-1, 2) - Qi[h2], axis=1)))
        boot.append([*angles(R_b, G2E)[:3], *C_b])
    boot = np.array(boot)
    sd = boot.std(0)

    # mount height: camera above the ground at the pole base
    xyz, _ = _depth_pointcloud(a.scan, frame_step=max(2, a.frame_step), pixel_step=4)
    U = xyz - shift
    r = np.hypot(U[:, 0] - C_utm[0], U[:, 1] - C_utm[1])
    ring = (r > 0.3) & (r < 0.8)
    ground = float(np.percentile(U[ring, 2], 20)) if ring.sum() > 100 else None
    mount_h = round(float(C_utm[2]) - ground, 3) if ground is not None else None

    P_utm = Pi - shift
    d = (P_utm - C_utm) @ G2E.T
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    rng_m = np.hypot(d[:, 0], d[:, 1])
    hull = cv2.convexHull(Qi.astype(np.float32))
    report = {
        "unit_id": a.unit_id, "photo": str(a.photo), "scan": str(a.scan), "calibration": str(a.calibration),
        "camera_lat": lat, "camera_lon": lon, "camera_alt_ellipsoid_m": float(C_utm[2]),
        "heading_deg": heading, "pitch_deg": pitch, "roll_deg": roll, "roll_fit_residual": float(rresid),
        "meridian_convergence_deg": gamma, "ground_at_base_ellipsoid_m": ground, "mount_height_m": mount_h,
        "correspondences": int(len(Q)), "inliers": int(len(inl)), "frames_matched": len(per_frame),
        "best_frames": sorted(per_frame, key=lambda t: -t[2])[:5],
        "rms_px": float(np.sqrt(np.mean(res ** 2))), "median_px": float(np.median(res)),
        "heldout_median_px": float(np.median(held)),
        "bootstrap_sd": {"heading_deg": sd[0], "pitch_deg": sd[1], "roll_deg": sd[2], "position_m": list(sd[3:])},
        "inlier_azimuth_deg_p5_p50_p95": list(np.percentile(az, [5, 50, 95])),
        "inlier_range_m_p5_p50_p95": list(np.percentile(rng_m, [5, 50, 95])),
        "inlier_photo_coverage": float(cv2.contourArea(hull) / photo.size),
        "frame_step": a.frame_step,
    }
    if a.check_csv:
        pts = {row["Name"]: row for row in csv.DictReader(open(a.check_csv, newline="", encoding="utf-8-sig"))}
        if a.check_point in pts:
            cp = pts[a.check_point]
            e, n = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
                float(cp["Longitude"]), float(cp["Latitude"]))
            report["check_point"] = {"name": a.check_point,
                                     "horizontal_m": math.hypot(C_utm[0] - e, C_utm[1] - n)}
            if str(cp.get("Ellipsoidal height", "")).strip():
                report["check_point"]["camera_above_m"] = float(C_utm[2]) - float(cp["Ellipsoidal height"])

    pose_config = {"mount_height_m": mount_h, "camera_lat": round(lat, 9), "camera_lon": round(lon, 9),
                   "heading_deg": round(heading, 2), "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2)}
    tag = f"{a.unit_id}_{a.date}"
    json.dump(pose_config, open(out / f"pose_config_{tag}.json", "w"), indent=2)
    json.dump(report, open(out / f"scan_pose_report_{tag}.json", "w"), indent=2, default=float)
    np.savez(out / f"scan_pose_matches_{tag}.npz", pts2=Q, pts3_local=P, inliers=inl, shift=shift)

    print(f"PnP inliers {len(inl)}/{len(Q)}, RMS {report['rms_px']:.2f} px, held-out median {report['heldout_median_px']:.2f} px, "
          f"photo coverage {report['inlier_photo_coverage'] * 100:.0f}%")
    print(f"heading {heading:.2f} ± {sd[0]:.2f}  pitch {pitch:.2f} ± {sd[1]:.2f}  roll {roll:.2f} ± {sd[2]:.2f} deg; "
          f"position ± {np.linalg.norm(sd[3:]) * 100:.1f} cm (noise only)")
    print(f"camera alt {C_utm[2]:.3f} m ellipsoidal, mount height {mount_h} m")
    if "check_point" in report:
        print("check point:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["check_point"].items()})
    print(f"wrote {out}/pose_config_{tag}.json; check it with scan_pose_overlay.py before uploading")


if __name__ == "__main__":
    main()
