"""Check a pose config by warping Pix4DCatch scan frames into the node photo.

The chosen scan frames (default: the best-matched frames in the scan_to_pose report) are
back-projected through their depth maps at full image resolution and projected into the
node photo with ufonet_geo.geometry.build_rotation_matrix and the calibration's distortion,
the same pose model the API uses. So this checks the pose config end to end (north
reference, heading, roll sign, position, height), not only the solver.

Outputs:
  * an image with one panel per pose: the photo with its right half replaced by the
    warped scan (a misaligned pose shows as a break at the white split line);
    panels = the pose config, then each --variant ("roll=flip", "heading=122", ...);
  * a match score per pose: correlation of edge strength (blurred gradient magnitude) of
    photo and warped scan, over the region the scan covers in the pose config's view
    (fixed for every pose, so scores compare). Colour and exposure differ between the
    cameras, so only structure is compared. 0.3 is a good match, ~0 unrelated;
  * --sweep N: scores with each angle moved by +-N/2 and +-N deg. The pose config should
    score highest in every row; a row peaking off zero means that angle is off.

Example:
    ../.venv-geo-api/bin/python -I scripts/scan_pose_overlay.py \\
        --scan private/survey-006-<date>/scan --photo private/survey-006-<date>/photo.jpg \\
        --calibration calibration.json --pose private/survey-006-<date>/pose_config_UFO006_<date>.json \\
        --report private/survey-006-<date>/scan_pose_report_UFO006_<date>.json \\
        --variant roll=flip --sweep 1 --out private/survey-006-<date>/overlay.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ufonet_geo.geometry import build_rotation_matrix, latlon_to_enu, project_to_pixels  # noqa: E402
from scan_to_pose import load_scan  # noqa: E402


def frame_cloud(scan, frames, intr, names, max_depth=10.0):
    """Full-resolution coloured points (scene frame) from the named scan frames."""
    f, cx, cy, _ = intr
    P, C = [], []
    for image, depth, pos, R in frames:
        if Path(image).stem.split("_")[-1] not in names and Path(image).stem not in names and image not in names:
            continue
        im = cv2.imread(str(scan / "images" / image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        dm = cv2.imread(str(scan / "images" / depth), cv2.IMREAD_UNCHANGED).astype(np.float32)
        cf = cv2.imread(str(scan / "images" / depth.replace("DepthMap", "Confidence")), cv2.IMREAD_UNCHANGED)
        hh, ww = im.shape[:2]
        dmu = cv2.resize(dm, (ww, hh), interpolation=cv2.INTER_LINEAR)
        cfu = cv2.resize(cf, (ww, hh), interpolation=cv2.INTER_NEAREST)
        v, u = np.mgrid[0:hh, 0:ww]
        z = dmu
        ok = (z > 0.3) & (z < max_depth) & (cfu >= 2)
        u, v, z = u[ok], v[ok], z[ok]
        cam = np.c_[(u - cx) / f * z, -(v - cy) / f * z, -z]       # OPF camera: x right, y up, z back
        P.append(pos + cam @ R.T)
        C.append(im[v, u])
    if not P:
        raise SystemExit(f"none of the frames {sorted(names)} found in the scan")
    return np.vstack(P), np.vstack(C)


def render(enu, rgb, pose, K, D, w, h):
    pc = enu @ build_rotation_matrix(pose["heading_deg"], pose["pitch_deg"], pose["roll_deg"]).T
    u, v = project_to_pixels(pc, K, D)
    ok = np.isfinite(u) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (pc[:, 2] > 0.2)
    order = np.argsort(-pc[ok, 2])               # far first, so near points win
    uu, vv, cc = u[ok][order].astype(int), v[ok][order].astype(int), rgb[ok][order]
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), bool)
    for du in (0, 1):
        for dv in (0, 1):
            img[vv + dv, uu + du] = cc
            mask[vv + dv, uu + du] = True
    return img, mask


def match_score(photo, img, region):
    def grad(x):
        g = cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    a, b = grad(photo)[region], grad(img)[region]
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")   # nan: the scan is out of view


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--scan", required=True, type=Path)
    ap.add_argument("--photo", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--pose", required=True, type=Path, help="pose config JSON (camera_lat/lon, heading/pitch/roll)")
    ap.add_argument("--report", type=Path, help="scan_pose_report JSON: camera_alt_ellipsoid_m and best_frames")
    ap.add_argument("--camera-alt", type=float, help="camera ellipsoidal height (m), instead of the report's")
    ap.add_argument("--frames", help="comma-separated scan frame numbers to warp (e.g. 000271,000273); "
                                     "default: the report's three best-matched frames")
    ap.add_argument("--variant", action="append", default=[],
                    help="extra panel: heading=<deg>, pitch=<deg>, roll=<deg> or <angle>=flip")
    ap.add_argument("--sweep", type=float, default=0.0, help="score each angle at +-N/2 and +-N deg")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=1296, help="output panel width (px)")
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text())
    report = json.loads(a.report.read_text()) if a.report else {}
    alt = a.camera_alt if a.camera_alt is not None else report.get("camera_alt_ellipsoid_m")
    if alt is None:
        raise SystemExit("need --report or --camera-alt")
    if a.frames:
        names = set(a.frames.split(","))
    elif report.get("best_frames"):
        names = {Path(fr[0]).stem.split("_")[-1] for fr in report["best_frames"][:3]}
    else:
        raise SystemExit("need --frames or a --report with best_frames")
    cal = json.loads(a.calibration.read_text())
    K = np.array(cal["K"], float)
    D = np.array(cal["D"], float).ravel()
    photo = cv2.imread(str(a.photo))
    h, w = photo.shape[:2]

    frames, intr, shift, crs = load_scan(a.scan, 1)
    P, C = frame_cloud(a.scan, frames, intr, names)
    U = P - shift
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(U[:, 0], U[:, 1])
    e, n = latlon_to_enu(lat, lon, pose["camera_lat"], pose["camera_lon"])
    enu = np.c_[e, n, U[:, 2] - alt]

    img0, m0 = render(enu, C, pose, K, D, w, h)
    region = cv2.erode(m0.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    print(f"frames {sorted(names)}; scan covers {region.mean() * 100:.0f}% of the photo")

    def variant(p, k, val):
        q = dict(p)
        q[f"{k}_deg"] = -p[f"{k}_deg"] if val == "flip" else float(val)
        return q

    poses = [("pose config", pose)] + [(v, variant(pose, *v.split("="))) for v in a.variant]
    panels = []
    for name, p in poses:
        img, m = (img0, m0) if p is pose else render(enu, C, p, K, D, w, h)
        sc = match_score(photo, img, region)
        print(f"  {name:>14}: match {sc:.3f}")
        out = photo.copy()
        out[:, w // 2:][m[:, w // 2:]] = img[:, w // 2:][m[:, w // 2:]]
        cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 4)
        label = f"{name}: heading {p['heading_deg']:.1f} pitch {p['pitch_deg']:.1f} roll {p['roll_deg']:.1f}  match {sc:.2f}"
        for col, th in (((0, 0, 0), 9), ((255, 255, 255), 3)):
            cv2.putText(out, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, th)
        panels.append(cv2.resize(out, (a.width, int(h * a.width / w))))
    cols = 2 if len(panels) > 1 else 1
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    grid = np.vstack([np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)])
    cv2.imwrite(str(a.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])

    if a.sweep:
        steps = (-a.sweep, -a.sweep / 2, 0.0, a.sweep / 2, a.sweep)
        print("sweep (the pose config, 0, should score highest in every row):")
        for k in ("heading", "pitch", "roll"):
            row = []
            for d in steps:
                q = dict(pose)
                q[f"{k}_deg"] += d
                row.append(f"{d:+.1f}: {match_score(photo, render(enu, C, q, K, D, w, h)[0], region):.3f}")
            print(f"  {k:>8}  " + "   ".join(row))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
