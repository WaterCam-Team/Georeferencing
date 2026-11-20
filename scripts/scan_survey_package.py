"""Pix4DCatch scan + solved pose -> an API site-survey folder: point cloud, site image, DEM, manifest.

Run after scan_to_pose.py (and its overlay check). From the scan's LiDAR depth maps
(confidence 2, 0.2-4 m from the phone) and RTK frame poses it writes, to --out-dir/import/:

  * scan_cloud.json  the 3D viewer's cloud (pos_b64/col_b64), voxel-averaged, x east / y up /
                     z south in metres from the camera, origin in "georef" (orthometric heights);
  * site_scan.png    a north-up top view (highest point per cell) on a lat/lon grid, for the map;
  * dem_<unit>_<date>.tif  a bare-ground DEM (see ground_dem()), orthometric, scan CRS + --vertical-epsg;
  * dem_preview.png  the DEM, water cells blue, camera red;
  * reference.jpg    the node photo the pose was solved from;
  * manifest.json    for the API's scripts/import_survey.py.

Ground model in the manifest's base_pose:
  * without --dem-id: a flat water surface; mount_height_m = camera height above the water plane;
  * with --dem-id N (the id the API gave the uploaded DEM): map_dem_id N and mount_height_m = camera
    height above the DEM at the camera, because the API places the camera at DEM(camera) + mount_height_m.
So: run once, upload the DEM, run again with --dem-id.

Water plane: --water-height (orthometric m, e.g. an RTK waterline point) or, by default, the 5th
percentile of scan heights in the camera's view --water-window metres out. Over shallow clear water
LiDAR may return the bed, so a surveyed waterline is better.

Checks printed: low-point percentiles in view, a DEM height profile along the heading, and the
API's own ground lookup (ufonet_geo.ortho) with the flat plane and with the DEM: the photo rows
each one maps. See docs/SCAN_SITE_SURVEY.md.

Example:
    ../.venv-geo-api/bin/python -I scripts/scan_survey_package.py \\
        --scan private/<visit>/scan --photo private/<visit>/photo.jpg --calibration calibration.json \\
        --pose private/<visit>/pose_config_<unit>_<date>.json \\
        --report private/<visit>/scan_pose_report_<unit>_<date>.json \\
        --device-id <api device id> --unit-id <unit> --date <date> \\
        --valid-from 2026-11-16T15:25:41-05:00 --name "<site>, from <date> (RTK site scan)" \\
        --out-dir private/<visit> [--dem-id N]
"""
import argparse
import base64
import csv
import json
import os
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ufonet_geo.geometry import build_rotation_matrix, latlon_to_enu  # noqa: E402
from scan_to_pose import load_scan  # noqa: E402


def geoid_separation(scan):
    """Mean GeoidSeparation (m) of the scan's RTK log: orthometric = ellipsoidal - N."""
    rows = [float(r["GeoidSeparation"]) for r in csv.DictReader(open(scan / "geolocations" / "rtkGPS.csv"))
            if r.get("GeoidSeparation")]
    if not rows:
        raise SystemExit("no GeoidSeparation in geolocations/rtkGPS.csv: can't make orthometric heights")
    return float(np.mean(rows))


def scan_points(scan, frames, intr, shift, centre_xy, radius, geoid_n, depth=(0.2, 4.0)):
    """Scan CRS x, y, orthometric z and BGR colour of the depth pixels within radius of centre_xy."""
    f, cx, cy, iw = intr
    P, C = [], []
    for image, dmap, pos, R in frames:
        if np.hypot(*(pos[:2] - shift[:2] - centre_xy)) > radius + 4:
            continue
        d = cv2.imread(str(scan / "images" / dmap), cv2.IMREAD_UNCHANGED)
        cf = cv2.imread(str(scan / "images" / dmap.replace("DepthMap", "Confidence")), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(str(scan / "images" / image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        ds = d.shape[1] / iw
        vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        ok = (d > depth[0]) & (d < depth[1]) & (cf == 2)
        U, V, z = uu[ok] / ds, vv[ok] / ds, d[ok]
        W = pos + np.c_[(U - cx) / f * z, -(V - cy) / f * z, -z] @ R.T - shift   # OPF: x right, y up, z back
        W[:, 2] -= geoid_n
        keep = np.hypot(W[:, 0] - centre_xy[0], W[:, 1] - centre_xy[1]) <= radius
        P.append(W[keep])
        C.append(img[np.clip(V[keep].astype(int), 0, img.shape[0] - 1),
                     np.clip(U[keep].astype(int), 0, img.shape[1] - 1)])
    if not P:
        raise SystemExit("no scan frames near the camera")
    return np.vstack(P), np.vstack(C)


def view_low_points(enu, heading_deg, window=(2.0, 5.0), half_width_deg=25.0):
    """Percentiles of the up coordinate of points in the camera's view wedge, window metres out."""
    az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360
    rng = np.hypot(enu[:, 0], enu[:, 1])
    sel = (np.abs((az - heading_deg + 180) % 360 - 180) < half_width_deg) & (rng > window[0]) & (rng < window[1])
    if sel.sum() < 100:
        raise SystemExit("fewer than 100 scan points in the camera's view: give --water-height")
    up = enu[sel, 2]
    return {"n": int(sel.sum()), **{f"p{q}": float(np.percentile(up, q)) for q in (2, 5, 10, 25, 50)}}


def cloud_payload(enu, bgr, voxel, session, georef):
    """Voxel-averaged cloud in the 3D viewer's JSON: x east, y up, z south (three.js)."""
    key = np.floor(enu / voxel).astype(np.int64)
    _, idx, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    idx = idx.ravel()
    sums = np.zeros((len(counts), 3)); np.add.at(sums, idx, enu)
    csum = np.zeros((len(counts), 3)); np.add.at(csum, idx, bgr.astype(np.float64))
    E, C = sums / counts[:, None], csum / counts[:, None]
    xyz = np.c_[E[:, 0], E[:, 2], -E[:, 1]].astype(np.float32)
    rgb = C[:, ::-1].round().clip(0, 255).astype(np.uint8)
    return {"session": session, "points": int(len(xyz)),
            "scene_dims": [round(float(np.ptp(xyz[:, i])), 1) for i in range(3)],
            "pos_b64": base64.b64encode(xyz.tobytes()).decode(),
            "col_b64": base64.b64encode(rgb.tobytes()).decode(), "georef": georef}


def ground_dem(x, y, z, x0, y1, n, res, water_h, open_m=1.1, tol_m=0.15, fill_m=0.5):
    """Bare-ground DEM (n x n cells of res m, top-left corner x0, y1) from points x, y, z.

    1. each cell: the 10th-percentile height of its points (lowest returns, robust to strays);
    2. objects narrower than open_m (box, pole, plants) removed: a grey opening of the
       nearest-filled grid; a cell more than tol_m above the opened surface takes the opened value;
    3. cells below water_h raised to it (the creek as a flat surface, not the bed);
    4. empty cells within fill_m of data take the nearest cell; the rest are NaN.
    Returns (dem, stats).
    """
    ix = ((x - x0) / res).astype(int)
    iy = ((y1 - y) / res).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    cell, h = iy[ok] * n + ix[ok], z[ok]
    o = np.lexsort((h, cell))
    cell, h = cell[o], h[o]
    starts = np.r_[0, np.flatnonzero(np.diff(cell)) + 1]
    counts = np.diff(np.r_[starts, len(cell)])
    low = np.full(n * n, np.nan)
    low[cell[starts]] = h[starts + (0.1 * (counts - 1)).astype(int)]
    low = low.reshape(n, n)
    data = np.isfinite(low)
    if not data.any():
        raise ValueError("no points in the DEM grid")
    near = ndimage.distance_transform_edt(~data, return_distances=False, return_indices=True)
    filled = low[tuple(near)]
    k = int(round(open_m / res)) | 1
    opened = ndimage.grey_opening(filled, size=(k, k))
    removed = filled - opened > tol_m
    ground = np.where(removed, opened, filled)
    raised = ground < water_h
    ground = np.maximum(ground, water_h)
    dist = ndimage.distance_transform_edt(~data) * res
    dem = np.where(dist <= fill_m, ground, np.nan).astype(np.float32)
    return dem, {"cells_with_data": int(data.sum()), "objects_removed": int((removed & data).sum()),
                 "raised_to_water": int((raised & data).sum())}


def write_dem(path, dem, crs, x0, y1, res, tags):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=dem.shape[1], height=dem.shape[0], count=1,
                       dtype="float32", crs=crs, transform=from_origin(x0, y1, res, res), nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(dem, 1)
        dst.update_tags(**tags)


def site_image(path, x, y, z, bgr, crs, res):
    """Top view (highest point per cell) warped to a lat/lon rectangle; returns [[S, W], [N, E]]."""
    import rasterio
    from rasterio.transform import array_bounds, from_origin
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    x0, y1 = x.min(), y.max()
    W_, H_ = int(np.ptp(x) / res) + 1, int(np.ptp(y) / res) + 1
    idx = ((y1 - y) / res).astype(int) * W_ + ((x - x0) / res).astype(int)
    o = np.argsort(z)
    img = np.zeros((H_ * W_, 3), np.uint8); has = np.zeros(H_ * W_, bool)
    img[idx[o]] = bgr[o]; has[idx[o]] = True
    rgba = np.dstack([img.reshape(H_, W_, 3)[..., ::-1], np.where(has.reshape(H_, W_), 255, 0).astype(np.uint8)])
    tr = from_origin(x0, y1, res, res)
    dst_tr, w2, h2 = calculate_default_transform(crs, "EPSG:4326", W_, H_, *array_bounds(H_, W_, tr))
    dst = np.zeros((4, h2, w2), np.uint8)
    for b in range(4):
        reproject(rgba[..., b], dst[b], src_transform=tr, src_crs=crs, dst_transform=dst_tr,
                  dst_crs="EPSG:4326", resampling=Resampling.nearest)
    cv2.imwrite(str(path), cv2.cvtColor(dst.transpose(1, 2, 0), cv2.COLOR_RGBA2BGRA))
    west, north = dst_tr.c, dst_tr.f
    return [[north + dst_tr.e * h2, west], [north, west + dst_tr.a * w2]]


def lookup_rows(cal, pose, mount, max_range, dem_path=None):
    """Photo rows (min, max) the API's ground lookup maps, flat plane or DEM."""
    from ufonet_geo import ortho
    from ufonet_geo.terrain import sample_dem_grid
    K = np.array(cal["K"], float); D = np.array(cal["D"], float).ravel(); w, h = cal["img_size"]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)          # node masks: undistorted, alpha=1
    R = build_rotation_matrix(pose["heading_deg"], pose["pitch_deg"], pose["roll_deg"])
    L = ortho.build_ground_lookup_auto(newK, None, w, h, R, pose["camera_lat"], pose["camera_lon"], mount,
                                       dem_sampler=partial(sample_dem_grid, str(dem_path)) if dem_path else None,
                                       max_range_m=max_range)
    ok = np.isfinite(L.v)
    return (float(np.nanmin(L.v)), float(np.nanmax(L.v)), float(ok.sum() * L.res_m ** 2)) if ok.any() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--scan", required=True, type=Path)
    ap.add_argument("--photo", required=True, type=Path, help="the node photo the pose was solved from")
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--pose", required=True, type=Path, help="pose_config JSON from scan_to_pose.py")
    ap.add_argument("--report", required=True, type=Path, help="scan_pose_report JSON (camera_alt_ellipsoid_m)")
    ap.add_argument("--device-id", required=True, help="the API's device id (e.g. watercam-NNN)")
    ap.add_argument("--unit-id", required=True)
    ap.add_argument("--date", required=True, help="scan date (YYYY-MM-DD), for names")
    ap.add_argument("--valid-from", required=True, help="survey start with time zone (when the mount was final)")
    ap.add_argument("--name", required=True, help="survey name shown on the dashboard")
    ap.add_argument("--session-id", help="point cloud id on the API (default <unit>-scan-<date>, lower case)")
    ap.add_argument("--out-dir", required=True, type=Path, help="under private/; files go to <out-dir>/import")
    ap.add_argument("--dem-id", type=int, help="the uploaded DEM's API id: use it as the ground model")
    ap.add_argument("--water-height", type=float, help="water surface, orthometric m (default: from the scan)")
    ap.add_argument("--water-window", default="2,5", help="view range (m) for the scan water estimate")
    ap.add_argument("--radius", type=float, default=9.0, help="keep points within this of the camera (m)")
    ap.add_argument("--voxel", type=float, default=0.015, help="cloud voxel (m)")
    ap.add_argument("--site-res", type=float, default=0.02, help="site image cell (m)")
    ap.add_argument("--dem-res", type=float, default=0.1, help="DEM cell (m)")
    ap.add_argument("--max-range", type=float, default=5.0, help="max_range_m in the base pose")
    ap.add_argument("--vertical-epsg", type=int, default=5703, help="DEM vertical CRS (5703 NAVD88)")
    ap.add_argument("--lens-note", default="", help="extra method line, e.g. which calibration is used")
    a = ap.parse_args()

    out = a.out_dir.resolve()
    if not str(out).startswith(str((ROOT / "private").resolve()) + os.sep):
        raise SystemExit("--out-dir must be under private/: the outputs hold site coordinates")
    imp = out / "import"
    imp.mkdir(parents=True, exist_ok=True)
    pose = json.loads(a.pose.read_text())
    rep = json.loads(a.report.read_text())
    cal = json.loads(a.calibration.read_text())
    lat0, lon0 = pose["camera_lat"], pose["camera_lon"]
    session = a.session_id or f"{a.unit_id.lower()}-scan-{a.date}"

    frames, intr, shift, crs = load_scan(a.scan, 1)
    N = geoid_separation(a.scan)
    h0 = rep["camera_alt_ellipsoid_m"] - N
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    e0, n0 = fwd.transform(lon0, lat0)
    P, C = scan_points(a.scan, frames, intr, shift, np.array([e0, n0]), a.radius, N)
    lon, lat = inv.transform(P[:, 0], P[:, 1])
    east, north = latlon_to_enu(lat, lon, lat0, lon0)
    enu = np.c_[east, north, P[:, 2] - h0]
    print(f"{len(P):,} scan points within {a.radius} m; camera {h0:.3f} m orthometric (geoid separation {N:.2f})")

    win = tuple(float(v) for v in a.water_window.split(","))
    low = view_low_points(enu, pose["heading_deg"], win)
    print("scan heights in view (m below camera): "
          + ", ".join(f"{k} {-v:.2f}" for k, v in low.items() if k != "n") + f"  ({low['n']:,} points)")
    water_h = a.water_height if a.water_height is not None else h0 + low["p5"]
    water_src = "surveyed" if a.water_height is not None else "the lowest scan returns in the camera's view"

    # cloud, site image, photo
    georef = {"origin_lat": lat0, "origin_lon": lon0, "origin_h_m": h0,
              "height": "orthometric (ellipsoidal minus the scan's geoid separation)",
              "axes": "x east, y up, z south (m)",
              "source": f"Pix4DCatch {a.date}, LiDAR depth + RTK frame poses"}
    cloud = cloud_payload(enu, C, a.voxel, session, georef)
    (imp / "scan_cloud.json").write_text(json.dumps(cloud))
    bounds = site_image(imp / "site_scan.png", P[:, 0], P[:, 1], P[:, 2], C, crs, a.site_res)
    (imp / "reference.jpg").write_bytes(a.photo.read_bytes())

    # DEM
    from rasterio.crs import CRS as RCRS
    from ufonet_geo.terrain import sample_dem_grid
    n = int(np.ceil(2 * a.radius / a.dem_res))
    x0, y1 = e0 - n * a.dem_res / 2, n0 + n * a.dem_res / 2
    dem, st = ground_dem(P[:, 0], P[:, 1], P[:, 2], x0, y1, n, a.dem_res, water_h)
    try:
        dcrs = RCRS.from_user_input(f"EPSG:{crs.to_epsg()}+{a.vertical_epsg}")
    except Exception:
        dcrs = RCRS.from_wkt(crs.to_wkt())
        print("warning: couldn't add the vertical CRS; the DEM has no vertical datum tag")
    dem_path = imp / f"dem_{a.unit_id}_{a.date}.tif"
    write_dem(dem_path, dem, dcrs, x0, y1, a.dem_res,
              {"source": f"Pix4DCatch {a.date}, bare-ground filter", "water_surface_m": f"{water_h:.3f}"})
    dem_cam = float(sample_dem_grid(str(dem_path), np.array([lat0]), np.array([lon0]))[0])
    if not np.isfinite(dem_cam):
        raise SystemExit("the DEM has no height at the camera: the API would refuse it (raise --radius?)")
    v = np.nan_to_num(dem, nan=np.nanmin(dem))
    prev = cv2.applyColorMap(((v - v.min()) / max(np.ptp(v), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    prev[np.isnan(dem)] = 0
    prev[np.isclose(dem, water_h)] = (255, 120, 0)
    cv2.circle(prev, (n // 2, n // 2), 3, (0, 0, 255), -1)
    cv2.imwrite(str(imp / "dem_preview.png"), cv2.resize(prev, (n * 3, n * 3), interpolation=cv2.INTER_NEAREST))
    print(f"DEM {n}x{n} at {a.dem_res} m, {np.isfinite(dem).mean() * 100:.0f}% of the square filled, "
          f"{st['objects_removed']} object cells removed, {st['raised_to_water']} raised to the water plane "
          f"{water_h:.3f} m; DEM at camera {dem_cam:.3f} m")
    hd = np.radians(pose["heading_deg"])
    prof = []
    for r in np.arange(0.5, a.radius, 0.5):
        c, rr = int(n / 2 + np.sin(hd) * r / a.dem_res), int(n / 2 - np.cos(hd) * r / a.dem_res)
        val = dem[rr, c] if 0 <= rr < n and 0 <= c < n else np.nan
        prof.append(f"{r:.1f}:{val - h0:+.2f}" if np.isfinite(val) else f"{r:.1f}:--")
    print("DEM along the heading (m out: height vs camera): " + " ".join(prof))

    mount_flat = round(h0 - water_h, 3)
    mount_dem = round(h0 - dem_cam, 3)
    flat = lookup_rows(cal, pose, mount_flat, a.max_range)
    dem_rows = lookup_rows(cal, pose, mount_dem, a.max_range, dem_path)
    fmt = lambda t: f"photo rows {t[0]:.0f}-{t[1]:.0f}, {t[2]:.1f} m2" if t else "nothing mapped"  # noqa: E731
    print(f"API ground lookup: flat (mount {mount_flat}) {fmt(flat)}; DEM (mount {mount_dem}) {fmt(dem_rows)}")

    # manifest
    base = {"camera_lat": lat0, "camera_lon": lon0, "camera_alt_m": h0,
            "heading_deg": round(pose["heading_deg"], 2), "pitch_deg": round(pose["pitch_deg"], 2),
            "roll_deg": round(pose["roll_deg"], 2), "max_range_m": a.max_range, "mask_frame": "undistorted"}
    if a.dem_id:
        base.update(mount_height_m=mount_dem, map_dem_id=a.dem_id)
        ground = (f"Ground: a {a.dem_res:g} m terrain model from the same scan (lowest returns per cell, objects "
                  f"under ~1 m wide removed), with the water drawn as a flat surface at {water_h:.2f} m ({water_src}), "
                  f"out to {a.max_range:g} m. Camera {mount_dem:.2f} m above the model.")
    else:
        base["mount_height_m"] = mount_flat
        ground = (f"Ground: a flat water surface at {water_h:.2f} m ({water_src}), {mount_flat:.2f} m below the "
                  f"camera, out to {a.max_range:g} m.")
    cp = rep.get("check_point") or {}
    sd = rep["bootstrap_sd"]
    method = ("Camera pose: the node's photo was matched to an RTK Pix4DCatch scan of the site; the matched "
              "points' scan positions give the camera's position and aim (free 6-DoF fit, no surveyed targets), "
              "checked by warping scan frames into the photo"
              + (f" and against a surveyed point ({cp['horizontal_m']:.2f} m from the camera)" if cp else "")
              + ".\n" + ground + (f"\n{a.lens_note}" if a.lens_note else ""))
    accuracy = (f"Pose: {rep['inliers']} matched points, {rep['heldout_median_px']:.1f} px held-out median error; "
                f"match noise {sd['heading_deg']:.2f} deg heading, {sd['pitch_deg']:.2f} deg pitch; the scan's own "
                "registration limits it to about 1 deg.\n"
                + ("Water surface: surveyed." if a.water_height is not None else
                   "Water surface height: LiDAR over shallow clear water can return the bed, so the surface may be "
                   "a few cm higher than drawn; water edges move by about 2-3x that error a few metres out."))
    manifest = {
        "device_id": a.device_id,
        "survey": {"name": a.name, "valid_from": a.valid_from, "valid_to": None, "base_pose": base,
                   "camera_calibration": {"K": cal["K"], "D": cal["D"], "img_size": cal["img_size"]},
                   "method": method, "accuracy": accuracy},
        "reference_photo": "reference.jpg",
        "site_image": {"file": "site_scan.png", "bounds": bounds,
                       "label": f"RTK iPhone scan, {a.date} (LiDAR points from above)"},
        "pointcloud": {"file": "scan_cloud.json", "session_id": session},
        "water_maps": [],
    }
    (imp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (imp / "dem_info.json").write_text(json.dumps(
        {"file": dem_path.name, "res_m": a.dem_res, "water_surface_m": water_h, "dem_at_camera_m": dem_cam,
         "camera_alt_m": h0, "mount_height_m_flat": mount_flat, "mount_height_m_with_dem": mount_dem}, indent=1))
    mb = (imp / "scan_cloud.json").stat().st_size / 1e6
    print(f"cloud {cloud['points']:,} points, {mb:.1f} MB; manifest ground: "
          + (f"DEM {a.dem_id}, mount {mount_dem}" if a.dem_id else f"flat, mount {mount_flat} "
             f"(upload {dem_path.name}, then rerun with --dem-id)") + f"; wrote {imp}")


if __name__ == "__main__":
    main()
