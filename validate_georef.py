"""
Georeferencing accuracy validation for Meadowbrook-006 data.

Compares terrain-aware georeferencing of a UFONet photo against:
1. DEM elevation consistency (camera height above terrain)
2. Pix4DCatch RTK scan spatial extent (do footprints overlap?)
3. Flat vs terrain-aware georeferencing difference at image center

Usage:
    python validate_georef.py \
        --image  /path/to/20260426-090402-NIR-OFF.jpg \
        --dem    /path/to/USGS_1M_18_x41y477.tif \
        --pix4d  /path/to/2026-04-26-14-07-06/

Result: prints validation report with footprint, overlap, and error estimates.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# EXIF parsing
# ─────────────────────────────────────────────────────────────────────────────

def _dms_to_dec(degrees, minutes, seconds, ref):
    val = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        val = -val
    return val


def read_exif_pose(image_path):
    """
    Return dict with lat, lon, altitude_m, heading, pitch, roll.
    Uses exiftool -j -n for numeric output.
    """
    import subprocess
    result = subprocess.run(
        ["exiftool", "-j", "-n", image_path],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)[0]
    lat = float(data["GPSLatitude"])
    lon = float(data["GPSLongitude"])
    if data.get("GPSLongitudeRef") == "W" and lon > 0:
        lon = -lon
    if data.get("GPSLatitudeRef") == "S" and lat > 0:
        lat = -lat
    alt = float(data.get("GPSAltitude", 0) or 0)
    pitch = float(data.get("Pitch", 0) or 0)
    roll = float(data.get("Roll", 0) or 0)
    yaw = float(data.get("Yaw", data.get("GPSTrack", 0)) or 0)
    return {
        "lat": lat, "lon": lon,
        "altitude_m": alt,
        "heading": yaw, "pitch": pitch, "roll": roll,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEM helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dem(dem_path):
    import rasterio
    from rasterio.warp import transform_bounds
    from pyproj import Transformer, CRS

    src = rasterio.open(dem_path)
    dem_crs = src.crs or CRS.from_epsg(4326)
    wgs84 = CRS.from_epsg(4326)
    to_dem = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True) if dem_crs != wgs84 else None
    bounds_wgs84 = transform_bounds(dem_crs, wgs84, *src.bounds)
    nodata = src.nodata
    _data_cache = [None]

    def _cache_data():
        if _data_cache[0] is None:
            _data_cache[0] = src.read(1)
        return _data_cache[0]

    def _valid(v):
        if nodata is not None and v == nodata:
            return False
        return not np.isnan(v)

    def get_elev(lon, lat, search_radius_px=200, warn=False):
        x, y = (to_dem.transform(lon, lat) if to_dem else (lon, lat))
        row, col = src.index(x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        import rasterio.windows
        d = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))
        v = float(d.flat[0])
        if _valid(v):
            return v
        # Exact cell is nodata — search nearest valid cell within radius
        data = _cache_data()
        r0 = max(0, row - search_radius_px)
        r1 = min(src.height, row + search_radius_px + 1)
        c0 = max(0, col - search_radius_px)
        c1 = min(src.width, col + search_radius_px + 1)
        patch = data[r0:r1, c0:c1]
        valid_mask = np.isfinite(patch)
        if nodata is not None:
            valid_mask &= (patch != nodata)
        rr, cc = np.where(valid_mask)
        if len(rr) == 0:
            return None
        dists = (rr - (row - r0)) ** 2 + (cc - (col - c0)) ** 2
        nearest = np.argmin(dists)
        dist_px = np.sqrt(dists[nearest])
        elev = float(patch[rr[nearest], cc[nearest]])
        if warn and dist_px > 1:
            res_m = abs(src.res[0])
            print(f"    [DEM] nodata at exact position; nearest valid cell "
                  f"{dist_px * res_m:.1f} m away  elev={elev:.2f} m")
        return elev

    return get_elev, bounds_wgs84


# ─────────────────────────────────────────────────────────────────────────────
# Pix4DCatch session parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_pix4d_session(session_dir):
    """
    Return dict with:
      rtk_points: list of (lat, lon, alt_ellipsoidal, alt_orthometric)
      camera_positions_utm: list of (utm_e, utm_n, utm_z) from projected_input_cameras.json
      crs_epsg: int or None
      shift: [dx, dy, dz] for local → UTM transform
    """
    result = {}

    # RTK GPS
    rtk_path = os.path.join(session_dir, "geolocations", "rtkGPS.csv")
    rtk_pts = []
    if os.path.exists(rtk_path):
        with open(rtk_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rtk_pts.append({
                        "lat": float(row["GPSLatitude"]),
                        "lon": float(row["GPSLongitude"]),
                        "alt_ellipsoidal": float(row["GPSAltitude"]),
                        "alt_ortho": float(row.get("ReferenceAltitude") or 0),
                        "h_acc": float(row.get("HAccuracy") or 0),
                        "quality": row.get("qualityIndicator", ""),
                    })
                except (ValueError, KeyError):
                    continue
    result["rtk_points"] = rtk_pts

    # Scene reference frame → UTM shift
    srf_path = os.path.join(session_dir, "opf_files", "scene_reference_frame.json")
    shift = None
    epsg = None
    if os.path.exists(srf_path):
        with open(srf_path) as f:
            srf = json.load(f)
        shift = srf.get("base_to_canonical", {}).get("shift", None)
        crs_def = srf.get("crs", {}).get("definition", "")
        import re
        # Last ID in WKT is the outer CRS (projected); first may be base geographic CRS
        matches = re.findall(r'ID\["EPSG",(\d+)\]', crs_def)
        if matches:
            epsg = int(matches[-1])
    result["shift"] = shift
    result["crs_epsg"] = epsg

    # Projected cameras → UTM positions → WGS84
    cam_path = os.path.join(session_dir, "opf_files", "projected_input_cameras.json")
    cam_wgs84 = []
    if os.path.exists(cam_path) and shift is not None and epsg is not None:
        with open(cam_path) as f:
            cams = json.load(f)
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
        for cap in cams.get("captures", []):
            pos = cap.get("geolocation", {}).get("position")
            if pos and len(pos) >= 2:
                utm_e = pos[0] - shift[0]
                utm_n = pos[1] - shift[1]
                lon, lat = to_wgs84.transform(utm_e, utm_n)
                cam_wgs84.append({"lat": lat, "lon": lon})
    result["camera_positions_wgs84"] = cam_wgs84

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Footprint: georeference grid of pixels
# ─────────────────────────────────────────────────────────────────────────────

def compute_footprint(K, R, cam_lat, cam_lon, cam_elev_m, get_elev, img_w, img_h,
                      grid_n=5, step_m=0.5, max_range_m=500.0):
    """
    Georeference a grid of pixels via terrain ray-casting.
    Returns list of (lat, lon, elev, slant_m, u, v) hits.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from georeference_terrain import pixel_to_gps_terrain

    # undistort K not available here; skip distortion for footprint estimation
    us = np.linspace(0, img_w - 1, grid_n)
    vs = np.linspace(0, img_h - 1, grid_n)
    hits = []
    for v in vs:
        for u in us:
            r = pixel_to_gps_terrain(
                (u, v), K, R, cam_lat, cam_lon, cam_elev_m,
                get_elev, step_m=step_m, max_range_m=max_range_m,
            )
            if r is not None:
                hits.append({"lat": r[0], "lon": r[1], "elev": r[2],
                             "slant_m": r[3], "u": u, "v": v})
    return hits


def flat_georeference_center(K, R, cam_lat, cam_lon, cam_height_agl_m, img_w, img_h):
    """Flat-terrain georeference of image center pixel."""
    from geo_core import pixel_to_world_flat
    u, v = img_w / 2.0, img_h / 2.0
    result = pixel_to_world_flat(u, v, K, R, cam_lat, cam_lon, cam_height_agl_m)
    return result  # (lat, lon) or None


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox(points, lat_key="lat", lon_key="lon"):
    lats = [p[lat_key] for p in points]
    lons = [p[lon_key] for p in points]
    return {
        "min_lat": min(lats), "max_lat": max(lats),
        "min_lon": min(lons), "max_lon": max(lons),
    }


def bbox_overlap(a, b):
    """Return True if two bounding boxes overlap."""
    return (
        a["min_lat"] < b["max_lat"] and a["max_lat"] > b["min_lat"] and
        a["min_lon"] < b["max_lon"] and a["max_lon"] > b["min_lon"]
    )


def bbox_area_m2(bb):
    """Approximate area in m² using haversine for width/height."""
    from pyproj import Proj
    p = Proj(proj="aeqd", lat_0=(bb["min_lat"] + bb["max_lat"]) / 2,
             lon_0=(bb["min_lon"] + bb["max_lon"]) / 2, datum="WGS84")
    x0, y0 = p(bb["min_lon"], bb["min_lat"])
    x1, y1 = p(bb["max_lon"], bb["max_lat"])
    return abs((x1 - x0) * (y1 - y0))


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Main validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(image_path, dem_path, ucfg, pix4d_dir=None, grid_n=7,
             cli_heading=None, cli_height_agl=None, cli_pitch=None, cli_roll=None,
             cli_calibration=None):
    import cv2
    import unit_config as _uc
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from camera_geometry import build_rotation_matrix
    from georeference_tool import load_calibrated_intrinsics, scale_intrinsics_for_resolution

    print("=" * 66)
    print("GEOREFERENCING ACCURACY VALIDATION")
    print("=" * 66)
    if ucfg.unit_id:
        print(f"Unit: {ucfg.unit_id}")

    # ── 1. EXIF ──────────────────────────────────────────────────────────────
    print(f"\n[1] PHOTO EXIF: {os.path.basename(image_path)}")
    pose = read_exif_pose(image_path)
    cam_lat, cam_lon = pose["lat"], pose["lon"]
    exif_alt = pose["altitude_m"]

    heading, h_src = ucfg.resolve_heading(cli_heading, pose.get("heading"), pose.get("heading"))
    pitch,   p_src = ucfg.resolve_pitch(cli_pitch, pose.get("pitch"))
    roll,    r_src = ucfg.resolve_roll(cli_roll, pose.get("roll"))

    print(f"    GPS:      lat={cam_lat:.6f}  lon={cam_lon:.6f}")
    print(f"    Altitude: {exif_alt:.2f} m (EXIF GPS altitude)")
    print(f"    Heading:  {heading:.2f}°  [{h_src}]")
    print(f"    Pitch:    {pitch:.2f}°   [{p_src}]")
    print(f"    Roll:     {roll:.2f}°    [{r_src}]")

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return
    img_h, img_w = img.shape[:2]
    print(f"    Image:    {img_w}×{img_h}")

    # ── 2. DEM ───────────────────────────────────────────────────────────────
    print(f"\n[2] DEM: {os.path.basename(dem_path)}")
    get_elev, dem_bounds = load_dem(dem_path)
    cam_in_dem = (dem_bounds[0] <= cam_lon <= dem_bounds[2] and
                  dem_bounds[1] <= cam_lat <= dem_bounds[3])
    print(f"    Coverage: lon=[{dem_bounds[0]:.4f}, {dem_bounds[2]:.4f}] "
          f"lat=[{dem_bounds[1]:.4f}, {dem_bounds[3]:.4f}]")
    print(f"    Camera in DEM: {'YES' if cam_in_dem else 'NO ← problem'}")

    terrain_at_cam = get_elev(cam_lon, cam_lat, warn=True)
    if terrain_at_cam is None:
        print("[ERROR] DEM has no data at camera position or within search radius.")
        return
    print(f"    DEM elev at camera pos: {terrain_at_cam:.2f} m")

    # ── 3. Altitude analysis ─────────────────────────────────────────────────
    print("\n[3] ALTITUDE CONSISTENCY")
    exif_agl = exif_alt - terrain_at_cam
    print(f"    EXIF GPS altitude:       {exif_alt:.2f} m")
    print(f"    DEM terrain at camera:   {terrain_at_cam:.2f} m")
    print(f"    EXIF-implied AGL:        {exif_agl:.2f} m  (GPS vertical noise — not mount height)")

    height_agl, mh_src = ucfg.resolve_mount_height(cli_height_agl)
    if height_agl is not None:
        inches = height_agl / 0.0254
        print(f"    Mount height:            {height_agl:.4f} m  ({inches:.1f} in)  [{mh_src}]")
    else:
        height_agl = exif_agl
        mh_src = "exif_implied"
        if height_agl < 0:
            print("    [WARN] Negative AGL — datum mismatch or GPS error. Set mount_height_m in unit config.")
        elif height_agl > 50:
            print(f"    [WARN] AGL {height_agl:.1f} m seems high. Set mount_height_m in unit config.")
        else:
            print(f"    Mount height [exif_implied]: {height_agl:.2f} m  (set mount_height_m in unit config for accuracy)")

    cam_elev_terrain_datum = terrain_at_cam + height_agl
    print(f"    Camera elev (terrain datum): {cam_elev_terrain_datum:.2f} m")

    # ── 4. Pix4DCatch reference (if given) ────────────────────────────────────
    pix4d = None
    if pix4d_dir and os.path.isdir(pix4d_dir):
        print(f"\n[4] PIX4DCATCH SESSION: {os.path.basename(pix4d_dir)}")
        pix4d = load_pix4d_session(pix4d_dir)
        rtk = pix4d["rtk_points"]
        if rtk:
            rtk_bb = bbox(rtk)
            lats = [p["lat"] for p in rtk]
            lons = [p["lon"] for p in rtk]
            alts_e = [p["alt_ellipsoidal"] for p in rtk]
            alts_o = [p["alt_ortho"] for p in rtk]
            print(f"    RTK points: {len(rtk)}")
            print(f"    RTK lat range: [{min(lats):.6f}, {max(lats):.6f}]")
            print(f"    RTK lon range: [{min(lons):.6f}, {max(lons):.6f}]")
            print(f"    RTK alt ellipsoidal: [{min(alts_e):.2f}, {max(alts_e):.2f}] m")
            print(f"    RTK alt orthometric: [{min(alts_o):.2f}, {max(alts_o):.2f}] m")
            # Geoid separation from Pix4DCatch data
            sep_vals = [p["alt_ellipsoidal"] - p["alt_ortho"] for p in rtk if p["alt_ortho"] != 0]
            if sep_vals:
                geoid_sep = np.mean(sep_vals)
                print(f"    Geoid separation N (ellipsoidal - orthometric): {geoid_sep:.3f} m")
                exif_orthometric_check = exif_alt  # EXIF says "above sea level"
                exif_ellipsoidal_equiv = exif_alt + geoid_sep
                print(f"    EXIF 156m as orthometric → ellipsoidal equiv: {exif_ellipsoidal_equiv:.2f} m")
                rtk_terrain_ortho = np.mean(alts_o)
                print(f"    Mean Pix4DCatch terrain ortho:  {rtk_terrain_ortho:.2f} m")
                print(f"    Camera AGL (EXIF orth - terrain): {exif_alt - rtk_terrain_ortho:.2f} m")

            qual_rtk = [p for p in rtk if "rtkFloat" in p["quality"] or "rtkFixed" in p["quality"]]
            print(f"    RTK quality (rtkFloat/Fixed): {len(qual_rtk)}/{len(rtk)} samples")
        cams_wgs84 = pix4d["camera_positions_wgs84"]
        if cams_wgs84:
            print(f"    OPF camera poses decoded: {len(cams_wgs84)}")
    else:
        print("\n[4] PIX4DCATCH: not provided or not found")

    # ── 5. Camera calibration + rotation ─────────────────────────────────────
    print(f"\n[5] CAMERA SETUP")
    ucfg_dir = os.path.dirname(os.path.abspath(
        getattr(ucfg, '_path', None) or image_path))
    calib_path = ucfg.resolve_calibration(cli_calibration, ucfg_dir)
    print(f"    Calibration: {calib_path}")
    K, D, calib_img_size, _ = load_calibrated_intrinsics(calib_path)
    if calib_img_size and (calib_img_size[0], calib_img_size[1]) != (img_w, img_h):
        K = scale_intrinsics_for_resolution(K, calib_img_size[0], calib_img_size[1], img_w, img_h)
        print(f"    K scaled from {calib_img_size} → {img_w}×{img_h}")
    print(f"    fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    R = build_rotation_matrix(heading, pitch, roll)

    # Undistort K for pixel operations
    import cv2
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), alpha=0)
    x0, y0, cw, ch = roi

    # ── 6. Flat-terrain vs terrain-aware center pixel ─────────────────────────
    print(f"\n[6] CENTER PIXEL GEOREFERENCING COMPARISON")
    u_c, v_c = img_w / 2.0, img_h / 2.0
    # Adjust for undistort crop
    u_c_ud = u_c - x0
    v_c_ud = v_c - y0

    flat_result = flat_georeference_center(K_new, R, cam_lat, cam_lon, height_agl, cw, ch)
    if flat_result:
        flat_lat, flat_lon = flat_result
        print(f"    Flat terrain:     lat={flat_lat:.6f}  lon={flat_lon:.6f}")
    else:
        flat_lat, flat_lon = None, None
        print("    Flat terrain: no intersection (ray not downward)")

    from georeference_terrain import pixel_to_gps_terrain
    terrain_result = pixel_to_gps_terrain(
        (u_c_ud, v_c_ud), K_new, R,
        cam_lat, cam_lon, cam_elev_terrain_datum,
        get_elev, step_m=0.5, max_range_m=500.0,
    )
    if terrain_result:
        t_lat, t_lon, t_elev, t_slant = terrain_result
        print(f"    Terrain-aware:    lat={t_lat:.6f}  lon={t_lon:.6f}  "
              f"elev={t_elev:.2f} m  slant={t_slant:.1f} m")
        if flat_lat is not None:
            diff_m = haversine_m(flat_lat, flat_lon, t_lat, t_lon)
            print(f"    Flat vs terrain diff: {diff_m:.1f} m")
    else:
        print("    Terrain-aware: no intersection")

    # ── 7. Full footprint ─────────────────────────────────────────────────────
    print(f"\n[7] IMAGE FOOTPRINT (terrain-aware, {grid_n}×{grid_n} grid)")
    # Use undistorted K but grid over original image → remap to undistorted coords
    us_raw = np.linspace(0, img_w - 1, grid_n)
    vs_raw = np.linspace(0, img_h - 1, grid_n)
    hits = []
    for v_r in vs_raw:
        for u_r in us_raw:
            u_ud = u_r - x0
            v_ud = v_r - y0
            r = pixel_to_gps_terrain(
                (u_ud, v_ud), K_new, R,
                cam_lat, cam_lon, cam_elev_terrain_datum,
                get_elev, step_m=0.5, max_range_m=500.0,
            )
            if r is not None:
                hits.append({"lat": r[0], "lon": r[1], "elev": r[2],
                             "slant_m": r[3], "u": u_r, "v": v_r})

    if not hits:
        print("    [ERROR] No terrain intersections — check orientation/altitude.")
        return

    fp_bb = bbox(hits)
    print(f"    Hit pixels: {len(hits)}/{grid_n*grid_n}")
    print(f"    Footprint lat: [{fp_bb['min_lat']:.6f}, {fp_bb['max_lat']:.6f}]")
    print(f"    Footprint lon: [{fp_bb['min_lon']:.6f}, {fp_bb['max_lon']:.6f}]")
    fp_area = bbox_area_m2(fp_bb)
    print(f"    Footprint bbox area: {fp_area:.0f} m²  (~{fp_area**0.5:.0f}×{fp_area**0.5:.0f} m)")
    slants = [h["slant_m"] for h in hits]
    print(f"    Slant range: {min(slants):.1f} – {max(slants):.1f} m  (mean {np.mean(slants):.1f} m)")

    # ── 8. Overlap with Pix4DCatch ────────────────────────────────────────────
    if pix4d and pix4d["rtk_points"]:
        print(f"\n[8] OVERLAP WITH PIX4DCATCH SCAN")
        rtk_bb = bbox(pix4d["rtk_points"])
        overlap = bbox_overlap(fp_bb, rtk_bb)
        print(f"    Photo footprint overlaps Pix4D scan bbox: {'YES ✓' if overlap else 'NO ✗'}")

        if overlap:
            # Compute distance from footprint centroid to Pix4DCatch centroid
            fp_clat = (fp_bb["min_lat"] + fp_bb["max_lat"]) / 2
            fp_clon = (fp_bb["min_lon"] + fp_bb["max_lon"]) / 2
            rtk_clat = np.mean([p["lat"] for p in pix4d["rtk_points"]])
            rtk_clon = np.mean([p["lon"] for p in pix4d["rtk_points"]])
            centroid_dist = haversine_m(fp_clat, fp_clon, rtk_clat, rtk_clon)
            print(f"    Footprint centroid:  ({fp_clat:.6f}, {fp_clon:.6f})")
            print(f"    Pix4D RTK centroid:  ({rtk_clat:.6f}, {rtk_clon:.6f})")
            print(f"    Centroid separation: {centroid_dist:.1f} m")

        if pix4d["camera_positions_wgs84"]:
            cam_bb = bbox(pix4d["camera_positions_wgs84"])
            cam_overlap = bbox_overlap(fp_bb, cam_bb)
            # Nearest distance from photo footprint centroid to each OPF camera
            fp_clat2 = (fp_bb["min_lat"] + fp_bb["max_lat"]) / 2
            fp_clon2 = (fp_bb["min_lon"] + fp_bb["max_lon"]) / 2
            cam_dists = [haversine_m(fp_clat2, fp_clon2, c["lat"], c["lon"])
                         for c in pix4d["camera_positions_wgs84"]]
            nearest_cam = min(cam_dists)
            cam_centroid_lat = np.mean([c["lat"] for c in pix4d["camera_positions_wgs84"]])
            cam_centroid_lon = np.mean([c["lon"] for c in pix4d["camera_positions_wgs84"]])
            print(f"    OPF camera centroid:  ({cam_centroid_lat:.6f}, {cam_centroid_lon:.6f})")
            print(f"    Nearest OPF cam to footprint centroid: {nearest_cam:.1f} m")
            print(f"    Photo footprint overlaps Pix4D camera bbox: {'YES ✓' if cam_overlap else f'NO (nearest cam {nearest_cam:.1f} m away)'}")
    else:
        print("\n[8] OVERLAP: No Pix4DCatch data to compare")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"  Camera AGL:          {height_agl:.1f} m")
    if terrain_result:
        print(f"  Center pixel ground: lat={t_lat:.6f}  lon={t_lon:.6f}  elev={t_elev:.2f} m")
        if flat_lat:
            print(f"  Flat vs terrain err: {diff_m:.1f} m  (terrain correction)")
    print(f"  Footprint:           {fp_area:.0f} m²  ({len(hits)}/{grid_n*grid_n} pixels hit terrain)")
    if pix4d and pix4d["rtk_points"]:
        print(f"  Pix4D overlap:       {'YES' if overlap else 'NO'}")
        if overlap:
            print(f"  Centroid offset:     {centroid_dist:.1f} m  (photo footprint vs scan)")


def main():
    p = argparse.ArgumentParser(
        description="Validate georeferencing accuracy against Pix4DCatch RTK data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Unit-specific constants (mount height, heading, calibration) belong in\n"
            "a unit config JSON passed via --unit-config, not on the command line.\n"
            "CLI args override the unit config for one-off testing only.\n\n"
            "Example:\n"
            "  %(prog)s --image photo.jpg --dem terrain.tif --unit-config unit_config_UFO006.json"
        ),
    )
    p.add_argument("--image", required=True, help="Path to photo")
    p.add_argument("--dem", required=True, help="Path to GeoTIFF DEM")
    p.add_argument("--pix4d", default=None, help="Path to Pix4DCatch session directory")
    p.add_argument("--grid", type=int, default=7, help="Grid size for footprint (default 7)")
    # One-off overrides — prefer unit config for permanent values
    p.add_argument("--calibration", "-c", default=None,
                   help="Override calibration JSON (prefer unit config)")
    p.add_argument("--heading", type=float, default=None,
                   help="Override heading (deg). Prefer unit config heading_deg.")
    p.add_argument("--height-agl", type=float, default=None,
                   help="Override mount height (m). Prefer unit config mount_height_m.")
    p.add_argument("--pitch", type=float, default=None,
                   help="Override pitch (deg). Prefer unit config pitch_deg.")
    p.add_argument("--roll", type=float, default=None,
                   help="Override roll (deg). Prefer unit config roll_deg.")
    import unit_config as _uc
    _uc.add_argument(p)
    args = p.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.dem):
        print(f"DEM not found: {args.dem}")
        sys.exit(1)

    ucfg = _uc.from_args(args)
    if args.unit_config:
        ucfg._path = os.path.abspath(args.unit_config)

    validate(
        args.image, args.dem, ucfg,
        pix4d_dir=args.pix4d,
        grid_n=args.grid,
        cli_heading=args.heading,
        cli_height_agl=args.height_agl,
        cli_pitch=args.pitch,
        cli_roll=args.roll,
        cli_calibration=args.calibration,
    )


if __name__ == "__main__":
    main()
