"""
Validate the quality of a DSM produced by pix4d_to_las_dem.py.

Checks performed
----------------
1. CRS        — must be EPSG:6347 (NAD83(2011) / UTM zone 18N)
2. Resolution — must match --expected-res (default 0.05 m)
3. Elevation  — z range must fall within plausible ellipsoidal bounds for the
                Syracuse NY area (100–125 m).  Adjust --z-min / --z-max for
                other sites.
4. Coverage   — fraction of bounding-box cells that contain valid data
5. Camera     — every camera pose from <scan>_camera_poses.csv must lie above
                the DSM surface; median height above ground must be 0.5–10 m
                (handheld close-range scan expectation)

Usage
-----
    .venv/bin/python scripts/validate_dsm.py output/pix4d/2026-04-24-13-11-52_dem.tif

Optional: supply the camera poses CSV explicitly
    .venv/bin/python scripts/validate_dsm.py output/pix4d/2026-04-24-13-11-52_dem.tif \\
        --poses output/pix4d/2026-04-24-13-11-52_camera_poses.csv

Exit code
---------
    0  all checks passed
    1  one or more checks failed
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
SKIP = "\033[90mSKIP\033[0m"


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return passed


def validate(dsm_path: Path, poses_path: Path | None,
             expected_res: float, z_min: float, z_max: float) -> bool:
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError as e:
        print(f"ERROR: missing dependency ({e}).  Run: pip install rasterio pyproj")
        sys.exit(1)

    print(f"\n=== DSM Quality Report: {dsm_path.name} ===\n")
    failures = 0

    # ------------------------------------------------------------------
    # Load raster
    # ------------------------------------------------------------------
    with rasterio.open(dsm_path) as src:
        data = src.read(1).astype(np.float64)
        nd = float(src.nodata) if src.nodata is not None else None
        res_x, res_y = src.res
        epsg = src.crs.to_epsg() if src.crs else None
        bounds = src.bounds
        transform = src.transform
        crs_obj = src.crs

    valid_mask = (data != nd) if nd is not None else np.ones(data.shape, dtype=bool)
    valid = data[valid_mask]
    total = data.size

    # ------------------------------------------------------------------
    # 1. CRS
    # ------------------------------------------------------------------
    print("Geometry")
    ok = _check(
        f"CRS = EPSG:6347",
        epsg == 6347,
        f"got EPSG:{epsg}",
    )
    failures += not ok

    # ------------------------------------------------------------------
    # 2. Resolution
    # ------------------------------------------------------------------
    ok = _check(
        f"Resolution = {expected_res:.3f} m",
        abs(res_x - expected_res) < 1e-6 and abs(res_y - expected_res) < 1e-6,
        f"got {res_x:.4f} × {res_y:.4f} m",
    )
    failures += not ok

    width_m = bounds.right - bounds.left
    height_m = bounds.top - bounds.bottom
    print(f"        Extent:     {width_m:.1f} × {height_m:.1f} m  "
          f"({data.shape[1]} × {data.shape[0]} px)")

    # ------------------------------------------------------------------
    # 3. Elevation range
    # ------------------------------------------------------------------
    print("\nElevation (ellipsoidal)")
    if len(valid) == 0:
        _check("Elevation range", False, "no valid cells")
        failures += 1
    else:
        z_lo, z_hi = float(valid.min()), float(valid.max())
        z_mean, z_std = float(valid.mean()), float(valid.std())
        ok = _check(
            f"z_min > {z_min:.0f} m",
            z_lo > z_min,
            f"got {z_lo:.2f} m  (datum error if outside expected range)",
        )
        failures += not ok
        ok = _check(
            f"z_max < {z_max:.0f} m",
            z_hi < z_max,
            f"got {z_hi:.2f} m",
        )
        failures += not ok
        print(f"        Z range:    {z_lo:.2f} – {z_hi:.2f} m  "
              f"(mean {z_mean:.2f} ± {z_std:.2f} m)")

    # ------------------------------------------------------------------
    # 4. Coverage
    # ------------------------------------------------------------------
    print("\nCoverage")
    n_valid = int(valid_mask.sum())
    frac = n_valid / total
    ok = _check(
        "Has valid cells",
        n_valid > 0,
        f"{n_valid:,} / {total:,} cells ({frac:.1%} of bounding box)",
    )
    failures += not ok
    if n_valid > 0:
        print(f"        Coverage:   {frac:.1%} of bounding-box  "
              f"(remainder is edge padding + scan gaps)")

    # ------------------------------------------------------------------
    # 5. Camera poses
    # ------------------------------------------------------------------
    print("\nCamera poses")
    if poses_path is None or not poses_path.exists():
        print(f"  {SKIP}  Camera poses CSV not found — skipping height check")
        print(f"        Expected: {poses_path}")
    else:
        tf = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
        heights = []
        below = []
        outside = 0

        with open(poses_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    lat = float(row["lat_deg"])
                    lon = float(row["lon_deg"])
                    alt = float(row["alt_ellipsoid_m"])
                except (KeyError, ValueError):
                    continue
                ex, ey = tf.transform(lon, lat)
                col = int((ex - transform.c) / transform.a)
                row_i = int((ey - transform.f) / transform.e)
                if not (0 <= row_i < data.shape[0] and 0 <= col < data.shape[1]):
                    outside += 1
                    continue
                z = data[row_i, col]
                if nd is not None and z == nd:
                    outside += 1
                    continue
                h = alt - z
                heights.append(h)
                if h < 0:
                    below.append((lat, lon, alt, z, h))

        if not heights:
            print(f"  {WARN}  No camera poses fell within DSM extent  "
                  f"(outside: {outside})")
        else:
            h_arr = np.array(heights)
            median_h = float(np.median(h_arr))

            ok = _check(
                "All cameras above DSM surface",
                len(below) == 0,
                f"{len(below)} camera(s) below surface" if below else
                f"{len(heights)} poses checked, {outside} outside extent",
            )
            failures += not ok
            if below:
                b = below[0]
                print(f"        First offender: lat={b[0]:.6f} lon={b[1]:.6f} "
                      f"cam={b[2]:.2f} m dsm={b[3]:.2f} m diff={b[4]:.2f} m")

            ok = _check(
                "Median height above ground 0.5–10 m",
                0.5 <= median_h <= 10.0,
                f"{median_h:.2f} m  (range {h_arr.min():.2f}–{h_arr.max():.2f} m)",
            )
            failures += not ok
            if not ok:
                print("        Hint: ~34 m offset → ellipsoidal/NAVD88 datum mismatch; "
                      "large negative → UTM shift sign error")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures == 0:
        print(f"Overall: {PASS}  (all checks passed)")
    else:
        print(f"Overall: {FAIL}  ({failures} check(s) failed)")
    print()

    return failures == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dsm", help="Path to the DSM GeoTIFF to validate")
    ap.add_argument("--poses", default=None,
                    help="Camera poses CSV (default: inferred from DSM filename)")
    ap.add_argument("--expected-res", type=float, default=0.05,
                    help="Expected grid resolution in metres (default: 0.05)")
    ap.add_argument("--z-min", type=float, default=100.0,
                    help="Minimum plausible ellipsoidal elevation in metres (default: 100)")
    ap.add_argument("--z-max", type=float, default=125.0,
                    help="Maximum plausible ellipsoidal elevation in metres (default: 125)")
    args = ap.parse_args()

    dsm_path = Path(args.dsm)
    if not dsm_path.exists():
        print(f"ERROR: DSM not found: {dsm_path}", file=sys.stderr)
        sys.exit(1)

    if args.poses:
        poses_path = Path(args.poses)
    else:
        # Infer: <stem>_camera_poses.csv beside the DSM, stripping _dem suffix
        stem = dsm_path.stem
        if stem.endswith("_dem"):
            stem = stem[:-4]
        poses_path = dsm_path.parent / f"{stem}_camera_poses.csv"

    ok = validate(dsm_path, poses_path, args.expected_res, args.z_min, args.z_max)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
