"""
drone_dsm_ingest.py
===================
Accept a drone-produced DSM/DEM GeoTIFF from any tool — DJI Terra, Pix4D,
OpenDroneMap, DroneDeploy — validate it, reproject to the project CRS if
needed, and print the ready `georeference_terrain.py` command.

Note: this script validates elevation data (z range, coverage, vertical datum).
It is not suitable for orthophotos or RGB rasters; use drone_gcp_match.py for
orthomosaic GCP matching.

Usage
-----
    python scripts/drone_dsm_ingest.py dsm.tif
    python scripts/drone_dsm_ingest.py dsm.tif --out-dir ./drone_dsm --target-res 0.05
    python scripts/drone_dsm_ingest.py dsm.tif --skip-reproject

Vertical datum guidance
-----------------------
Most drone tools embed only a 2D horizontal CRS; the vertical convention is:

  Tool              | Default vertical
  ------------------|------------------
  DJI Terra         | WGS84 ellipsoidal
  OpenDroneMap      | WGS84 ellipsoidal (default)
  Pix4D Desktop     | Depends on GCP/base-station datum; often NAVD88 in US
  DroneDeploy       | WGS84 ellipsoidal

Pass --terrain-vertical-datum to georeference_terrain.py accordingly.
Syracuse NY geoid separation (WGS84 ellipsoid → NAVD88): ≈ +34 m.

Requires: rasterio, numpy, pyproj
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import rasterio
    from rasterio.crs import CRS as RasterioCRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError:
    sys.exit("Install rasterio: pip install rasterio")

try:
    from vertical_datum import infer_vertical_datum_from_rasterio, VERTICAL_ELLIPSOID
except ImportError:
    infer_vertical_datum_from_rasterio = None  # type: ignore[assignment]
    VERTICAL_ELLIPSOID = "wgs84_ellipsoid"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[36mINFO\033[0m"


@dataclass
class IngestResult:
    src_path: Path
    out_path: Optional[Path]
    src_epsg: Optional[int]
    target_epsg: int
    reprojected: bool
    src_res_x: float
    src_res_y: float
    out_res: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]
    coverage_frac: Optional[float]
    vertical_hint: Optional[str]
    checks_passed: bool
    failures: list[str] = field(default_factory=list)


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    return passed


def _warn(label: str, detail: str = "") -> None:
    line = f"  {WARN}  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)


def ingest_dsm(
    src_path: Path,
    out_dir: Path,
    target_epsg: int = 6347,
    target_res: float = 0.05,
    skip_reproject: bool = False,
) -> IngestResult:
    """
    Validate a drone DSM GeoTIFF and reproject to target_epsg if needed.
    Returns an IngestResult describing what was done.
    """
    failures: list[str] = []

    print(f"\n=== Drone DSM Ingest: {src_path.name} ===\n")

    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_res_x = abs(src.transform.a)
        src_res_y = abs(src.transform.e)
        nd = src.nodata
        bounds = src.bounds
        src_epsg = src_crs.to_epsg() if src_crs else None
        total_count = src.width * src.height

        # Compute stats block-wise to avoid loading the full band into memory.
        # Drone DSMs at 3–5 cm resolution can be multi-GB.
        _nd_is_nan = nd is not None and np.isnan(nd)
        valid_count = 0
        z_min_acc = float("inf")
        z_max_acc = float("-inf")

        for _, window in src.block_windows(1):
            block = src.read(1, window=window).astype(np.float64)
            if _nd_is_nan:
                bmask = ~np.isnan(block)
            elif nd is not None:
                bmask = (block != nd) & ~np.isnan(block)
            else:
                bmask = ~np.isnan(block)
            vb = block[bmask]
            valid_count += vb.size
            if vb.size > 0:
                z_min_acc = min(z_min_acc, float(vb.min()))
                z_max_acc = max(z_max_acc, float(vb.max()))

        vertical_hint = (
            infer_vertical_datum_from_rasterio(src)
            if infer_vertical_datum_from_rasterio is not None
            else None
        )

    # --- Checks ---
    print("Input raster")
    ok = _check("CRS present", src_crs is not None,
                f"EPSG:{src_epsg}" if src_epsg else "(no CRS)")
    if not ok:
        failures.append("no CRS")

    print(f"  {INFO}  Resolution: {src_res_x:.4f} × {src_res_y:.4f} m")
    print(f"  {INFO}  Bounds: W={bounds.left:.2f} S={bounds.bottom:.2f} "
          f"E={bounds.right:.2f} N={bounds.top:.2f}")

    z_min = z_max = coverage_frac = None
    if valid_count > 0:
        z_min, z_max = z_min_acc, z_max_acc
        coverage_frac = float(valid_count) / float(total_count)
        print(f"  {INFO}  Elevation range: {z_min:.2f}–{z_max:.2f} m")
        print(f"  {INFO}  Coverage: {coverage_frac:.1%} of bounding box "
              f"({valid_count:,} / {total_count:,} cells)")
        _check("Has valid elevation cells", True)
    else:
        _check("Has valid elevation cells", False, "all cells are nodata")
        failures.append("no valid cells")

    print("\nVertical datum")
    if vertical_hint:
        print(f"  {INFO}  Inferred from CRS WKT: {vertical_hint}")
    else:
        _warn("Vertical datum not in CRS WKT",
              "assuming WGS84 ellipsoidal (typical for DJI Terra / ODM)")
        vertical_hint = VERTICAL_ELLIPSOID

    # --- Reprojection decision ---
    if src_epsg is not None:
        needs_reproject = (src_epsg != target_epsg)
    elif src_crs is not None:
        # Valid CRS with no EPSG code (custom WKT, ESRI definition, etc.)
        # — compare objects directly so we don't silently skip reprojection.
        needs_reproject = not src_crs.equals(RasterioCRS.from_epsg(target_epsg))
    else:
        # No CRS at all — handled cleanly in the block below.
        needs_reproject = False
    reprojected = False
    out_path: Optional[Path] = None
    out_res: Optional[float] = None

    print("\nReprojection")
    if src_crs is None:
        _warn("Skipping reprojection — no CRS in source file",
              "pass a file with an embedded CRS or reproject manually with gdalwarp")
        failures.append("no CRS; reprojection skipped")
        out_path = src_path
    elif skip_reproject:
        print(f"  {INFO}  --skip-reproject: leaving in source CRS")
        out_path = src_path
    elif not needs_reproject:
        print(f"  {PASS}  Already EPSG:{target_epsg}; no reprojection needed")
        out_path = src_path
    else:
        dst_crs = RasterioCRS.from_epsg(target_epsg)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{src_path.stem}_ingest.tif"

        print(f"  {INFO}  EPSG:{src_epsg} → EPSG:{target_epsg} "
              f"at {target_res} m resolution ...")

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs,
                src.width, src.height,
                *src.bounds,
                resolution=target_res,
            )
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            })

            with rasterio.open(out_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=nd,
                        dst_nodata=nd,
                    )

        reprojected = True
        out_res = target_res
        _check(f"Reprojected to EPSG:{target_epsg}", True, f"→ {out_path}")

    checks_passed = len(failures) == 0

    print()
    _print_usage(out_path, vertical_hint, checks_passed)

    return IngestResult(
        src_path=src_path,
        out_path=out_path,
        src_epsg=src_epsg,
        target_epsg=target_epsg,
        reprojected=reprojected,
        src_res_x=src_res_x,
        src_res_y=src_res_y,
        out_res=out_res,
        z_min=z_min,
        z_max=z_max,
        coverage_frac=coverage_frac,
        vertical_hint=vertical_hint,
        checks_passed=checks_passed,
        failures=failures,
    )


def _print_usage(out_path: Optional[Path], vertical_hint: Optional[str],
                 ok: bool) -> None:
    if not ok or out_path is None:
        print(f"  {FAIL}  Ingest incomplete — address errors above before continuing.")
        return

    datum_flag = (
        f"--terrain-vertical-datum {vertical_hint}"
        if vertical_hint
        else "--terrain-vertical-datum wgs84_ellipsoid  # adjust if NAVD88"
    )

    print("Ready — use in georeference_terrain.py:")
    print(f"  python georeference_terrain.py \\")
    print(f"      --dem {out_path} \\")
    print(f"      {datum_flag} \\")
    print(f"      --lat <camera_lat> --lon <camera_lon> \\")
    print(f"      --heading <h> --pitch <p> --roll <r> \\")
    print(f"      --height-above-ground <measured_m>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and ingest a drone DSM GeoTIFF into the georeferencing pipeline.",
    )
    p.add_argument("dsm", help="Path to drone DSM GeoTIFF")
    p.add_argument(
        "--out-dir", default="./drone_dsm",
        help="Output directory for reprojected GeoTIFF (default: ./drone_dsm)"
    )
    p.add_argument(
        "--target-crs", type=int, default=6347, metavar="EPSG",
        help="Target CRS EPSG code (default: 6347 — NAD83(2011)/UTM 18N)"
    )
    p.add_argument(
        "--target-res", type=float, default=0.05, metavar="M",
        help="Target pixel resolution in metres (default: 0.05)"
    )
    p.add_argument(
        "--skip-reproject", action="store_true",
        help="Skip reprojection; validate and print stats only"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    src_path = Path(args.dsm)
    if not src_path.exists():
        print(f"[ERR] DSM not found: {src_path}", file=sys.stderr)
        return 2

    result = ingest_dsm(
        src_path=src_path,
        out_dir=Path(args.out_dir),
        target_epsg=args.target_crs,
        target_res=args.target_res,
        skip_reproject=args.skip_reproject,
    )
    return 0 if result.checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
