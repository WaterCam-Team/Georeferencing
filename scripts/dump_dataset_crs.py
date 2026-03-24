#!/usr/bin/env python3
"""
Print CRS, extent, and quick elevation stats for DEM rasters and LAS/LAZ clouds.

Helps verify metadata when changing study area (horizontal CRS, nodata, vertical hints).

Examples:
  python scripts/dump_dataset_crs.py --dem site.tif
  python scripts/dump_dataset_crs.py --las survey.laz --las-crs 32618
  python scripts/dump_dataset_crs.py --dem a.img --dem b.tif --las c.las
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root: scripts/ -> parent
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from vertical_datum import infer_vertical_datum_from_rasterio
except ImportError:
    infer_vertical_datum_from_rasterio = None  # type: ignore[misc, assignment]


def _print_dem(path: Path) -> None:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        print(f"[DEM] {path}: install rasterio to inspect rasters")
        return

    if not path.is_file():
        print(f"[DEM] {path}: not found")
        return

    with rasterio.open(path) as src:
        print(f"\n=== DEM: {path} ===")
        print(f"  Driver: {src.driver}")
        print(f"  Size: {src.width} × {src.height}")
        crs = src.crs
        print(f"  CRS: {crs}")
        if crs is not None:
            try:
                c = crs.to_epsg()
                if c:
                    print(f"  EPSG (if defined): {c}")
            except Exception:
                pass
        wgs84 = None
        try:
            from rasterio.crs import CRS as RCRS

            if crs is not None:
                wgs84 = transform_bounds(crs, RCRS.from_epsg(4326), *src.bounds)
        except Exception:
            wgs84 = None
        print(f"  Bounds (file CRS): {src.bounds}")
        if wgs84:
            print(f"  Bounds (WGS84 approx): {wgs84}")
        print(f"  Transform: {src.transform}")
        nodata = src.nodata
        print(f"  Nodata: {nodata}")
        for i in range(1, min(src.count, 3) + 1):
            desc = src.descriptions[i - 1]
            print(f"  Band {i} description: {desc!r}")
        tags = src.tags()
        if tags:
            stat_keys = [k for k in tags if k.upper().startswith("STATISTICS")]
            if stat_keys:
                print("  Tag statistics (subset):")
                for k in sorted(stat_keys)[:12]:
                    print(f"    {k}={tags[k]}")
        if infer_vertical_datum_from_rasterio:
            inferred = infer_vertical_datum_from_rasterio(src)
            if inferred:
                print(f"  Inferred vertical hint (from CRS WKT): {inferred}")
            else:
                print("  Inferred vertical hint: (none — check dataset documentation)")
        pixels = src.width * src.height
        _MAX_FULL_READ = 25_000_000
        if pixels > _MAX_FULL_READ:
            print(
                f"  Band 1 min/max: skipped full read ({pixels} pixels > {_MAX_FULL_READ}); "
                "use gdalinfo or QGIS for statistics on large rasters."
            )
        else:
            try:
                import numpy as np

                data = src.read(1, masked=True)
                if isinstance(data, np.ma.MaskedArray):
                    valid = data.compressed()
                else:
                    valid = np.asarray(data, dtype=float).ravel()
                if valid.size > 0:
                    arr = np.asarray(valid, dtype=float)
                    print(f"  Band 1 data min/max (full read): {float(arr.min()):.6f} / {float(arr.max()):.6f}")
            except Exception as e:
                print(f"  Band 1 stats: could not compute ({e})")


def _print_las(path: Path, las_crs_epsg: int | None) -> None:
    try:
        import laspy
    except ImportError:
        print(f"[LAS] {path}: install laspy to inspect point clouds")
        return

    if not path.is_file():
        print(f"[LAS] {path}: not found")
        return

    laz_backend = None
    if str(path).lower().endswith(".laz"):
        try:
            available = laspy.compression.LazBackend.detect_available()
            if available:
                laz_backend = available[0]
        except Exception:
            pass

    try:
        if laz_backend is not None:
            las = laspy.read(str(path), laz_backend=laz_backend)
        else:
            las = laspy.read(str(path))
    except Exception as e:
        print(f"[LAS] {path}: read failed: {e}")
        return

    x = las.x
    y = las.y
    z = las.z
    n = len(x)

    print(f"\n=== LAS/LAZ: {path} ===")
    print(f"  Points: {n}")
    print(f"  X min/max: {float(x.min()):.6f} / {float(x.max()):.6f}")
    print(f"  Y min/max: {float(y.min()):.6f} / {float(y.max()):.6f}")
    print(f"  Z min/max: {float(z.min()):.6f} / {float(z.max()):.6f}")

    pc_crs = None
    if hasattr(las, "header") and hasattr(las.header, "parse_crs"):
        try:
            pc_crs = las.header.parse_crs()
        except Exception:
            pass
    if pc_crs is None and las_crs_epsg is not None:
        try:
            from pyproj import CRS

            pc_crs = CRS.from_epsg(las_crs_epsg)
        except Exception:
            pass

    if pc_crs is not None:
        print(f"  CRS (from file or --las-crs): {pc_crs!r}")
        if hasattr(pc_crs, "to_epsg"):
            e = pc_crs.to_epsg()
            if e:
                print(f"  EPSG: {e}")
    else:
        print("  CRS: (not in header — pass --las-crs EPSG for this tool / georeference_terrain)")
    print("  Vertical: (not standardized in LAS — use provider docs + georeference_terrain datums)")


def main() -> int:
    p = argparse.ArgumentParser(description="Dump CRS and quick stats for DEM and LAS/LAZ inputs")
    p.add_argument("--dem", action="append", default=[], help="Raster path (repeatable)")
    p.add_argument("--las", action="append", default=[], help="LAS/LAZ path (repeatable)")
    p.add_argument("--las-crs", type=int, default=None, help="EPSG for LAS when file has no CRS")
    args = p.parse_args()

    if not args.dem and not args.las:
        p.print_help()
        return 2

    for d in args.dem:
        _print_dem(Path(d))
    for L in args.las:
        _print_las(Path(L), args.las_crs)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
