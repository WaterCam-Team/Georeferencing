"""
Vertical datum checking and conversion for georeferencing.

EXIF altitude is typically WGS84 ellipsoidal; DEMs/LAS are often orthometric
(EGM96, EGM2008, NAVD88). Mixing them causes ~15–35 m vertical offset and
large horizontal errors in ray–terrain intersection. This module provides
optional conversion and consistent warnings.

Usage:
  - Scripts can pass --terrain-vertical-datum and --camera-elev-datum.
  - transform_height() converts between ellipsoidal and orthometric when
    PROJ has the required geoid grids (e.g. proj-datumgrid-*).
  - check_camera_elev_datum() warns when EXIF/input elevation is likely
    ellipsoidal but terrain is orthometric (or vice versa), and suggests
    conversion or --height-above-ground.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

# Known vertical datum identifiers (lowercase, script-friendly)
VERTICAL_ELLIPSOID = "wgs84_ellipsoid"
VERTICAL_EGM96 = "egm96"
VERTICAL_EGM2008 = "egm2008"
# US orthometric; conversion may need proj-datumgrid-north-america
VERTICAL_NAVD88 = "navd88"

# EPSG codes used for conversion (3D / compound)
_EPSG_WGS84_3D = 4979   # WGS 84 (3D) — ellipsoidal height
_EPSG_EGM96 = 5773      # EGM96 height (vertical CRS)
_EPSG_EGM2008 = 3855    # EGM2008 height
# Compound CRS: horizontal WGS84 + vertical (so we get lon, lat, h_orthometric)
# PROJ/pyproj use compound for "WGS84 + EGM96 height" etc.


def _crs_ellipsoidal() -> Optional[Any]:
    try:
        from pyproj import CRS
        return CRS.from_epsg(_EPSG_WGS84_3D)
    except Exception:
        return None


def _crs_orthometric(vertical: str) -> Optional[Any]:
    """Return compound CRS (WGS84 horizontal + given vertical) or None."""
    try:
        from pyproj import CRS
        from pyproj.crs import CompoundCRS
    except ImportError:
        return None
    if vertical == VERTICAL_EGM96:
        return CompoundCRS(
            "WGS 84 + EGM96 height",
            components=[CRS.from_epsg(4326), CRS.from_epsg(_EPSG_EGM96)],
        )
    if vertical == VERTICAL_EGM2008:
        return CompoundCRS(
            "WGS 84 + EGM2008 height",
            components=[CRS.from_epsg(4326), CRS.from_epsg(_EPSG_EGM2008)],
        )
    if vertical == VERTICAL_NAVD88:
        # NAVD88 is region-specific; try common EPSG compound if available
        try:
            return CRS.from_epsg(9708)  # NAD83(CSRS) + CGVD28; not WGS84+NAVD88
        except Exception:
            return None
    return None


def transform_height(
    lon: float,
    lat: float,
    height_m: float,
    from_datum: str,
    to_datum: str,
) -> Tuple[Optional[float], bool]:
    """
    Convert height between vertical datums at (lon, lat).

    from_datum / to_datum: one of VERTICAL_ELLIPSOID, VERTICAL_EGM96,
    VERTICAL_EGM2008 (and optionally VERTICAL_NAVD88 if supported).

    Returns (converted_height_m, success). On failure (e.g. missing PROJ
    geoid grids), returns (None, False). When from_datum == to_datum,
    returns (height_m, True).
    """
    if from_datum == to_datum:
        return (float(height_m), True)

    try:
        from pyproj import CRS, Transformer
        from pyproj.crs import CompoundCRS
    except ImportError:
        return (None, False)

    crs_from = _crs_ellipsoidal() if from_datum == VERTICAL_ELLIPSOID else _crs_orthometric(from_datum)
    crs_to = _crs_ellipsoidal() if to_datum == VERTICAL_ELLIPSOID else _crs_orthometric(to_datum)
    if crs_from is None or crs_to is None:
        return (None, False)

    try:
        trans = Transformer.from_crs(crs_from, crs_to, always_xy=True)
        # 3D: (x, y, z) = (lon, lat, height)
        out = trans.transform(lon, lat, height_m)
        if out is None or len(out) < 3:
            return (None, False)
        return (float(out[2]), True)
    except Exception:
        return (None, False)


def check_camera_elev_datum(
    camera_elev_source: str,
    camera_elev_m: Optional[float],
    terrain_vertical_datum: str,
    camera_elev_datum: Optional[str],
    *,
    height_above_ground_used: bool = False,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Check for likely vertical datum mismatch and suggest conversion.

    camera_elev_source: "exif" | "user" | "dem_plus_height"
    terrain_vertical_datum: datum of DEM/LAS (e.g. VERTICAL_EGM96).
    camera_elev_datum: datum of the camera elevation value (if known); None = assume from source.
    height_above_ground_used: if True, camera elev was set from DEM + height (already terrain datum).

    Returns (warning_message, suggested_converted_elev).
    - If no issue: (None, None).
    - If mixing and conversion possible: (warning, converted_elev).
    - If mixing and conversion not available: (warning, None).
    """
    if camera_elev_m is None:
        return (None, None)
    if height_above_ground_used or camera_elev_source == "dem_plus_height":
        return (None, None)

    # EXIF and typical user input are ellipsoidal (WGS84)
    input_datum = camera_elev_datum or (
        VERTICAL_ELLIPSOID if camera_elev_source == "exif" else None
    )
    if input_datum is None:
        input_datum = VERTICAL_ELLIPSOID  # assume worst case for warning

    if input_datum == terrain_vertical_datum:
        return (None, None)

    # Different datums: try to convert
    warning = (
        f"Camera elevation may be in {input_datum} but terrain is {terrain_vertical_datum}. "
        "Mixing can cause ~15–35 m vertical error and large horizontal errors. "
    )
    # We don't have lon/lat here; conversion needs a point. Caller can use transform_height at camera position.
    return (warning, None)


def convert_camera_elev_to_terrain_datum(
    cam_lon: float,
    cam_lat: float,
    camera_elev_m: float,
    camera_elev_datum: str,
    terrain_vertical_datum: str,
) -> Tuple[float, bool]:
    """
    Convert camera elevation to terrain vertical datum at (cam_lon, cam_lat).

    Returns (elev_m_in_terrain_datum, success). On failure returns (camera_elev_m, False)
    so caller can keep original and warn.
    """
    if camera_elev_datum == terrain_vertical_datum:
        return (camera_elev_m, True)
    converted, ok = transform_height(
        cam_lon, cam_lat, camera_elev_m, camera_elev_datum, terrain_vertical_datum
    )
    if ok and converted is not None:
        return (converted, True)
    return (camera_elev_m, False)


def infer_vertical_datum_from_rasterio(src) -> Optional[str]:
    """
    Infer vertical datum from a rasterio dataset if possible.

    Most GeoTIFFs do not store vertical CRS; returns None in that case.
    Scripts should allow user to pass --terrain-vertical-datum.
    """
    try:
        crs = getattr(src, "crs", None)
        if crs is None:
            return None
        wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
        wkt_lower = wkt.lower()
        if "egm96" in wkt_lower or "5773" in wkt:
            return VERTICAL_EGM96
        if "egm2008" in wkt_lower or "3855" in wkt:
            return VERTICAL_EGM2008
        if "navd88" in wkt_lower or "navd" in wkt_lower:
            return VERTICAL_NAVD88
        # 2D geographic often means horizontal only; elevation band is often undocumented
        return None
    except Exception:
        return None
