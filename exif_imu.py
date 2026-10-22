"""
Central EXIF / IMU metadata reading for georeferencing scripts.

Single implementation for GPS (lat, lon, alt) and IMU (Roll, Pitch, Yaw)
from image EXIF. UserComment format: "Roll R Pitch P Yaw Y" (degrees),
e.g. "Roll 1.5 Pitch -8.0 Yaw 120.0" (same as add_imu.py / SU-WaterCam).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# SU-WaterCam / add_imu.py UserComment format
_IMU_RE = re.compile(
    r"Roll\s+([-\d.]+)\s+Pitch\s+([-\d.]+)\s+Yaw\s+([-\d.]+)",
    re.IGNORECASE,
)


_CALIB_RE = re.compile(r"CalibMag\s+(\d+)", re.IGNORECASE)
_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_QUAT_RE = re.compile(r"Quaternion\s+" + ",".join([_NUM] * 4) , re.IGNORECASE)
_GRAV_RE = re.compile(r"Gravity\s+" + ",".join([_NUM] * 3), re.IGNORECASE)
_POSE_TIME_RE = re.compile(r"PoseTime\s+(\S+)", re.IGNORECASE)
_GPSFIX_RE = re.compile(r"GPSFix\s+(\w+)(?:\s+AgeS\s+" + _NUM + ")?", re.IGNORECASE)

# XMP tag names written by newer SU-WaterCam photos (attribute or element form)
_XMP_NUM_TAGS = ("QuaternionW", "QuaternionX", "QuaternionY", "QuaternionZ",
                 "GravityX", "GravityY", "GravityZ", "GPSFixAgeSeconds")
_XMP_STR_TAGS = ("PoseReadTime", "GPSFixSource")


def parse_user_comment(txt: str) -> dict:
    """
    Parse the SU-WaterCam UserComment string. Tolerates old photos (only
    "Roll R Pitch P Yaw Y") and the newer extra words:
      CalibMag n | Quaternion w,x,y,z | Gravity x,y,z | PoseTime <ISO> |
      GPSFix CACHED AgeS n
    Missing items are None. Returns keys: roll_deg, pitch_deg, yaw_deg,
    calib_mag, quaternion (w,x,y,z), gravity (x,y,z), pose_time,
    gps_fix_source, gps_fix_age_s.
    """
    out = {"roll_deg": None, "pitch_deg": None, "yaw_deg": None,
           "calib_mag": None, "quaternion": None, "gravity": None,
           "pose_time": None, "gps_fix_source": None, "gps_fix_age_s": None}
    m = _IMU_RE.search(txt)
    if m:
        out["roll_deg"], out["pitch_deg"], out["yaw_deg"] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)))
    m = _CALIB_RE.search(txt)
    if m:
        out["calib_mag"] = int(m.group(1))
    m = _QUAT_RE.search(txt)
    if m:
        out["quaternion"] = tuple(float(g) for g in m.groups())
    m = _GRAV_RE.search(txt)
    if m:
        out["gravity"] = tuple(float(g) for g in m.groups())
    m = _POSE_TIME_RE.search(txt)
    if m:
        out["pose_time"] = m.group(1)
    m = _GPSFIX_RE.search(txt)
    if m:
        out["gps_fix_source"] = m.group(1).upper()
        if m.group(2) is not None:
            out["gps_fix_age_s"] = float(m.group(2))
    return out


def parse_xmp_pose(data: bytes) -> dict:
    """Extract the optional XMP pose fields from raw XMP/JPEG bytes."""
    txt = data.decode("utf-8", errors="ignore")
    out = {}
    for tag in _XMP_NUM_TAGS + _XMP_STR_TAGS:
        m = re.search(r'(?:\w+:)?%s\s*=\s*"([^"]*)"' % tag, txt) or \
            re.search(r"<(?:\w+:)?%s>([^<]*)<" % tag, txt)
        if not m:
            continue
        val = m.group(1).strip()
        if tag in _XMP_NUM_TAGS:
            try:
                out[tag] = float(val)
            except ValueError:
                continue
        else:
            out[tag] = val
    return out


def _to_float(r) -> float:
    """Convert EXIF rational (IFDRational, tuple, or number) to float."""
    if isinstance(r, (int, float)):
        return float(r)
    if hasattr(r, "numerator") and hasattr(r, "denominator"):
        return float(r.numerator) / float(r.denominator)
    if isinstance(r, (tuple, list)) and len(r) == 2:
        return float(r[0]) / float(r[1])
    return float(r)


def _dms_to_deg(dms, ref: str) -> float:
    """DMS tuple + ref (N/S/E/W) -> decimal degrees."""
    d = _to_float(dms[0])
    m = _to_float(dms[1])
    s = _to_float(dms[2])
    val = d + m / 60.0 + s / 3600.0
    return -val if ref in ("S", "W") else val


def read_gps_imu_from_exif(path: str | Path) -> dict:
    """
    Read GPS and IMU from image EXIF.

    GPS from standard EXIF GPSInfo (Pillow). IMU from EXIF UserComment
    text "Roll R Pitch P Yaw Y" (degrees). Missing fields are None.

    Returns dict with keys:
        lat, lon, alt, roll_deg, pitch_deg, yaw_deg,
        calib_mag (0-3 or None), quaternion, gravity, pose_time,
        gps_fix_source, gps_fix_age_s, xmp (dict of XMP pose fields).
    Heading must only be taken from yaw_deg when calib_mag == 3
    (see unit_config.resolve_heading).
    """
    path = Path(path)
    meta = {
        "lat": None,
        "lon": None,
        "alt": None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "calib_mag": None,
        "quaternion": None,
        "gravity": None,
        "pose_time": None,
        "gps_fix_source": None,
        "gps_fix_age_s": None,
        "xmp": {},
    }

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
    except ImportError:
        return meta

    # GPS from Pillow EXIF
    try:
        img = Image.open(path)
        exif = img._getexif() if hasattr(img, "_getexif") else None
        if exif:
            decoded = {TAGS.get(k, k): v for k, v in exif.items()}
            gps_raw = decoded.get("GPSInfo", {})
            gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            if "GPSLatitude" in gps:
                meta["lat"] = _dms_to_deg(
                    gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N")
                )
            if "GPSLongitude" in gps:
                meta["lon"] = _dms_to_deg(
                    gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E")
                )
            if "GPSAltitude" in gps:
                alt = _to_float(gps["GPSAltitude"])
                if gps.get("GPSAltitudeRef", 0) == 1:
                    alt = -alt
                meta["alt"] = alt
    except Exception:
        pass

    # IMU from piexif UserComment
    try:
        import piexif
        ed = piexif.load(str(path))
        uc = (ed.get("Exif") or {}).get(piexif.ExifIFD.UserComment)
        if uc:
            txt = uc.decode(errors="ignore") if isinstance(uc, bytes) else str(uc)
            meta.update(parse_user_comment(txt))
    except Exception:
        pass

    # Optional XMP pose fields (newer photos); the packet is near the file start
    try:
        with open(path, "rb") as f:
            head = f.read(262144)
        if b"<x:xmpmeta" in head or b"<?xpacket" in head:
            meta["xmp"] = parse_xmp_pose(head)
    except Exception:
        pass

    return meta
