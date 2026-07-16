"""
Per-unit sensor configuration loader.

Each physical sensor unit has fixed characteristics (mount height, camera
heading if fixed-mount, calibration file, GPS altitude datum) that should
not be re-entered on every run. This module loads a JSON unit config and
merges it with per-image EXIF values and CLI overrides.

Precedence (highest to lowest):
  1. Explicit CLI args (e.g. --heading, --height-above-ground)
  2. Unit config JSON  (e.g. unit_config_UFO006.json)
  3. Per-image EXIF    (GPS lat/lon, altitude, IMU yaw/pitch/roll)
  4. Script defaults

Unit config JSON schema
-----------------------
{
  "unit_id":            "UFO-006",        // human identifier
  "calibration":        "./calibration.json",
  "mount_height_m":     0.8382,           // camera height above ground (m); null = derive from EXIF altitude
  "heading_deg":        265.0,            // fixed mount heading (deg, 0=N); null = use EXIF Yaw
  "pitch_deg":          null,             // fixed mount pitch; null = use EXIF Pitch
  "roll_deg":           null,             // fixed mount roll;  null = use EXIF Roll
  "camera_elev_datum":  "wgs84_ellipsoid",// datum of GPS altitude from this unit's GPS module
  "notes":              ""               // free-text description
}

Fields set to null (or omitted) are filled from EXIF or left at script defaults.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional


_REQUIRED_FIELDS: list[str] = []          # none strictly required
_KNOWN_FIELDS: set[str] = {
    "unit_id", "calibration", "mount_height_m",
    "heading_deg", "pitch_deg", "roll_deg",
    "camera_elev_datum", "notes",
    # IMU / BNO055 fields
    "imu_mount_offset_deg",        # physical mount rotation to add to raw heading (e.g. 180.0 if rotated)
    "imu_magnetic_declination_deg",# site magnetic declination (degrees, positive = East)
    "imu_heading_correction_deg",  # residual correction from validate_heading(); update after calibration
    "imu_calibration_file",        # path to BNO055 offset JSON saved by bno055_calibration.py
}


class UnitConfig:
    def __init__(self, data: dict):
        unknown = set(data) - _KNOWN_FIELDS
        if unknown:
            print(f"[unit_config] Unknown fields (ignored): {unknown}")
        self._d = data

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def unit_id(self) -> Optional[str]:
        return self._d.get("unit_id")

    @property
    def calibration(self) -> Optional[str]:
        return self._d.get("calibration") or None

    @property
    def mount_height_m(self) -> Optional[float]:
        v = self._d.get("mount_height_m")
        return float(v) if v is not None else None

    @property
    def heading_deg(self) -> Optional[float]:
        v = self._d.get("heading_deg")
        return float(v) if v is not None else None

    @property
    def pitch_deg(self) -> Optional[float]:
        v = self._d.get("pitch_deg")
        return float(v) if v is not None else None

    @property
    def roll_deg(self) -> Optional[float]:
        v = self._d.get("roll_deg")
        return float(v) if v is not None else None

    @property
    def camera_elev_datum(self) -> Optional[str]:
        return self._d.get("camera_elev_datum") or None

    @property
    def imu_mount_offset_deg(self) -> float:
        v = self._d.get("imu_mount_offset_deg")
        return float(v) if v is not None else 0.0

    @property
    def imu_magnetic_declination_deg(self) -> float:
        v = self._d.get("imu_magnetic_declination_deg")
        return float(v) if v is not None else 0.0

    @property
    def imu_heading_correction_deg(self) -> float:
        v = self._d.get("imu_heading_correction_deg")
        return float(v) if v is not None else 0.0

    @property
    def imu_calibration_file(self) -> Optional[str]:
        return self._d.get("imu_calibration_file") or None

    @property
    def notes(self) -> str:
        return self._d.get("notes", "")

    # ── Merge helpers ─────────────────────────────────────────────────────────

    def resolve_calibration(self, cli_override: Optional[str] = None,
                            config_dir: str = ".") -> str:
        """
        Return calibration path, searching relative to the config file's directory.
        CLI override wins; unit config next; falls back to './calibration.json'.
        """
        path = cli_override or self.calibration or "calibration.json"
        if not os.path.isabs(path):
            path = os.path.join(config_dir, path)
        return os.path.normpath(path)

    def resolve_heading(self, cli_override: Optional[float],
                        exif_yaw: Optional[float],
                        exif_gps_track: Optional[float] = None) -> tuple[float, str]:
        """
        Return (heading_deg, source_label).
        Source: 'cli' > 'unit_config' > 'exif_yaw' > 'exif_gps_track' > 0.0

        When heading comes from EXIF, per-unit IMU corrections are applied:
        mount offset, magnetic declination, and post-calibration residual.
        unit_config and cli values are assumed to already be true headings.
        """
        if cli_override is not None:
            return float(cli_override), "cli"
        if self.heading_deg is not None:
            return self.heading_deg, "unit_config"
        imu_offset = (self.imu_mount_offset_deg
                      + self.imu_magnetic_declination_deg
                      + self.imu_heading_correction_deg)
        if exif_yaw is not None:
            corrected = (float(exif_yaw) + imu_offset) % 360.0
            return corrected, "exif_yaw_corrected"
        if exif_gps_track is not None:
            return float(exif_gps_track), "exif_gps_track"
        return 0.0, "default"

    def resolve_pitch_roll(self,
                           cli_pitch: Optional[float], cli_roll: Optional[float],
                           exif_pitch: Optional[float], exif_roll: Optional[float],
                           ) -> tuple[float, float, str]:
        """
        Return (pitch_deg, roll_deg, source_label).
        Source: 'cli' > 'unit_config' > 'exif_corrected' > 'exif' > 0.0

        Raw EXIF pitch comes straight from the BNO055 hardware Euler register
        (SU-WaterCam's bno055_imu.py / add_metadata.py write it unmodified,
        by design). Its native sign convention is the OPPOSITE of this
        module's (0deg=level, -90deg=straight down) — confirmed empirically
        2026-07-15 via RTK-validated GCP refinement on two independent units
        (imu_mount_offset_deg=0 and =180): the raw/passthrough pitch pointed
        the camera above the horizon in both cases.

        Root cause (confirmed 2026-07-16 against the Bosch datasheet,
        BST-BNO055-DS000-18 rev1.8, Table 3-13 "Rotation angle conventions",
        p.32, and the UNIT_SEL register default in the Page-0 register map,
        ~p.56): the BNO055 has two selectable Euler output formats, Android
        vs. Windows, and PITCH is defined with opposite sign between them
        ("turning clockwise decreases values" in Android vs. "increases
        values" in Windows). UNIT_SEL powers on to 0x80 (Android format,
        bit 7 set) and the vendored `adafruit_bno055` driver never writes
        UNIT_SEL, so the sensor stays in Android format — opposite of what
        this module assumes. Roll and Heading/Yaw are explicitly
        format-independent per the same table (identical convention in both
        formats), which is why only pitch needs this correction; empirically
        confirmed too (2026-07-16 roll-isolation test against RTK ground
        truth showed no sign inversion for roll).

        So raw EXIF pitch is negated first, before the mount-offset rotation
        below is applied. Roll is left as-is.

        When both values come from EXIF, the mount rotation is then applied:
            corrected_pitch = pitch * cos(θ) − roll * sin(θ)
            corrected_roll  = pitch * sin(θ) + roll * cos(θ)
        where θ = imu_mount_offset_deg and pitch is the sign-corrected value
        above. This un-does the apparent pitch/roll swap introduced by
        mounting the sensor at an arbitrary yaw angle relative to the camera
        body.

        CLI and unit_config values are assumed to already be in camera frame.
        """
        if cli_pitch is not None or cli_roll is not None:
            p = float(cli_pitch) if cli_pitch is not None else 0.0
            r = float(cli_roll)  if cli_roll  is not None else 0.0
            return p, r, "cli"

        if self.pitch_deg is not None or self.roll_deg is not None:
            p = self.pitch_deg if self.pitch_deg is not None else 0.0
            r = self.roll_deg  if self.roll_deg  is not None else 0.0
            return p, r, "unit_config"

        if exif_pitch is not None and exif_roll is not None:
            theta = math.radians(self.imu_mount_offset_deg)
            p = -float(exif_pitch)
            r = float(exif_roll)
            corrected_pitch = p * math.cos(theta) - r * math.sin(theta)
            corrected_roll  = p * math.sin(theta) + r * math.cos(theta)
            return corrected_pitch, corrected_roll, "exif_corrected"

        # Partial EXIF — only one axis available, pass through without rotation
        if exif_pitch is not None:
            return -float(exif_pitch), 0.0, "exif"
        if exif_roll is not None:
            return 0.0, float(exif_roll), "exif"

        return 0.0, 0.0, "default"

    def resolve_mount_height(self, cli_override: Optional[float]) -> tuple[Optional[float], str]:
        if cli_override is not None:
            return float(cli_override), "cli"
        if self.mount_height_m is not None:
            return self.mount_height_m, "unit_config"
        return None, "none"

    def resolve_camera_elev_datum(self, cli_override: Optional[str]) -> str:
        from vertical_datum import VERTICAL_ELLIPSOID
        return cli_override or self.camera_elev_datum or VERTICAL_ELLIPSOID

    def summary(self) -> str:
        parts = []
        if self.unit_id:
            parts.append(f"unit={self.unit_id}")
        if self.mount_height_m is not None:
            parts.append(f"mount={self.mount_height_m:.4f}m")
        if self.heading_deg is not None:
            parts.append(f"heading={self.heading_deg:.1f}°")
        if self.pitch_deg is not None:
            parts.append(f"pitch={self.pitch_deg:.1f}°")
        if self.calibration:
            parts.append(f"calib={self.calibration}")
        return "  ".join(parts) if parts else "(empty)"


# ── Public API ────────────────────────────────────────────────────────────────

def load(path: str) -> UnitConfig:
    """Load a unit config JSON file. Returns a UnitConfig."""
    with open(path) as f:
        data = json.load(f)
    cfg = UnitConfig(data)
    uid = cfg.unit_id or os.path.basename(path)
    print(f"[unit_config] Loaded '{uid}': {cfg.summary()}")
    return cfg


def empty() -> UnitConfig:
    """Return an empty unit config (all values None / defaults)."""
    return UnitConfig({})


def add_argument(parser) -> None:
    """Add --unit-config argument to an argparse parser."""
    parser.add_argument(
        "--unit-config", "-u",
        default=None,
        metavar="JSON",
        help="Path to unit config JSON (mount height, heading, calibration, etc.)",
    )


def from_args(args) -> UnitConfig:
    """Load unit config from parsed args.unit_config, or return empty config."""
    path = getattr(args, "unit_config", None)
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Unit config not found: {path}")
        return load(path)
    return empty()
