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
import os
from typing import Any, Optional


_REQUIRED_FIELDS: list[str] = []          # none strictly required
_KNOWN_FIELDS: set[str] = {
    "unit_id", "calibration", "mount_height_m",
    "heading_deg", "pitch_deg", "roll_deg",
    "camera_lat", "camera_lon", "camera_alt_ellipsoid_m",
    "camera_elev_datum", "notes",
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
    def camera_lat(self) -> Optional[float]:
        v = self._d.get("camera_lat")
        return float(v) if v is not None else None

    @property
    def camera_lon(self) -> Optional[float]:
        v = self._d.get("camera_lon")
        return float(v) if v is not None else None

    @property
    def camera_alt_ellipsoid_m(self) -> Optional[float]:
        v = self._d.get("camera_alt_ellipsoid_m")
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
    def notes(self) -> str:
        return self._d.get("notes", "")

    # ── Merge helpers ─────────────────────────────────────────────────────────

    def resolve_position(
        self,
        cli_lat: Optional[float],
        cli_lon: Optional[float],
        exif_lat: Optional[float] = None,
        exif_lon: Optional[float] = None,
    ) -> tuple[Optional[float], Optional[float], str]:
        """
        Return (lat, lon, source_label).
        Source: 'cli' > 'unit_config' > 'exif' > (None, None, 'none')
        """
        if cli_lat is not None and cli_lon is not None:
            return float(cli_lat), float(cli_lon), "cli"
        if self.camera_lat is not None and self.camera_lon is not None:
            return self.camera_lat, self.camera_lon, "unit_config"
        if exif_lat is not None and exif_lon is not None:
            return float(exif_lat), float(exif_lon), "exif"
        return None, None, "none"

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
        """
        if cli_override is not None:
            return float(cli_override), "cli"
        if self.heading_deg is not None:
            return self.heading_deg, "unit_config"
        if exif_yaw is not None:
            return float(exif_yaw), "exif_yaw"
        if exif_gps_track is not None:
            return float(exif_gps_track), "exif_gps_track"
        return 0.0, "default"

    def resolve_pitch(self, cli_override: Optional[float],
                      exif_pitch: Optional[float]) -> tuple[float, str]:
        if cli_override is not None:
            return float(cli_override), "cli"
        if self.pitch_deg is not None:
            return self.pitch_deg, "unit_config"
        if exif_pitch is not None:
            return float(exif_pitch), "exif"
        return 0.0, "default"

    def resolve_roll(self, cli_override: Optional[float],
                     exif_roll: Optional[float]) -> tuple[float, str]:
        if cli_override is not None:
            return float(cli_override), "cli"
        if self.roll_deg is not None:
            return self.roll_deg, "unit_config"
        if exif_roll is not None:
            return float(exif_roll), "exif"
        return 0.0, "default"

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
        if self.camera_lat is not None:
            parts.append(f"lat={self.camera_lat:.6f}")
        if self.camera_lon is not None:
            parts.append(f"lon={self.camera_lon:.6f}")
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
