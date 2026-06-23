"""Tests for UnitConfig.resolve_heading IMU correction logic."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unit_config


def _cfg(data: dict) -> unit_config.UnitConfig:
    return unit_config.UnitConfig(data)


class TestResolveHeading:
    def test_cli_wins_over_everything(self):
        cfg = _cfg({"heading_deg": 90.0, "imu_mount_offset_deg": 180.0})
        heading, src = cfg.resolve_heading(cli_override=45.0, exif_yaw=10.0)
        assert heading == 45.0
        assert src == "cli"

    def test_unit_config_wins_over_exif(self):
        cfg = _cfg({"heading_deg": 70.0, "imu_mount_offset_deg": 180.0})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=10.0)
        assert heading == 70.0
        assert src == "unit_config"

    def test_unit_config_heading_not_offset_corrected(self):
        # heading_deg is a true heading — mount offset must NOT be added
        cfg = _cfg({"heading_deg": 70.0, "imu_mount_offset_deg": 180.0})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=None)
        assert heading == 70.0

    def test_exif_yaw_gets_mount_offset(self):
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=250.0)
        assert abs(heading - 70.0) < 0.001   # (250 + 180) % 360
        assert src == "exif_yaw_corrected"

    def test_exif_yaw_gets_declination(self):
        cfg = _cfg({"imu_magnetic_declination_deg": -12.5})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=82.5)
        assert abs(heading - 70.0) < 0.001   # 82.5 + (-12.5)
        assert src == "exif_yaw_corrected"

    def test_exif_yaw_gets_all_three_corrections(self):
        cfg = _cfg({
            "imu_mount_offset_deg": 180.0,
            "imu_magnetic_declination_deg": -12.5,
            "imu_heading_correction_deg": 2.5,
        })
        # raw yaw = 250, expected true heading = (250 + 180 - 12.5 + 2.5) % 360 = 60.0
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=250.0)
        assert abs(heading - 60.0) < 0.001
        assert src == "exif_yaw_corrected"

    def test_exif_yaw_wraps_correctly(self):
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=300.0)
        assert abs(heading - 120.0) < 0.001   # (300 + 180) % 360

    def test_gps_track_used_when_no_exif_yaw(self):
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=None,
                                           exif_gps_track=75.0)
        assert heading == 75.0
        assert src == "exif_gps_track"

    def test_default_zero_when_nothing_available(self):
        cfg = _cfg({})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=None)
        assert heading == 0.0
        assert src == "default"

    def test_no_corrections_when_no_imu_fields(self):
        # Unit with no IMU fields — offset defaults to 0, exif_yaw passes through unchanged
        cfg = _cfg({})
        heading, src = cfg.resolve_heading(cli_override=None, exif_yaw=72.3)
        assert abs(heading - 72.3) < 0.001
        assert src == "exif_yaw_corrected"
