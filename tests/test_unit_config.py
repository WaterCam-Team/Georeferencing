"""Tests for UnitConfig.resolve_heading and resolve_pitch_roll IMU correction logic."""
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


class TestResolvePitchRoll:
    def test_zero_offset_negates_pitch_only(self):
        # Raw EXIF pitch is negated to correct for the BNO055's native sign
        # convention being opposite of camera_geometry.py's (confirmed via
        # RTK-validated GCP refinement, 2026-07-15); roll's sign is untouched.
        cfg = _cfg({})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, 5.0, 3.0)
        assert abs(pitch - (-5.0)) < 0.001
        assert abs(roll  - 3.0) < 0.001
        assert src == "exif_corrected"

    def test_180_degree_pitch_unchanged_roll_negated(self):
        # At 180°, the base pitch-sign correction and the mount-offset
        # rotation's negation cancel out, so pitch ends up equal to the raw
        # EXIF value; roll still flips as before.
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, 5.0, 3.0)
        assert abs(pitch - 5.0) < 0.001
        assert abs(roll  - (-3.0)) < 0.001
        assert src == "exif_corrected"

    def test_90_degree_swaps_with_sign(self):
        # θ=90°: pitch_out = -roll_in (unaffected by the base pitch-sign fix
        # since it only depends on roll here), roll_out = -pitch_in
        cfg = _cfg({"imu_mount_offset_deg": 90.0})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, 5.0, 3.0)
        assert abs(pitch - (-3.0)) < 0.001
        assert abs(roll  - (-5.0)) < 0.001
        assert src == "exif_corrected"

    def test_270_degree_swaps_with_opposite_sign(self):
        # θ=270°: pitch_out = roll_in (unaffected), roll_out = pitch_in
        cfg = _cfg({"imu_mount_offset_deg": 270.0})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, 5.0, 3.0)
        assert abs(pitch -   3.0)  < 0.001
        assert abs(roll  -   5.0)  < 0.001
        assert src == "exif_corrected"

    def test_cli_wins_no_rotation_applied(self):
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        pitch, roll, src = cfg.resolve_pitch_roll(10.0, 2.0, 5.0, 3.0)
        assert pitch == 10.0
        assert roll  ==  2.0
        assert src == "cli"

    def test_partial_cli_override(self):
        # Only pitch CLI set — roll defaults to 0, no rotation
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        pitch, roll, src = cfg.resolve_pitch_roll(10.0, None, 5.0, 3.0)
        assert pitch == 10.0
        assert roll  ==  0.0
        assert src == "cli"

    def test_unit_config_wins_no_rotation_applied(self):
        cfg = _cfg({"pitch_deg": 7.0, "roll_deg": 1.0, "imu_mount_offset_deg": 180.0})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, 5.0, 3.0)
        assert pitch == 7.0
        assert roll  == 1.0
        assert src == "unit_config"

    def test_partial_exif_no_rotation(self):
        # Only pitch in EXIF — cannot apply mount-offset rotation, but the
        # base pitch-sign correction still applies.
        cfg = _cfg({"imu_mount_offset_deg": 180.0})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, 5.0, None)
        assert abs(pitch - (-5.0)) < 0.001
        assert roll == 0.0
        assert src == "exif"

    def test_default_when_nothing_available(self):
        cfg = _cfg({})
        pitch, roll, src = cfg.resolve_pitch_roll(None, None, None, None)
        assert pitch == 0.0
        assert roll  == 0.0
        assert src == "default"


class TestResolvePosition:
    # Synthetic coordinates only: real surveyed positions are site data.
    SURVEYED = {"camera_lat": 40.0, "camera_lon": -74.0}

    def test_cli_wins_over_everything(self):
        cfg = _cfg(self.SURVEYED)
        assert cfg.resolve_position(40.5, -74.5, 40.1, -74.1) == (40.5, -74.5, "cli")

    def test_surveyed_position_wins_over_exif(self):
        cfg = _cfg(self.SURVEYED)
        assert cfg.resolve_position(None, None, 40.1, -74.1) == (40.0, -74.0, "unit_config")

    def test_falls_back_to_exif(self):
        assert _cfg({}).resolve_position(None, None, 40.1, -74.1) == (40.1, -74.1, "exif")

    def test_half_a_cli_position_is_ignored(self):
        cfg = _cfg(self.SURVEYED)
        assert cfg.resolve_position(40.5, None, 40.1, -74.1) == (40.0, -74.0, "unit_config")

    def test_nothing_known(self):
        assert _cfg({}).resolve_position(None, None) == (None, None, "none")

    def test_position_fields_are_known_and_not_echoed(self, capsys):
        cfg = _cfg({**self.SURVEYED, "camera_alt_ellipsoid_m": 87.0})
        assert "Unknown fields" not in capsys.readouterr().out
        assert cfg.camera_alt_ellipsoid_m == 87.0
        assert "40.0" not in cfg.summary() and "position=surveyed" in cfg.summary()
