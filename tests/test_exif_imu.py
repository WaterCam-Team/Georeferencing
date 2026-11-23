"""Tests for UserComment / XMP parsing of the IMU pose fields."""
from exif_imu import parse_user_comment, parse_xmp_pose


def test_old_format_still_parses():
    d = parse_user_comment("Roll 1.5 Pitch -8.0 Yaw 120.0")
    assert (d["roll_deg"], d["pitch_deg"], d["yaw_deg"]) == (1.5, -8.0, 120.0)
    assert d["calib_mag"] is None and d["quaternion"] is None
    assert d["gps_fix_source"] is None


def test_new_words_parsed():
    txt = ("Roll 1.5 Pitch -8.0 Yaw 120.0 CalibMag 3 "
           "Quaternion 0.5,-0.5,0.5,0.5 Gravity 0.1,-9.7,0.2 "
           "PoseTime 2026-01-02T03:04:05Z GPSFix CACHED AgeS 42.5")
    d = parse_user_comment(txt)
    assert d["yaw_deg"] == 120.0
    assert d["calib_mag"] == 3
    assert d["quaternion"] == (0.5, -0.5, 0.5, 0.5)
    assert d["gravity"] == (0.1, -9.7, 0.2)
    assert d["pose_time"] == "2026-01-02T03:04:05Z"
    assert d["gps_fix_source"] == "CACHED"
    assert d["gps_fix_age_s"] == 42.5


def test_partial_and_garbage_tolerated():
    d = parse_user_comment("Roll 0 Pitch 0 Yaw 10 CalibMag 1 GPSFix LIVE")
    assert d["calib_mag"] == 1
    assert d["gps_fix_source"] == "LIVE" and d["gps_fix_age_s"] is None
    assert parse_user_comment("")["yaw_deg"] is None
    assert parse_user_comment("Quaternion nonsense")["quaternion"] is None


def test_xmp_attribute_and_element_forms():
    xmp = (b'<x:xmpmeta><rdf:Description wc:QuaternionW="0.7" wc:GravityZ="9.8" '
           b'wc:GPSFixSource="CACHED"><wc:GPSFixAgeSeconds>12</wc:GPSFixAgeSeconds>'
           b'<wc:PoseReadTime>2026-01-02T03:04:05Z</wc:PoseReadTime>'
           b'</rdf:Description></x:xmpmeta>')
    d = parse_xmp_pose(xmp)
    assert d["QuaternionW"] == 0.7 and d["GravityZ"] == 9.8
    assert d["GPSFixAgeSeconds"] == 12.0
    assert d["GPSFixSource"] == "CACHED"
    assert d["PoseReadTime"] == "2026-01-02T03:04:05Z"
    assert "QuaternionX" not in d


def test_heading_trust_fields_from_xmp():
    # SU-WaterCam #329 writes these with the other XMP pose fields
    xmp = (b'<x:xmpmeta><rdf:Description dc:MagOffsetsLoaded="true" dc:FusionAgeS="95" '
           b'dc:HeadingTrusted="true"/></x:xmpmeta>')
    d = parse_xmp_pose(xmp)
    assert d["MagOffsetsLoaded"] == "true" and d["FusionAgeS"] == 95.0 and d["HeadingTrusted"] == "true"


def test_read_sets_heading_settled_from_the_photo(tmp_path):
    import pytest
    pytest.importorskip("PIL")
    from PIL import Image
    from exif_imu import read_gps_imu_from_exif

    def photo(age):
        path = tmp_path / f"p{age}.jpg"
        Image.new("RGB", (4, 4)).save(path, "JPEG")
        xmp = (b'<?xpacket begin=""?><x:xmpmeta><rdf:Description dc:MagOffsetsLoaded="true" '
               b'dc:FusionAgeS="%d"/></x:xmpmeta><?xpacket end="w"?>' % age)
        data = path.read_bytes()
        path.write_bytes(data[:2] + xmp + data[2:])      # the reader only looks for the packet
        return path
    assert read_gps_imu_from_exif(photo(95))["heading_settled"] is True
    assert read_gps_imu_from_exif(photo(20))["heading_settled"] is False
