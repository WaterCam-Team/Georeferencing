"""
BNO055 Calibration & Orientation Logger
=========================================
Guides calibration of the BNO055 IMU, saves calibration offsets to disk,
reloads them on subsequent runs, and logs orientation data alongside image
capture timestamps for use by the georeferencing pipeline.

HARDWARE SETUP:
    BNO055 → Raspberry Pi via I2C
        VIN  → 3.3V  (Pin 1)
        GND  → GND   (Pin 6)
        SDA  → GPIO2 (Pin 3)
        SCL  → GPIO3 (Pin 5)
        PS1  → GND   (selects I2C mode; leave PS0 floating)

    Default I2C address: 0x28 (PS1=low, PS0=low)
    Alternate address:   0x29 (PS1=low, PS0=high → tie PS0 to 3.3V)

    Enable I2C on the Pi:
        sudo raspi-config → Interface Options → I2C → Enable
        sudo apt install i2c-tools python3-smbus
        i2cdetect -y 1     (should show 0x28 or 0x29)

IMPORTANT — MOUNTING NOTE:
    The BNO055 must be mounted rigidly to the camera box, not the pole.
    Any flex between the sensor and cameras introduces orientation error.
    Once mounted, do not move or re-solder the sensor — the hard-iron
    calibration (magnetometer offsets) is specific to that physical location
    because nearby ferromagnetic materials in the camera box shift the
    local magnetic field. If you remount the sensor, recalibrate from scratch.

DEPENDENCIES:
    pip install smbus2
"""

import smbus2
import struct
import time
import json
import csv
import os
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BNO055 REGISTER MAP
# ─────────────────────────────────────────────────────────────────────────────

# I2C
BNO055_ADDRESS      = 0x28   # change to 0x29 if PS0 tied to 3.3V
I2C_BUS             = 1      # /dev/i2c-1 on all 40-pin Pi models

# Page 0 registers
REG_CHIP_ID         = 0x00   # should read 0xA0 — confirms comms working
REG_OPR_MODE        = 0x3D   # operation mode
REG_PWR_MODE        = 0x3E
REG_SYS_TRIGGER     = 0x3F
REG_UNIT_SEL        = 0x3B   # units: degrees vs radians, m/s² vs mg, etc.
REG_CALIB_STAT      = 0x35   # calibration status (read-only)
REG_SYS_STATUS      = 0x39
REG_SYS_ERR         = 0x3A
REG_TEMP            = 0x34   # on-chip temperature (°C)

# Euler angle output (heading, roll, pitch) — 3× 2-byte signed int16
REG_EUL_HEADING_LSB = 0x1A
REG_EUL_ROLL_LSB    = 0x1C
REG_EUL_PITCH_LSB   = 0x1E

# Quaternion output — 4× 2-byte signed int16 (w, x, y, z)
REG_QUA_W_LSB       = 0x20

# Calibration offset storage — 22 bytes at 0x55–0x6A
# Accel offset XYZ (6 bytes) + Mag offset XYZ (6 bytes) +
# Gyro offset XYZ (6 bytes) + Accel radius (2 bytes) + Mag radius (2 bytes)
REG_CALIB_OFFSET_START = 0x55
CALIB_OFFSET_LEN       = 22

# Operation modes
MODE_CONFIG         = 0x00   # required to write config registers
MODE_NDOF           = 0x0C   # 9-DOF fusion: accel + gyro + mag (recommended)
MODE_IMUPLUS        = 0x08   # 6-DOF: accel + gyro only (no magnetometer)
                             # Use IMUPLUS if near strong magnetic interference
                             # (motors, large metal structures). Heading will
                             # drift over time but pitch/roll are stable.
MODE_AMG            = 0x07   # raw sensor data only (no fusion)

# Unit selection (REG_UNIT_SEL)
UNIT_DEG            = 0x00   # Euler angles in degrees (recommended)
UNIT_RAD            = 0x04

# Euler angle scale
DEG_SCALE           = 16.0   # 1 degree = 16 LSB


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LOW-LEVEL I2C INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class BNO055:
    """
    Minimal driver for the BNO055 via smbus2.
    Avoids the Adafruit CircuitPython/Blinka dependency chain, which can
    be unreliable on non-standard Python installations.
    """

    def __init__(self, bus_num: int = I2C_BUS, address: int = BNO055_ADDRESS):
        self.bus     = smbus2.SMBus(bus_num)
        self.address = address
        self._verify_chip()
        self._configure()

    def _read_byte(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def _write_byte(self, reg: int, value: int):
        self.bus.write_byte_data(self.address, reg, value)
        time.sleep(0.01)   # BNO055 needs ~7 ms after most register writes

    def _read_bytes(self, reg: int, length: int) -> bytes:
        return bytes(self.bus.read_i2c_block_data(self.address, reg, length))

    def _set_mode(self, mode: int):
        """Switch operation mode. Must go through CONFIG mode first."""
        self._write_byte(REG_OPR_MODE, MODE_CONFIG)
        time.sleep(0.025)   # 19 ms transition time per datasheet
        self._write_byte(REG_OPR_MODE, mode)
        time.sleep(0.012)   # 7 ms transition to fusion modes

    def _verify_chip(self):
        chip_id = self._read_byte(REG_CHIP_ID)
        if chip_id != 0xA0:
            raise IOError(
                f"BNO055 not found at 0x{self.address:02X} on bus {I2C_BUS}. "
                f"Got chip ID 0x{chip_id:02X} (expected 0xA0). "
                f"Check wiring, I2C address (PS0/PS1 pins), and that I2C "
                f"is enabled: sudo raspi-config → Interface Options → I2C"
            )
        print(f"[BNO055] Chip verified at 0x{self.address:02X}")

    def _configure(self):
        """Initial configuration: reset, set units, enter NDOF fusion mode."""
        # Reset
        self._write_byte(REG_SYS_TRIGGER, 0x20)
        time.sleep(0.65)   # 650 ms boot time after reset

        # Use degrees, m/s², Celsius
        self._set_mode(MODE_CONFIG)
        self._write_byte(REG_UNIT_SEL, UNIT_DEG)

        # Enter full 9-DOF fusion mode
        self._set_mode(MODE_NDOF)
        print("[BNO055] Initialised in NDOF mode")

    # ── Calibration status ────────────────────────────────────────────────────

    def calibration_status(self) -> dict:
        """
        Read calibration status for each sub-system.
        Returns values 0–3 for: system, gyro, accelerometer, magnetometer.
        3 = fully calibrated, 0 = uncalibrated.

        The system status reflects the lowest of the three sensor statuses.
        For reliable heading output you need mag=3.
        For reliable tilt/roll you need accel=3.
        Gyro reaches 3 quickly (a few seconds of stillness).
        """
        stat = self._read_byte(REG_CALIB_STAT)
        return {
            "system": (stat >> 6) & 0x03,
            "gyro":   (stat >> 4) & 0x03,
            "accel":  (stat >> 2) & 0x03,
            "mag":    (stat >> 0) & 0x03,
        }

    def is_fully_calibrated(self) -> bool:
        s = self.calibration_status()
        return s["gyro"] == 3 and s["accel"] == 3 and s["mag"] == 3

    # ── Euler angles ──────────────────────────────────────────────────────────

    def euler_angles(self) -> dict:
        """
        Read heading, roll, and pitch in degrees.

        BNO055 Euler angle convention:
            heading : 0–360°, 0 = magnetic North, increases clockwise
            roll    : –180 to +180°, positive = right side down
            pitch   : –180 to +180°, positive = nose up

        NOTE on heading vs yaw:
            The BNO055 'heading' is a compass bearing (magnetic North reference).
            In the georeferencing script, HEADING_DEG expects a compass bearing
            in the same convention, so use heading directly.
            Magnetic declination for your site should be added to convert to
            true North if needed. Look up your declination at:
            https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
        """
        data    = self._read_bytes(REG_EUL_HEADING_LSB, 6)
        heading = struct.unpack_from('<h', data, 0)[0] / DEG_SCALE
        roll    = struct.unpack_from('<h', data, 2)[0] / DEG_SCALE
        pitch   = struct.unpack_from('<h', data, 4)[0] / DEG_SCALE
        return {"heading": heading, "roll": roll, "pitch": pitch}

    def quaternion(self) -> dict:
        """
        Read orientation as a quaternion (w, x, y, z).
        More numerically stable than Euler angles for interpolation.
        Useful if you want to log raw orientation and convert later.
        """
        data  = self._read_bytes(REG_QUA_W_LSB, 8)
        scale = 1.0 / (1 << 14)   # 1/16384 per datasheet
        w, x, y, z = [struct.unpack_from('<h', data, i*2)[0] * scale
                       for i in range(4)]
        return {"w": w, "x": x, "y": y, "z": z}

    def temperature(self) -> float:
        return float(self._read_byte(REG_TEMP))

    # ── Calibration offsets ───────────────────────────────────────────────────

    def read_offsets(self) -> bytes:
        """
        Read the 22-byte calibration offset block from the sensor.
        Must be in CONFIG mode to read these registers accurately.
        """
        self._set_mode(MODE_CONFIG)
        offsets = self._read_bytes(REG_CALIB_OFFSET_START, CALIB_OFFSET_LEN)
        self._set_mode(MODE_NDOF)
        return offsets

    def write_offsets(self, offsets: bytes):
        """
        Write previously saved calibration offsets back to the sensor.
        This restores calibration across power cycles without re-running
        the calibration procedure. Must be in CONFIG mode.
        """
        if len(offsets) != CALIB_OFFSET_LEN:
            raise ValueError(f"Expected {CALIB_OFFSET_LEN} bytes, got {len(offsets)}")
        self._set_mode(MODE_CONFIG)
        for i, byte in enumerate(offsets):
            self._write_byte(REG_CALIB_OFFSET_START + i, byte)
        self._set_mode(MODE_NDOF)
        time.sleep(0.1)

    def close(self):
        self.bus.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SAVE / LOAD CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

CALIB_FILE = "./bno055_calibration.json"

def save_calibration(imu: BNO055, path: str = CALIB_FILE,
                     node_id: str = "node_1",
                     magnetic_declination_deg: float = 0.0):
    """
    Save calibration offsets and metadata to a JSON file.

    PARAMETERS:
        node_id : identifier for this specific camera node.
                  Calibration is node-specific because the hard-iron offsets
                  depend on the local magnetic environment of the camera box.
        magnetic_declination_deg : local magnetic declination in degrees.
                  Positive = East (magnetic North is East of true North).
                  Look this up for your deployment site at:
                  https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
                  Syracuse NY is approximately -12.5° (as of 2025).
                  Add this value to BNO055 heading readings to get true North.
    """
    offsets = imu.read_offsets()
    status  = imu.calibration_status()

    data = {
        "node_id":             node_id,
        "timestamp":           datetime.now().isoformat(),
        "calibration_status":  status,
        "magnetic_declination_deg": magnetic_declination_deg,
        "offsets_hex":         offsets.hex(),
        "offsets_bytes":       list(offsets),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[CALIB] Saved to {path}")
    print(f"  Status: sys={status['system']} gyro={status['gyro']} "
          f"accel={status['accel']} mag={status['mag']}")


def load_calibration(imu: BNO055, path: str = CALIB_FILE) -> float:
    """
    Load saved calibration offsets into the sensor.

    Returns magnetic declination in degrees (for adding to heading readings).

    WHEN TO RELOAD vs RECALIBRATE:
        Reload (this function):
            - Normal power-on of a deployed node
            - Same physical mounting, same location
        Recalibrate from scratch:
            - Sensor has been physically moved or remounted
            - Camera box has been modified (new hardware added nearby)
            - Heading readings seem consistently wrong by a fixed offset
            - More than ~6 months have passed (Earth's field drifts slowly)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No calibration file found at {path}. "
            f"Run calibration procedure first."
        )
    with open(path) as f:
        data = json.load(f)

    offsets = bytes(data["offsets_bytes"])
    imu.write_offsets(offsets)

    age = (datetime.now() -
           datetime.fromisoformat(data["timestamp"])).days
    decl = data.get("magnetic_declination_deg", 0.0)

    print(f"[CALIB] Loaded from {path}")
    print(f"  Calibrated: {data['timestamp']} ({age} days ago)")
    print(f"  Node: {data['node_id']}")
    print(f"  Magnetic declination: {decl}°")
    if age > 180:
        print(f"  WARNING: Calibration is {age} days old. "
              f"Consider recalibrating.")

    return decl


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GUIDED CALIBRATION PROCEDURE
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_procedure(imu: BNO055,
                               save_path: str = CALIB_FILE,
                               node_id: str = "node_1",
                               magnetic_declination_deg: float = 0.0):
    """
    Interactive calibration procedure.

    The BNO055 calibrates its three sub-systems through specific motions.
    This function monitors calibration status in real time and prompts
    you through each stage.

    BEFORE STARTING:
        - Mount the BNO055 rigidly in the camera box in its final position.
        - Move well away from large metal objects, motors, and power cables.
        - The calibration captures hard-iron offsets specific to this mounting.

    GYROSCOPE (fastest, ~5 seconds):
        Place the camera box on a stable surface and leave it completely still.
        The gyro calibrates automatically during stillness.

    ACCELEROMETER (moderate, ~1–2 minutes):
        Place the camera box in 6 different stable orientations:
        flat, upside-down, on each of its four sides.
        Hold each position for 3–5 seconds until you feel the accel status
        increment. You do not need to be precise — any 6 distinct stable
        orientations covering different gravity vector directions work.

    MAGNETOMETER (most important for heading, ~1–2 minutes):
        Hold the camera box and rotate it slowly through a figure-eight
        pattern in the air. This motion samples the local magnetic field
        from many directions, allowing the sensor to compute hard-iron
        and soft-iron correction offsets.

        CRITICAL: Do this at the deployment site (or a magnetically similar
        environment), not at a workbench near computers or power supplies.
        Large ferromagnetic objects within ~30 cm will corrupt the offsets.
        Repeat after any significant modification to the camera box hardware.
    """
    print("\n" + "="*60)
    print("  BNO055 CALIBRATION PROCEDURE")
    print("="*60)
    print("\nThis will guide you through calibrating all three sub-sensors.")
    print("Status: 0 = uncalibrated → 3 = fully calibrated\n")

    stages = {
        "gyro":  {"done": False,
                  "instruction": "Set the camera box on a stable surface "
                                 "and leave it completely still."},
        "accel": {"done": False,
                  "instruction": "Place the camera box in 6 different stable "
                                 "orientations (flat, upside-down, each side). "
                                 "Hold each for ~3 seconds."},
        "mag":   {"done": False,
                  "instruction": "Hold the camera box and rotate it slowly "
                                 "through a figure-eight pattern. Do this at "
                                 "the deployment site, away from metal objects."},
    }

    # Show instructions for first stage
    print(f"STEP 1 — GYROSCOPE\n  {stages['gyro']['instruction']}\n")

    try:
        while True:
            status = imu.calibration_status()
            angles = imu.euler_angles()

            bar = lambda v: "[" + "#"*v + "."*(3-v) + "]"
            print(f"\r  sys={bar(status['system'])}  "
                  f"gyro={bar(status['gyro'])}  "
                  f"accel={bar(status['accel'])}  "
                  f"mag={bar(status['mag'])}    "
                  f"hdg={angles['heading']:6.1f}°  "
                  f"pitch={angles['pitch']:6.1f}°  "
                  f"roll={angles['roll']:6.1f}°  ",
                  end="", flush=True)

            # Prompt for next stage when current one completes
            if not stages["gyro"]["done"] and status["gyro"] == 3:
                stages["gyro"]["done"] = True
                print(f"\n\n✓ Gyroscope calibrated.")
                print(f"\nSTEP 2 — ACCELEROMETER\n  "
                      f"{stages['accel']['instruction']}\n")

            if not stages["accel"]["done"] and status["accel"] == 3:
                stages["accel"]["done"] = True
                print(f"\n\n✓ Accelerometer calibrated.")
                print(f"\nSTEP 3 — MAGNETOMETER\n  "
                      f"{stages['mag']['instruction']}\n")

            if not stages["mag"]["done"] and status["mag"] == 3:
                stages["mag"]["done"] = True
                print(f"\n\n✓ Magnetometer calibrated.")

            if imu.is_fully_calibrated():
                print(f"\n\n✓ All sub-systems fully calibrated.")
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        s = imu.calibration_status()
        print(f"\n\nCalibration interrupted. "
              f"Current status: sys={s['system']} gyro={s['gyro']} "
              f"accel={s['accel']} mag={s['mag']}")
        if s["gyro"] < 3 or s["accel"] < 3 or s["mag"] < 3:
            print("WARNING: Saving partial calibration. "
                  "Heading accuracy may be reduced.")

    save_calibration(imu, save_path, node_id, magnetic_declination_deg)
    return imu.calibration_status()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: VALIDATE HEADING AGAINST KNOWN BEARING
# ─────────────────────────────────────────────────────────────────────────────

def validate_heading(imu: BNO055,
                     known_bearing_deg: float,
                     magnetic_declination_deg: float = 0.0,
                     n_samples: int = 50):
    """
    Compare the IMU heading against a known true bearing.

    PROCEDURE:
        1. Survey two points in the camera's field of view with the viDoc RTK
           (e.g. two painted GCPs, or a known landmark and the node itself).
        2. Compute the true bearing between them using their RTK coordinates.
        3. Point the camera directly at the far point and run this function.

    This tells you the combined heading error from:
        - Magnetic declination (systematic, fixed for a location)
        - Hard-iron offsets not fully corrected by calibration
        - Soft-iron distortion from the camera box structure

    The residual offset printed here can be added to all subsequent heading
    readings as a fine correction beyond what the BNO055 calibration alone
    provides.

    PARAMETERS:
        known_bearing_deg        : true bearing from camera to target (degrees)
        magnetic_declination_deg : local magnetic declination (from calibration file)
    """
    readings = []
    print(f"[VALIDATE] Collecting {n_samples} heading samples...")
    for _ in range(n_samples):
        h = imu.euler_angles()["heading"]
        # Apply magnetic declination to convert magnetic heading to true heading
        true_heading = (h + magnetic_declination_deg) % 360.0
        readings.append(true_heading)
        time.sleep(0.05)

    mean_h = sum(readings) / len(readings)
    std_h  = (sum((r - mean_h)**2 for r in readings) / len(readings)) ** 0.5
    error  = mean_h - known_bearing_deg

    # Wrap error to [-180, 180]
    error = (error + 180) % 360 - 180

    print(f"\n[VALIDATE] Results:")
    print(f"  Known true bearing  : {known_bearing_deg:.2f}°")
    print(f"  IMU mean heading    : {mean_h:.2f}° (after declination correction)")
    print(f"  Residual error      : {error:+.2f}°")
    print(f"  Heading std dev     : {std_h:.3f}° (noise level)")
    print()

    if abs(error) > 5.0:
        print("  WARNING: Residual error > 5°. Possible causes:")
        print("    - Magnetometer calibration incomplete (mag status < 3)")
        print("    - Strong ferromagnetic material near the sensor")
        print("    - Incorrect magnetic declination value")
        print("    - Sensor not pointing at the reference target")
    elif abs(error) > 2.0:
        print("  CAUTION: Residual error 2–5°. Acceptable for flood mapping")
        print("    but consider adding this as a fixed correction offset.")
    else:
        print("  GOOD: Residual error < 2°. Heading is well calibrated.")

    print(f"\n  To correct for this, add {-error:+.2f}° to all heading readings,")
    print(f"  or set HEADING_CORRECTION_DEG = {-error:.2f} in the logger.")
    return error


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ORIENTATION LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def run_logger(imu: BNO055,
               log_path: str = "./imu_log.csv",
               interval_s: float = 1.0,
               magnetic_declination_deg: float = 0.0,
               heading_correction_deg: float = 0.0):
    """
    Log orientation data continuously to a CSV file.

    The CSV format matches what load_imu_orientation() in the georeferencing
    script expects:
        timestamp, tilt_deg, roll_deg, yaw_deg, temp_c, calib_mag, calib_sys

    PARAMETERS:
        interval_s              : seconds between log entries
        magnetic_declination_deg: added to heading to get true North bearing
        heading_correction_deg  : residual correction from validate_heading()

    INTEGRATION WITH IMAGE CAPTURE:
        When the camera captures an image, record the timestamp in ISO format.
        Pass that timestamp to load_imu_orientation() and it will return the
        closest IMU reading, giving you the orientation at time of capture.

    CALIBRATION WATCH:
        If mag calibration drops below 2 during logging, the heading data
        becomes unreliable — this is flagged in the log and printed to console.
        This can happen if the camera box is moved near a large metal object.
        Re-run the magnetometer figure-eight if this occurs.
    """
    fieldnames = ["timestamp", "tilt_deg", "roll_deg", "yaw_deg",
                  "temp_c", "calib_sys", "calib_gyro",
                  "calib_accel", "calib_mag"]

    file_exists = os.path.exists(log_path)
    print(f"[LOGGER] Logging to {log_path}  (Ctrl+C to stop)")

    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            while True:
                ts      = datetime.now().isoformat()
                angles  = imu.euler_angles()
                status  = imu.calibration_status()
                temp    = imu.temperature()

                # Apply declination and fine correction to get true heading
                true_heading = ((angles["heading"]
                                 + magnetic_declination_deg
                                 + heading_correction_deg) % 360.0)

                row = {
                    "timestamp":  ts,
                    "tilt_deg":   round(angles["pitch"],   3),
                    "roll_deg":   round(angles["roll"],    3),
                    "yaw_deg":    round(true_heading,      3),
                    "temp_c":     round(temp,              1),
                    "calib_sys":  status["system"],
                    "calib_gyro": status["gyro"],
                    "calib_accel":status["accel"],
                    "calib_mag":  status["mag"],
                }
                writer.writerow(row)
                f.flush()

                # Console status
                mag_warn = " ⚠ MAG UNCAL" if status["mag"] < 2 else ""
                print(f"\r  {ts}  "
                      f"hdg={true_heading:6.1f}°  "
                      f"pitch={angles['pitch']:6.1f}°  "
                      f"roll={angles['roll']:6.1f}°  "
                      f"T={temp}°C  "
                      f"cal={status['system']}/{status['gyro']}/"
                      f"{status['accel']}/{status['mag']}"
                      f"{mag_warn}  ",
                      end="", flush=True)

                time.sleep(interval_s)

    except KeyboardInterrupt:
        print(f"\n[LOGGER] Stopped. Log saved to {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIGURATION ─────────────────────────────────────────────────────────
    NODE_ID    = "node_1"
    CALIB_PATH = f"./{NODE_ID}_bno055_calibration.json"
    LOG_PATH   = f"./{NODE_ID}_imu_log.csv"

    # Magnetic declination for your site (degrees).
    # Syracuse NY ≈ -12.5° (magnetic North is 12.5° West of true North).
    # Look up your exact value: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
    MAGNETIC_DECLINATION = -12.5

    # Residual correction from validate_heading() — set to 0.0 initially,
    # then update after running the validation procedure.
    HEADING_CORRECTION = 0.0

    # ── CHOOSE MODE ───────────────────────────────────────────────────────────
    MODE = "log"
    # Options:
    #   "calibrate" — run full calibration procedure and save offsets
    #   "validate"  — check heading accuracy against a known bearing
    #   "log"       — load saved calibration and start logging

    # ── RUN ───────────────────────────────────────────────────────────────────
    imu = BNO055()

    if MODE == "calibrate":
        run_calibration_procedure(
            imu, CALIB_PATH, NODE_ID, MAGNETIC_DECLINATION
        )

    elif MODE == "validate":
        decl = load_calibration(imu, CALIB_PATH)
        # Set this to the RTK-derived true bearing from camera to a known point
        KNOWN_BEARING = 47.3   # example — replace with your surveyed value
        validate_heading(imu, KNOWN_BEARING, decl)

    elif MODE == "log":
        decl = load_calibration(imu, CALIB_PATH)
        run_logger(imu, LOG_PATH,
                   interval_s=1.0,
                   magnetic_declination_deg=decl,
                   heading_correction_deg=HEADING_CORRECTION)

    imu.close()
