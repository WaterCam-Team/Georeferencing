# Step-by-Step Procedure: Camera Calibration, IMU Calibration, GCP Collection, and Photo Georeferencing

This document describes the procedure a researcher should follow to **calibrate the camera**, **calibrate the BNO055 or BNO08x IMU**, **collect ground control points (GCPs)**, and **accurately georeference photos** of an area. It is written for fixed-position cameras (e.g. flood or monitoring cameras) and for mobile capture with GPS + IMU.

---

## Table of contents

1. [Prerequisites and conventions](#1-prerequisites-and-conventions)
2. [Camera calibration (checkerboard)](#2-camera-calibration-checkerboard)
3. [IMU calibration (BNO055 / BNO08x)](#3-imu-calibration-bno055--bno08x)
4. [Collecting ground control points (GCPs)](#4-collecting-ground-control-points-gcps)
5. [Photo collection for georeferencing](#5-photo-collection-for-georeferencing)
6. [Georeferencing workflow](#6-georeferencing-workflow)
7. [Vertical datum and accuracy](#7-vertical-datum-and-accuracy)
8. [Quality checks and troubleshooting](#8-quality-checks-and-troubleshooting)

---

## 1. Prerequisites and conventions

### 1.1 Software and dependencies

- **Python 3** with: `opencv-python`, `numpy`, `pyproj`, `Pillow`, `piexif`
- Optional: `rasterio` (GeoTIFF DEM), `laspy` (LAS/LAZ), `gdal` (GeoTIFF export)
- For on-device capture (e.g. Raspberry Pi): `gpsd-py3`, `adafruit-circuitpython-bno055` (or equivalent for BNO08x), `libcamera-apps`

### 1.2 Coordinate and angle conventions (used in this project)

- **World frame (ENU):** X = East, Y = North, Z = Up  
- **Camera frame:** X = right, Y = down, Z = forward (into the scene)  
- **Heading (yaw):** 0° = North, 90° = East (compass bearing of camera boresight)  
- **Pitch:** 0° = level (horizon), **negative** = looking down (e.g. −75° for a downward flood camera)  
- **Roll:** 0° = level; positive = right side of camera tilts down  

All georeferencing scripts in this repository use these conventions consistently (see `camera_geometry.py`).

### 1.3 Two ways to provide orientation to the tools

- **IMU log (CSV):** Timestamped rows with `tilt_deg`, `roll_deg`, `yaw_deg` (and optionally `temp_c`). The main georeferencing tool (`georeference_tool.py`) can load this and match by image timestamp.  
- **EXIF UserComment:** Plain text in the image EXIF: `Roll R Pitch P Yaw Y` (degrees), e.g. `Roll 1.5 Pitch -8.0 Yaw 120.0`. Scripts such as `add_imu.py`, `georeference3d.py`, and `flood_map.py` read this format.

---

## 2. Camera calibration (checkerboard)

Camera calibration estimates the **intrinsic** parameters (focal length, principal point, distortion) so that pixel coordinates can be converted to 3D rays. Good calibration is essential for accurate georeferencing.

### 2.1 Checkerboard and setup

- **Print a checkerboard** (search for “OpenCV checkerboard PDF”). Use a rigid, flat mount; avoid warping or reflections.  
- **Measure the square size** accurately (e.g. with a ruler) and note it in **meters** (e.g. 2.5 cm → 0.025 m).  
- In **`camera_calibration.py`**, set at the top of the file:
  - `BOARD_W`, `BOARD_H` = number of **interior corners** (e.g. 9×6 squares → 8×5 interior corners).  
  - `SQUARE_SIZE_M` = physical size of one square in meters.

### 2.2 Capturing calibration images

**Option A — Using this repo (desktop):**

1. Place the checkerboard in front of the camera.  
2. Run the calibration script; it will prompt for:
   - Folder containing checkerboard images (default `./calib_images`),  
   - Path to save calibration (default `./calibration.json`),  
   - **Camera height above ground in meters** (optional but recommended; stored in `calibration.json` for georeferencing).  
3. Either:
   - **Pre-capture:** Take 15–25 photos yourself and put them in the folder. **Vary pose:** tilt the board (not always level), use different distances (near and far), and move the board to different positions (center, corners, edges). Level-only or same-distance views give poor calibration.  
   - Or use the optional **live capture** in `camera_calibration.py` (`capture_calibration_images()`) to save frames when you press space.

**Option B — On-device (e.g. Raspberry Pi with BNO055):**

- Use **`calibration-checkerboard.py`** if available: it captures images with libcamera, records GPS (via gpsd) and IMU (BNO055) per frame, and can run OpenCV calibration on the detected corners. Checkerboard dimensions and square size in that script must match your printed board.

### 2.3 Running calibration

1. Put all checkerboard images (`.jpg` or `.png`) in the chosen folder.  
2. Run:
   ```bash
   python camera_calibration.py
   ```
3. Enter the image folder path, output path (`calibration.json`), and camera height when prompted.  
4. Check the **RMS reprojection error** printed at the end:
   - **&lt; 0.5 px** = excellent  
   - **&lt; 1.0 px** = good  
   - **&gt; 1.0 px** = retake with more varied poses or remove outlier images (the script can reject high-error images automatically).  
5. Inspect the **per-image RMS table**; remove or re-shoot any image with unusually high error.  
6. The script saves **`calibration.json`** containing `K`, `D`, `img_size`, `rms`, and optionally `camera_height_m`. Use this file for all subsequent georeferencing with that camera.

### 2.4 Important for accuracy

- **Same resolution:** Use calibration images at the **same resolution** as the photos you will georeference, or the tools will scale the camera matrix (e.g. in `georeference_tool.py`) when the image size differs.  
- **Square size:** An error in `SQUARE_SIZE_M` directly scales your focal length and degrades accuracy.  
- **Pose variety:** More varied board poses (tilt, distance, position) give a more stable and accurate calibration.

---

## 3. IMU calibration (BNO055 / BNO08x)

The IMU provides **heading (yaw), pitch, and roll**. Heading is the main source of large errors if the magnetometer is not calibrated.

### 3.1 Magnetometer calibration (BNO055 / BNO08x)

1. **Environment:** Perform calibration **away from** strong magnetic interference (motors, metal structures, computers, reinforced concrete).  
2. **Figure-8 procedure:**  
   - Hold the sensor **roughly level** (parallel to the ground).  
   - Move it in a **figure-8 pattern** in the air, rotating through all orientations (including different headings).  
   - Continue until the magnetometer calibration status reaches **3** (fully calibrated).  
3. **Status:** Use the sensor’s calibration status API (e.g. BNO055 returns 0–3 for mag calibration). Many Adafruit/CircuitPython examples show how to read this.  
4. **Persistence (if supported):** Some workflows allow saving calibration offsets to non-volatile memory so the sensor starts calibrated after power-up. Use this when available to avoid repeating the figure-8 every time.

### 3.2 Mounting and alignment

- **Fixed camera:** Mount the IMU so its axes align as closely as possible with the camera (or document the fixed offset). The software assumes orientation is that of the camera (heading = direction the camera is facing).  
- **Roll:** For a well-mounted fixed camera, roll should be near 0°. Large roll may indicate misalignment or a need for a roll-offset in software.

### 3.3 Providing orientation to the pipeline

- **If using an IMU log (CSV):** Log columns `timestamp`, `tilt_deg`, `roll_deg`, `yaw_deg` (and optionally `temp_c`) synchronized with image capture. In `georeference_tool.py`, set `IMU_LOG_PATH` and optionally `IMAGE_TIMESTAMP` so the nearest IMU reading is used for each image.  
- **If embedding in EXIF:** Use the same format as `add_imu.py`: write to EXIF UserComment the string `Roll R Pitch P Yaw Y` (degrees). Example: `Roll 1.5 Pitch -8.0 Yaw 120.0`. Then scripts that read EXIF (e.g. `georeference3d.py`, `flood_map.py`) can use orientation per image without a separate log file.

---

## 4. Collecting ground control points (GCPs)

GCPs are **known geographic positions** (and optionally elevation) that you can use to **check or refine** georeferencing accuracy.

### 4.1 What to use as GCPs

- **Surveyed points:** RTK-GPS or total station coordinates (lat, lon, and elevation in a known datum).  
- **Clearly identifiable features:** Corners of buildings, road markings, manhole covers, or other features you can both **see in the image** and **locate on a map or survey**.

### 4.2 How many and where

- Use **at least 3–5 GCPs** per scene; more and well spread (center and corners of the area of interest) improve checks and any future refinement.  
- Spread GCPs across the **image** (not all in the center) to constrain both position and orientation errors.  
- Record for each GCP:
  - **Label** (e.g. “GCP1”, “NW corner building”)  
  - **Latitude, longitude** (decimal degrees, WGS84 unless you document otherwise)  
  - **Elevation** (meters, and note the datum: e.g. orthometric/NAVD88 vs ellipsoidal)  
  - **Where it appears in the photo** (e.g. “center of manhole in lower-left of image”) so you can click the same point in the georeferencing tool.

### 4.3 Using GCPs to validate and improve georeferencing

**In `georeference_tool.py` (interactive):**

1. Run the tool and load an image with camera pose (EXIF or manual).  
2. **Add GCPs:** Left-click a point in the image, then press **G**. Enter the **known** latitude, longitude, and optionally elevation for that point. Repeat for at least 3 GCPs (spread across the image).  
3. **Refine pose:** Press **R** to refine camera position (lat, lon), height, and orientation (heading, pitch, roll) by minimizing the error at GCPs. The tool prints the RMS residual and per-GCP error in metres.  
4. **Optional — TPS warp:** With **6 or more** GCPs, press **W** to fit a thin-plate spline from pixel (u, v) to (lat, lon). When TPS is ON, every click uses this warp instead of the camera model, which can improve local accuracy where GCPs are dense.  
5. **Load/save GCPs:** Press **L** to load GCPs from a CSV file; press **E** to save the current GCPs to CSV (columns: label, pixel_u, pixel_v, lat, lon, elev_m). You can set `GCP_CSV` in the script config to load GCPs automatically on start.

**Validation only (no refinement):** Click each GCP in the image and compare the **predicted** (lat, lon) to known coordinates. Large residuals indicate camera height, heading/pitch bias, or vertical datum issues; use pose refinement (R) or adjust and re-run.

---

## 5. Photo collection for georeferencing

### 5.1 Fixed-position camera (e.g. flood or monitoring camera)

- **GPS position:** Must be known (from EXIF if the device writes it, or from a one-time survey of the mount).  
- **Orientation:** From IMU (calibrated as above) or from manual measurement. For downward-looking cameras, **pitch should be negative** (e.g. −75°).  
- **Height:** Measure **camera height above ground** (or above the vertical datum you use for terrain). This can be stored in `calibration.json` and is used automatically by `georeference_tool.py` when present.  
- **Resolution:** Use the **same resolution** as the calibration images when possible, or rely on the tool’s scaling of the camera matrix to the current image size.

### 5.2 Mobile capture (e.g. walking or vehicle)

- **Per-image GPS + orientation:** Each photo should have **GPS** (lat, lon, alt) and **orientation** (heading, pitch, roll) either in EXIF (e.g. UserComment) or in a synchronized IMU log.  
- **Timing:** Ensure timestamps of images and IMU log align so the correct orientation is used for each frame.

### 5.3 General tips

- Avoid motion blur; ensure good lighting.  
- For flat-ground georeferencing, the model assumes a **horizontal ground plane** at the camera height reference. For sloped or complex terrain, use the **terrain-aware** pipeline (`georeference_terrain.py`) with a DEM or LiDAR-derived surface.

---

## 6. Georeferencing workflow

### 6.1 Flat-ground (single fixed camera)

1. **Inputs:**  
   - Field image(s),  
   - `calibration.json` (from camera calibration),  
   - Camera position (lat, lon) — from EXIF or manual,  
   - Camera orientation (heading, pitch, roll) — from IMU log or EXIF,  
   - Camera height above ground — from calibration file or manual.  

2. **Run:**  
   ```bash
   python georeference_tool.py
   ```  
   Configure at the bottom of the script (or via your own wrapper): `IMAGE_PATH`, `CALIB_PATH`, `IMU_LOG_PATH`, `IMAGE_TIMESTAMP`, `HEADING_DEG`, `PITCH_DEG`, `ROLL_DEG`, `CAMERA_HEIGHT_M`, and output paths.  

3. **Use:**  
   - **Interactive:** Click points in the image to get (lat, lon); right-click to label; save to CSV.  
   - **Batch / GeoTIFF:** Use the script’s batch mode or GeoTIFF export if you need many points or a georeferenced raster.

### 6.2 Terrain-aware (DEM / LiDAR)

When the scene is **not flat**, use **`georeference_terrain.py`** so that rays are intersected with a **terrain surface** (DEM or rasterized LAS/LAZ) instead of a horizontal plane.

1. **Inputs:**  
   - Field image,  
   - Calibration (or nominal intrinsics),  
   - Camera lat, lon, **elevation** (in the **same vertical datum** as the terrain data),  
   - Orientation (heading, pitch, roll),  
   - GeoTIFF DEM and/or LAS/LAZ point cloud.  

2. **Run:**  
   ```bash
   python georeference_terrain.py path/to/photo.jpg --dem path/to/dem.tif --lat 43.04 --lon -76.13 --elev 120 --heading 90 --pitch -15
   ```  
   Camera elevation must be consistent with the DEM (e.g. orthometric if the DEM is orthometric).  

3. **Datum-safe elevation:** To avoid mixing EXIF altitude with DEM datum, use **`--height-above-ground`** instead of `--elev` when you have terrain data. The tool then sets camera elevation = DEM elevation at the camera + height above ground (see `geo_core.camera_elev_from_dem`). Example:  
   ```bash
   python georeference_terrain.py photo.jpg --dem dem.tif --lat 43.04 --lon -76.13 --height-above-ground 4.0 --pitch -15
   ```

4. **Output:** Click-to-GPS with **elevation** from the terrain; optional CSV of points with (lat, lon, elev).

### 6.3 3D / multi-image (if available)

For **multiple photos** with EXIF GPS + IMU, **`georeference3d.py`** can run structure-from-motion (SfM), align the model to GPS, and provide click-to-GPS with optional terrain or flat fallback. Use it when you have many overlapping images and want 3D-consistent coordinates.

---

## 7. Vertical datum and accuracy

- **EXIF altitude** is often **ellipsoidal** (WGS84) and can be noisy. **DEMs** are often **orthometric** (e.g. EGM96, EGM2008, NAVD88). **Do not mix** them without conversion: a 15–35 m vertical error can cause large horizontal errors in ray intersection. The `vertical_datum` module and `georeference_terrain.py` options `--terrain-vertical-datum` / `--camera-elev-datum` allow checking and conversion when PROJ geoid grids are available.  
- **Best practice for terrain workflows:** Compute camera elevation as **ground elevation at the camera (from DEM) + mounting height above ground**, and use that value (in the same datum as the DEM) for ray–terrain intersection. The helper `geo_core.camera_elev_from_dem(get_elevation, cam_lat, cam_lon, mount_height_m)` does this; `georeference_terrain.py` uses it when you pass `--height-above-ground`.  
- **Flat-ground workflows:** Camera height is “height above ground”; the model assumes the ground is a horizontal plane at that reference.

---

## 8. Quality checks and troubleshooting

- **Calibration:** RMS &lt; 1.0 px; per-image RMS table free of obvious outliers.  
- **IMU:** Magnetometer calibration status = 3 before use; orientation stable when camera is stationary.  
- **Georeferencing:**  
  - Check a few GCPs: click in the image and compare predicted (lat, lon) to known values.  
  - If errors grow with distance from center, consider **distortion** (undistort is applied when using the tool correctly) or **intrinsics** (wrong resolution or bad calibration).  
  - If errors are systematic in one direction, check **heading** (and magnetometer calibration) or **pitch** sign (downward = negative).  
- **Terrain:** If using DEM/LAS, ensure **no-data** and **extent** cover your area; otherwise rays may miss the surface and return no coordinate.

---

## Summary checklist

| Step | Action |
|------|--------|
| 1 | Print checkerboard; set `BOARD_W`, `BOARD_H`, `SQUARE_SIZE_M` in `camera_calibration.py`. |
| 2 | Capture 15–25 calibration images with **varied** poses; run `camera_calibration.py`; aim RMS &lt; 1.0 px; save `calibration.json` and camera height. |
| 3 | Calibrate BNO055/BNO08x magnetometer (figure-8 until status 3); align IMU with camera; log or embed orientation (CSV or EXIF UserComment). |
| 4 | Collect 3–5+ GCPs (surveyed or from map); record lat, lon, elevation (datum), and where they appear in the image. |
| 5 | Capture photos with known GPS and orientation (fixed camera: one survey + IMU; mobile: per-image EXIF or log). |
| 6 | Run `georeference_tool.py` (flat) or `georeference_terrain.py` (DEM/LAS); use GCPs to validate and tune height/heading/pitch if needed. |

This procedure, together with the scripts in this repository, should allow a researcher to **calibrate the camera**, **calibrate the IMU**, **collect GCPs**, and **georeference photos** with consistent conventions and attention to accuracy (including vertical datum when using terrain).
