# UFO-Net Georeferencing Pipeline: Independent Study Report
**Author:** Mandeep Shergill  
**Supervisor:** [Supervisor Name]  
**Course:** Independent Study  
**Institution:** Syracuse University  
**Date:** April 2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background: Georeferencing Techniques](#2-background-georeferencing-techniques)
3. [System Architecture](#3-system-architecture)
4. [Work Completed This Semester](#4-work-completed-this-semester)
5. [Validation and Accuracy Assessment](#5-validation-and-accuracy-assessment)
6. [Results](#6-results)
7. [Future Work (Summer)](#7-future-work-summer)
8. [Appendix A: Error Budget](#appendix-a-error-budget)
9. [Appendix B: File and Module Reference](#appendix-b-file-and-module-reference)

---

## 1. Project Overview

The Urban Flood Observation Network (UFO-Net) is a distributed IoT sensor network designed to provide real-time, spatially-resolved observations of urban flooding. The network deploys multispectral and thermal imaging sensors on fixed urban infrastructure — lampposts, utility poles, bridge abutments — at heights of roughly 5–15 m, oriented steeply downward toward road surfaces. An edge-AI classifier running on each node segments flood pixels from the camera image in near-real time and transmits a binary flood mask over the cellular network.

The pixel coordinates output by the classifier are not directly useful for flood mapping. A flood mask identifies which rows and columns of an image are wet, but does not answer the question a city engineer or hydraulic modeler needs answered: *where on the ground is the water?* Converting pixel coordinates to geographic coordinates — georeferencing — is the problem this independent study addresses.

**Project goals:**

1. Develop a portable, documented software pipeline that converts a pixel mask from any UFO-Net camera node into a georeferenced flood boundary polygon.
2. Quantify the spatial accuracy of the georeferenced output as a function of the input data quality (GPS, IMU orientation, terrain model).
3. Demonstrate the complete chain — from raw camera image to GIS-ingestible output — using real data collected at the Meadowbrook-006 monitoring site in Syracuse, NY.
4. Establish a ground control point (GCP) workflow using ArUco fiducial markers and RTK photogrammetry so that systematic georeferencing errors can be measured and corrected at each node installation.

The target spatial accuracy is 5 cm at ground level, which is the resolution at which road drainage features (curb lips, drain inlets, road crown) meaningfully influence flood routing. As documented in the error budget (Appendix A), this target is currently limited to approximately 8–22 cm by IMU heading uncertainty, but the pipeline is architected so that improvements to sensor hardware or additional GCPs directly translate to improved output accuracy.

---

## 2. Background: Georeferencing Techniques

### 2.1 What Georeferencing Means for Camera Imagery

Georeferencing is the process of assigning geographic coordinates to each pixel of an image. For satellite and aerial imagery acquired with rigidly stabilized platforms, this is routinely accomplished using the sensor's orbital or flight parameters combined with a digital elevation model. Ground-level cameras present harder problems: the orientation of a street-mounted camera is far less precisely known than that of an aerial sensor, and the terrain the camera is viewing may vary significantly across the image footprint.

The mathematical core is the *camera projection model*. Given a calibrated camera with intrinsic matrix **K** and a rotation matrix **R** describing the camera's orientation relative to the world coordinate frame, a pixel at image coordinates *(u, v)* corresponds to a ray in 3D space:

$$\mathbf{r}_{\text{cam}} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \qquad \mathbf{r}_{\text{world}} = R^\top \mathbf{r}_{\text{cam}}$$

The geographic location of the ground point visible at pixel *(u, v)* is found by intersecting this ray with a surface model. The choice of surface model is the central design decision in any camera georeferencing system, and it is discussed in detail below.

### 2.2 Coordinate Conventions

This project uses the **ENU (East-North-Up)** local tangent plane as the world frame, with the camera position as origin:

- **X** = East  
- **Y** = North  
- **Z** = Up  

The camera frame follows the OpenCV convention:
- **X** = right  
- **Y** = down (into the image)  
- **Z** = forward (into the scene)  

Camera orientation is specified by three angles:
- **Heading (yaw):** compass bearing of the camera boresight, 0° = North, 90° = East
- **Pitch:** tilt from horizontal; 0° = level, −90° = straight down
- **Roll:** rotation about the boresight; 0° = level, positive = right side tilts down

These conventions are standardized in `camera_geometry.py` and are shared across all scripts in the pipeline.

### 2.3 Camera Intrinsic Calibration

Before any georeferencing can be performed, the camera's *intrinsic parameters* must be known. These are:
- **Focal length** (f_x, f_y in pixels): determines the angle corresponding to each pixel
- **Principal point** (c_x, c_y): the pixel where the optical axis intersects the sensor plane
- **Distortion coefficients**: characterize the lens aberrations that cause straight lines in the world to appear curved in the image

The dominant distortion in the OV5647 lens used in UFO-Net is *barrel distortion* (k₁ ≈ −0.44), which causes objects near the image corners to be displaced significantly from their true projection. At the full 2592×1944 resolution, this displacement reaches approximately 180 pixels in the corners, equivalent to ~53 cm of ground error at 5 m range. Correcting for it reduces this to less than 1 cm.

The standard method for estimating these parameters is **Zhang's checkerboard method** implemented in OpenCV's `calibrateCamera`. A flat checkerboard is photographed in 15–25 poses; the known geometry of the grid provides the constraints needed to solve for **K** and **D** simultaneously. Quality is assessed by the **RMS reprojection error** — how closely the calibrated model re-projects the detected corners back to their observed positions in each image. An RMS below 0.5 pixels is considered excellent; above 1.0 pixels warrants recalibration.

The calibration parameters are stored in a `calibration.json` file and reused for all images from the same camera, since intrinsics do not change as long as the lens focus is fixed.

### 2.4 Surface Models: Flat Ground vs. Terrain Intersection

The simplest surface model is a **horizontal ground plane** at a fixed elevation. This is computationally trivial — each ray from the camera intersects a plane z = 0 at a unique point — and is valid when the ground is genuinely flat and the camera is looking close to straight down.

For urban flood monitoring, however, the flat-plane assumption is inadequate because:
- Road camber (typically 1–3% cross-slope) shifts the intersection point by several centimetres per metre of range
- Curb heights (typically 10–15 cm) cause large lateral displacement at close range in oblique views
- Drain inlet depressions and road crown relief span vertical ranges comparable to the flood depths being measured

The primary method used in this project is **ray-terrain intersection with a Digital Surface Model (DSM)**. Instead of a plane, the surface is represented by a raster of elevation values. The camera ray is marched forward in small steps; at each step, the ray's current height is compared to the DSM elevation at that horizontal position. A zero crossing — where the ray transitions from above the surface to below — is found by a coarse march followed by bisection, giving a precise intersection point.

Two terrain data sources are used in this project:

- **USGS 1-meter DEM:** A nationwide 1-meter resolution digital elevation model from the U.S. Geological Survey's 3D Elevation Program. This provides broad coverage at moderate accuracy.
- **Pix4DCatch point cloud / DSM:** A high-accuracy (1–3 cm horizontal) DSM generated from ground-level photogrammetry using the iPhone 13 Pro with viDoc RTK receiver. Coverage is limited to the immediate area surveyed on foot, but accuracy is far higher than the national DEM.

### 2.5 Ground Control Points

**Ground control points (GCPs)** are points whose both pixel location (in the camera image) and geographic location (on the ground) are known precisely. They serve two roles:

1. **Validation:** Clicking a GCP in the interactive georeferencing tool and comparing the computed coordinates against the known ground truth directly measures the total end-to-end error of the pipeline.
2. **Pose refinement:** With three or more GCPs, it is possible to *refine* the camera's estimated pose — adjusting heading, pitch, roll, and position — by minimizing the residuals between predicted and known GCP positions. This is implemented via nonlinear least-squares optimization in `gcp.py`.

For the UFO-Net pipeline, GCPs are collected using **ArUco fiducial markers** — machine-readable black-and-white square patterns with unique IDs. Placing ArUco markers in the camera's field of view before a Pix4DCatch scan allows the same markers to be located in both the UFONet photo (giving pixel coordinates) and the Pix4DCatch RTK depth maps (giving world coordinates), fully automating GCP generation.

### 2.6 Vertical Datums

A frequently overlooked but critical source of error in terrain-aware georeferencing is **vertical datum inconsistency**. The elevation value stored in a camera's GPS EXIF metadata is typically an *ellipsoidal height* — measured above the WGS84 reference ellipsoid. Most DEMs, however, store *orthometric heights* — elevations above mean sea level as approximated by a geoid model (commonly EGM96, EGM2008, or NAVD88).

The difference between ellipsoidal and orthometric height depends on location; in the Syracuse, NY area it is approximately −34 m (the geoid is ~34 m below the ellipsoid). If EXIF altitude is naively compared to DEM elevation without correcting for this offset, the camera appears to be 34 m higher than the terrain, causing all ray-terrain intersections to be projected far downrange.

This project implements explicit vertical datum tracking in `vertical_datum.py`, and the primary workaround — using `--height-above-ground` measured with a tape measure plus DEM elevation at the camera location — avoids the issue entirely by ensuring camera and terrain heights are in the same datum by construction.

### 2.7 Related Techniques

Several related georeferencing techniques were considered during this project's design:

**Structure-from-Motion (SfM):** Building a 3D model from multiple overlapping images and aligning it to GPS. This is how Pix4DCatch works. It provides highly accurate terrain models but requires multiple images and is not suitable for real-time operation at a fixed sensor node.

**GCP-only warping (TPS):** With six or more GCPs, a thin-plate spline (TPS) can warp pixel coordinates directly to geographic coordinates without using a camera model at all. This is implemented in `gcp.py` using scipy's `RBFInterpolator`. TPS is useful for correcting residual errors after camera-model georeferencing but does not provide elevation information.

**Orthorectification:** Used for aerial and satellite imagery. The image is resampled so that every pixel corresponds to a consistent ground resolution. For an oblique ground-level camera this is not directly applicable, but Planet PlanetScope orthorectified GeoTIFFs are used in this project as reference imagery for GCP extraction.

---

## 3. System Architecture

### 3.1 Hardware

Each UFO-Net node ("HazMapper unit") consists of:

| Component | Details |
|-----------|---------|
| **Camera** | Dorhea 5MP OmniVision OV5647, 3.6 mm focal length, RGB+NIR |
| **Thermal sensor** | FLIR Lepton 3.5, 160×120 px, LWIR (8–14 µm) |
| **IMU** | Bosch BNO055, 9-DOF (accelerometer, gyroscope, magnetometer) |
| **GNSS** | Quectel cellular modem, integrated GNSS, 1–3 m accuracy |
| **Compute** | Raspberry Pi (edge AI, image processing, telemetry) |
| **Survey tool** | Apple iPhone 13 Pro + viDoc RTK rover, Pix4DCatch software |

The OV5647 camera (unit UFO-006 at Meadowbrook) is mounted on a 33-inch pole at 0.8382 m above ground, oriented west at a fixed heading of 265°.

### 3.2 Software Stack

The pipeline is written in Python 3.13 and organized around a set of focused modules:

```
camera_geometry.py      — rotation matrix, coordinate conventions (shared)
geo_core.py             — pixel rays, flat-ground intersection
georeference_terrain.py — primary tool: DEM/LAS ray-terrain intersection, interactive
georeference_tool.py    — secondary tool: flat-ground, GCP pose refinement
gcp.py                  — GCP loading/saving, pose refinement, TPS warp
aruco_gcp.py            — ArUco detection, Pix4DCatch GCP extraction
unit_config.py          — per-sensor JSON configuration loader
vertical_datum.py       — ellipsoidal/orthometric datum conversion
validate_georef.py      — accuracy validation against Pix4DCatch RTK
camera_calibration.py   — checkerboard calibration
scripts/pix4d_to_las_dem.py   — Pix4DCatch OPF → LAZ + DEM GeoTIFF
scripts/pix4d_to_viewer.py    — self-contained HTML 3D point cloud viewer
planet_gcp_match.py     — automatic GCPs from Planet ortho imagery
planet_gcp_manual.py    — manual GCPs from Planet ortho imagery
```

Core dependencies: `opencv-python`, `numpy`, `pyproj`, `Pillow`, `piexif`, `scipy`, `rasterio`, `laspy[lazrs]`.

### 3.3 Data Flow

```
Camera image (EXIF GPS, EXIF IMU)
       │
       ▼
unit_config.json  ──►  Resolve pose: heading, pitch, roll, mount height
       │
       ▼
camera_calibration.json  ──►  Undistort image, get K_undist
       │
       ▼
USGS DEM / Pix4DCatch DSM  ──►  Build terrain provider (get_elevation)
       │
       ▼
georeference_terrain.py  ──►  Ray-terrain intersection per click/pixel
       │
       ▼
georeferenced_points.csv  ──►  (lat, lon, elev_m, slant_range_m)
       │
       ▼
GeoJSON / GeoTIFF export
```

---

## 4. Work Completed This Semester

### 4.1 Camera Calibration

Using 25 checkerboard images captured at varied distances and angles, the OV5647 camera on unit UFO-006 was calibrated using Zhang's method via OpenCV's `calibrateCamera`. The resulting calibration parameters are:

| Parameter | Value |
|-----------|-------|
| Focal length f_x | 2944.74 px |
| Focal length f_y | 2945.02 px |
| Principal point c_x | 1271.28 px |
| Principal point c_y | 932.89 px |
| Radial distortion k₁ | −0.4402 |
| Radial distortion k₂ | +0.2642 |
| Radial distortion k₃ | −0.0830 |
| Tangential p₁, p₂ | −0.00071, +0.00014 |
| **RMS reprojection error** | **0.278 px** |
| Image resolution | 2592 × 1944 px |

The RMS of 0.278 px is well below the 0.5 px target for excellent calibration. The barrel distortion (k₁ = −0.44) is strong enough that corner pixels are displaced approximately 180 px from their undistorted positions; undistortion is applied as the first step in every subsequent georeferencing operation.

### 4.2 Core Georeferencing Modules

**`camera_geometry.py`** implements a single `build_rotation_matrix(heading, pitch, roll)` function that constructs the world-to-camera rotation matrix from compass-convention orientation angles. This module is imported by every other script that constructs camera rays, ensuring consistent angle conventions across the pipeline. The rotation is built by constructing camera axes explicitly in ENU space, which avoids Euler-angle ambiguities in the gimbal lock region.

**`geo_core.py`** provides the flat-ground ray-intersection functions (`pixel_to_world_flat`, `intersect_flat`, `camera_elev_from_dem`) as a shared library. These are used by `georeference_tool.py` and by the `gcp.py` pose refinement, which needs a fast approximate model.

**`vertical_datum.py`** implements datum-aware elevation handling. It provides named constants for supported vertical datums (`wgs84_ellipsoid`, `egm96`, `egm2008`, `navd88`) and conversion functions via PROJ's compound CRS support. When PROJ geoid grids are not installed, it degrades gracefully and warns the user rather than silently producing wrong values.

### 4.3 Terrain-Aware Georeferencing (`georeference_terrain.py`)

This is the primary output of the semester's work. The script intersects camera rays with a terrain surface (DEM GeoTIFF and/or rasterized LAS/LAZ point cloud) using a robust bisection algorithm:

1. **Coarse march:** Step along the ray at configurable intervals (default 0.5 m), evaluating terrain height minus ray height at each step.
2. **Sign-change bracket:** When the ray transitions from above the terrain to below, record the bracketing step pair.
3. **Bisection refinement:** Iterate to a configurable tolerance (default 0.2 m vertical) using binary search within the bracket.
4. **Linear fallback:** If bisection fails (e.g., nodata cells within the bracket), fall back to linear interpolation across the bracket.

The algorithm correctly handles several edge cases: DEM nodata at the camera location, rays that briefly pass underground on slopes, and cameras located at/below local terrain elevation (a common failure mode when mixing ellipsoidal EXIF altitude with orthometric DEMs).

The interactive session presents an OpenCV window with the undistorted image. Left-clicking a pixel fires the ray-terrain intersection, prints the resulting (lat, lon, elevation, slant range) to the terminal, and overlays the result on the image with resolution-proportional annotation. Right-clicking labels the point; pressing **S** exports a CSV; pressing **Q** exits.

Key capabilities added during the semester:
- Dual terrain source: DEM GeoTIFF (via `rasterio`) and/or LAS/LAZ (via `laspy`), with the LAS rasterized at configurable resolution
- Coverage-check warnings at startup when the camera is outside terrain bounds
- Resolution-proportional font and marker scaling so the window is legible at any image resolution
- Unit configuration loading via `--unit-config` (see §4.5 below)
- Pre-loaded GCP overlay via `--gcps`
- ArUco marker overlay via `--aruco-dict`

### 4.4 GCP System (`gcp.py`)

**`gcp.py`** provides a data class for individual GCPs and two higher-level functions:

**`refine_pose_from_gcps(gcps, K, heading, pitch, roll, lat, lon, height)`** performs a 6-degree-of-freedom pose optimization using `scipy.optimize.least_squares`. Starting from an initial pose estimate (typically from EXIF or unit config), it adjusts heading, pitch, roll, camera latitude, camera longitude, and mount height to minimize the geographic residuals at the provided GCPs. Minimum 3 GCPs are required; 5+ distributed across the image give stable results.

**`fit_tps_warp(gcps)`** fits a thin-plate spline from pixel (u, v) to geographic (lat, lon) using `scipy.interpolate.RBFInterpolator`. With 6 or more well-distributed GCPs, this warp captures residual errors that the camera model cannot explain (e.g., small lens model errors, slight mount flex). The TPS warp is used in the flat-ground tool as an optional override; for terrain work, the camera model + DSM is preferred for elevation accuracy.

### 4.5 Per-Unit Configuration System (`unit_config.py`, `unit_config_UFO006.json`)

Early in the semester, it became clear that hard-coding sensor-specific values (mount height, heading, calibration file path) into scripts was not sustainable for a network of multiple sensor nodes. The unit configuration system addresses this by loading these values from a per-unit JSON file.

The configuration file for UFO-006 is:

```json
{
  "unit_id":           "UFO-006",
  "calibration":       "./calibration.json",
  "mount_height_m":    0.8382,
  "heading_deg":       265.0,
  "pitch_deg":         null,
  "roll_deg":          null,
  "camera_elev_datum": "wgs84_ellipsoid",
  "notes":             "OV5647 camera, west-facing fixed mount, 33-inch pole"
}
```

Fields set to `null` are filled from EXIF metadata or left at script defaults. The precedence chain is:

```
CLI args  >  unit_config.json  >  EXIF  >  script defaults
```

This means a researcher can override any value at the command line for testing while the deployed defaults remain correct for routine processing. The `unit_config.py` module exposes `resolve_heading()`, `resolve_pitch()`, `resolve_roll()`, `resolve_mount_height()`, and `resolve_calibration()` methods that each return a `(value, source_label)` tuple so the pipeline can print exactly where each parameter came from.

### 4.6 Pix4DCatch Data Format and Processing

The viDoc RTK + Pix4DCatch workflow produces an OPF (Open Photogrammetry Format) directory for each field scan. The format was reverse-engineered and documented in `docs/PIX4DCATCH_DATA_FORMAT.md`. Key elements:

- **`opf_files/input_cameras.json`:** Per-frame GPS (lat/lon/ellipsoidal altitude, ±10–30 cm horizontal) and orientation (yaw/pitch/roll, ±1–2°)
- **`opf_files/projected_input_cameras.json`:** Per-frame orientation in the local scene frame, using a different Euler convention
- **`opf_files/scene_reference_frame.json`:** CRS definition (NAD83(2011) / UTM Zone 18N, EPSG:6347) and local-to-UTM shift vector
- **`images/Image_XXXXXX.jpg`:** RGB frames at 1920×1440 px
- **`images/DepthMap_XXXXXX.tiff`:** LiDAR depth maps at 192×256 px (float32, metres), 1/7.5 scale of the color frame
- **`point_clouds/legacy/pointcloud.bin`:** Interleaved POSITION+COLOR float32, 24 bytes/point, local scene coordinates
- **`geolocations/rtkGPS.csv`:** 10 Hz RTK GPS stream

Rotation convention for `projected_input_cameras.json` (camera → world ENU):

$$R = R_x(\omega) \cdot R_y(\phi) \cdot R_z(\kappa)$$

with camera frame X=right, Y=up, Z=backward (OpenGL convention). This was verified empirically: for ground-facing Pix4DCatch frames, the decoded elevation angles fall between −23° and −43° (downward), consistent with walking-height imagery.

**`scripts/pix4d_to_las_dem.py`** converts a Pix4DCatch OPF directory to three outputs:
- `<scan>.laz`: point cloud in NAD83(2011)/UTM 18N, readable by `laspy` and PDAL
- `<scan>_dem.tif`: gridded DEM GeoTIFF at configurable resolution (default 5 cm) using either Delaunay interpolation or max-Z gridding
- `<scan>_camera_poses.csv`: per-frame GPS and orientation, formatted for direct use as `--lat`/`--lon`/`--heading` inputs to `georeference_terrain.py`

**`scripts/pix4d_to_viewer.py`** generates a self-contained HTML file that renders the point cloud in 3D in a browser using Three.js, with no server dependency.

There are 17 Pix4DCatch scan sessions available covering the Meadowbrook area, collected between July 2024 and April 2026. These are stored at `/var/home/manu/UFONet iPhone Data/`.

### 4.7 ArUco GCP Workflow (`aruco_gcp.py`)

The ArUco GCP module automates the most labor-intensive step of the georeferencing validation workflow. Manual GCP collection requires visiting the field with a GPS receiver, measuring the exact positions of visible targets, photographing the targets from the node, and manually clicking the target locations in the georeferencing tool. The ArUco workflow reduces this to placing printed markers and running a script.

**Module design:**

`detect_in_photo(image, K, D, dict_name)` — Undistorts the image, runs the OpenCV ArUco detector (`ArucoDetector` with `DICT_4X4_50`), and returns for each detected marker ID its pixel centroid and corner positions in the undistorted image coordinate system.

`locate_in_pix4d(session_dir, dict_name, every_n_frames, ...)` — Scans a Pix4DCatch session directory frame by frame, detecting ArUco markers in the color images. For each detection, the corresponding depth map is sampled at the marker center (scaled by the 1:7.5 depth-to-color ratio), and the 3D world position is computed by back-projecting through the camera pose:

```python
x_c =  (u - cx) / fx * depth_m
y_c = -(v - cy) / fy * depth_m   # camera Y up, image v down
z_c = -depth_m                    # scene in -Z (camera Z backward)
enu_offset = R_c2w @ [x_c, y_c, z_c]
world_xyz  = cam_pos_utm + enu_offset
```

Observations from multiple frames are averaged; markers seen in fewer than `min_views` frames are discarded. The result is a dictionary keyed by marker ID, giving (lat, lon, elev_m) with multi-view standard deviation as a quality indicator.

`write_gcp_csv(photo_detections, pix4d_locations, output_path)` — Merges the pixel coordinates from the UFONet photo with the world coordinates from Pix4DCatch into a GCP CSV file compatible with `gcp.py`.

**Integration with `georeference_terrain.py`:**

In the same session that `aruco_gcp.py` was finalized, ArUco detection was integrated into the `TerrainGeoreferencer` interactive class. On startup, if `--aruco-dict` is specified, detected markers are overlaid on the undistorted image as cyan crosses labeled with their IDs. If `--gcps` points to a pre-computed GCP CSV from `aruco_gcp.py`, those GCPs are pre-loaded into the session automatically. This eliminates manual pixel-clicking for any marker that was both detected in the UFONet photo and located in the Pix4DCatch scan.

**Recommended marker design:**

- Material: weather-proof PVC or aluminium composite, matte finish
- Size: 15×15 cm minimum (ensures detection from 5–15 m mounting height)
- Dictionary: `DICT_4X4_50` (50 unique IDs, robust to partial occlusion)
- Deployment: 4–6 markers per sensor location, distributed across the full image footprint
- NIR compatibility: standard black/white print is detectable in NIR channel (OV5647 is sensitive to 850 nm)

### 4.8 Validation Framework (`validate_georef.py`)

The `validate_georef.py` script provides a structured accuracy assessment for any image-sensor pair. It:

1. Reads camera pose from EXIF or unit config
2. Loads the USGS DEM and reads terrain elevation at the camera location
3. Georeferences the image center pixel using both flat-ground and terrain methods, and computes the displacement between them
4. Georeferences a 7×7 grid of pixels spanning the image and reports the geographic footprint
5. Loads the Pix4DCatch scan directory and computes the RTK camera-position centroid
6. Checks whether the georeferenced footprint overlaps the Pix4DCatch spatial extent

This produces a text report covering:
- Camera pose sources (EXIF vs unit config vs CLI)
- Vertical datum diagnostics
- Flat vs terrain displacement at image center
- Estimated footprint dimensions
- Overlap assessment with RTK ground truth

### 4.9 Planet Ortho GCP Tools

Two scripts support generating GCPs from PlanetScope orthorectified GeoTIFFs:

**`planet_gcp_match.py`:** Uses ORB feature matching between the (undistorted) field photo and a co-registered Planet GeoTIFF to automatically identify corresponding pixel-geographic pairs. This produces a GCP CSV suitable for pose refinement in `georeference_tool.py`.

**`planet_gcp_manual.py`:** An interactive dual-window tool where the user manually clicks corresponding points in the field photo and the Planet ortho. Used when automatic matching fails due to seasonal change, shadows, or strong perspective difference.

**`planet_scene_pull.py`:** Uses the Planet API to fetch PlanetScope 3-band analytic surface reflectance scenes that contain a given GPS coordinate, for use as the reference GeoTIFF in the above scripts.

### 4.10 Tests

A test suite in `tests/` covers the core numerical routines:

| Test file | What it verifies |
|-----------|-----------------|
| `test_camera_geometry.py` | Rotation matrix orthogonality; ray directions for known headings |
| `test_pixel_rays.py` | `pixel_ray()` correctly inverts K; known-pixel checks |
| `test_flat_ground.py` | `pixel_to_world_flat()` geometry at simple angles |
| `test_terrain_ray.py` | Ray-terrain bisection converges; handles nodata |
| `test_gcp.py` | GCP CSV load/save round-trip; pose refinement convergence |
| `test_vertical_datum.py` | Datum conversion produces expected sign/magnitude |
| `test_intrinsics_scaling.py` | K scaling with resolution change |
| `test_planet_gcp_match.py` | Homography computation and GCP extraction |
| `test_dsm_validation.py` | DSM validation script output format |

Tests are run with `pytest` from the repository root.

### 4.11 Documentation

Four detailed reference documents were written to support current and future researchers:

- **`docs/GEOREFERENCING_PROCEDURE.md`:** Step-by-step checklist for camera calibration, IMU calibration, GCP collection, and georeferencing workflow.
- **`docs/GEOREFERENCING_PROCESS_DETAILED.md`:** Terrain-first operational workflow for `georeference_terrain.py`, including terrain data inspection, datum handling, and Planet/Pix4DCatch integration.
- **`docs/ACCURACY_AND_EXTERNAL_RESOURCES.md`:** Error decomposition and strategies for improving accuracy when onboard GPS/IMU is coarse.
- **`docs/PIX4DCATCH_DATA_FORMAT.md`:** Complete description of the OPF format, binary point cloud encoding, coordinate transforms, and Linux processing pipeline.

---

## 5. Validation and Accuracy Assessment

### 5.1 Calibration Accuracy

The OV5647 calibration achieved RMS = 0.278 px across 25 images. The per-image reprojection error table showed no significant outliers, and the distortion model visually eliminates the barrel distortion visible in raw images.

### 5.2 Terrain Intersection Diagnostics — Meadowbrook-006

Full end-to-end validation was run on image `20260426-090402-NIR-OFF.jpg` using `validate_georef.py`:

```bash
python validate_georef.py \
    --image Meadowbrook-006/20260426-090402-NIR-OFF.jpg \
    --dem USGS_1M_18_x41y477_NY_FEMAR2_Central_2018_D19.tif \
    --pix4d "/var/home/manu/iPhone UFONet/2026-04-24-13-11-52" \
    --unit-config unit_config_UFO006.json \
    --grid 9
```

Key results from this run:

| Metric | Value |
|--------|-------|
| EXIF GPS position | 43.039784°N, −76.082792°W |
| EXIF altitude | 156.00 m (orthometric; GPS noise ±8.5 m vs DEM) |
| DEM terrain at camera (USGS) | 147.47 m |
| Geoid separation (from RTK) | −34.430 m (confirmed by Pix4DCatch rtkGPS.csv) |
| Camera AGL used | 0.9652 m (unit config; measured at Meadowbrook-006 site) |
| Heading used | 265.0° (unit config, overriding EXIF Yaw = 359.9°) |
| Center pixel — flat ground | 43.039783°N, −76.082809°W |
| Center pixel — terrain-aware | 43.039783°N, −76.082804°W, elev 147.62 m |
| Flat vs. terrain difference | 0.4 m |
| Terrain intersection success rate | 81 / 81 pixels (100%) |
| Georeferenced footprint (9×9 grid) | 6 m² (~2×2 m), slant 1.2–3.2 m, mean 1.7 m |

The 0.4 m flat vs. terrain difference is small because the camera is at 0.97 m AGL; at production mount heights of 5–15 m the terrain correction becomes the dominant factor. Mount heights vary unit to unit and must be measured and set in each node's `unit_config.json`.

**Key debugging discoveries during this test:**

- **EXIF heading vs. physical heading:** The EXIF Yaw field recorded 359.9° (approximately north), while the camera is physically oriented at 265° (west). The EXIF Yaw is the GPS track direction at the time of capture, not the fixed mount heading. The unit_config system provides the correct resolution.
- **EXIF altitude datum:** The EXIF GPSAltitude of 156 m is approximately orthometric. The USGS DEM gives 147.47 m at the same location — an 8.5 m discrepancy consistent with typical smartphone GNSS vertical noise. The pipeline ignores EXIF altitude and uses `mount_height_m` from unit config instead.
- **EPSG regex in OPF files:** The OPF CRS WKT string contains multiple nested `ID["EPSG", ...]` fields; the innermost is the base geographic CRS (EPSG:6318), while the outermost is the projected CRS (EPSG:6347). Using `re.search()` returns the wrong value; the fix uses `re.findall()[-1]`.
- **Pix4DCatch DSM vertical datum:** The `scripts/pix4d_to_las_dem.py` script initially wrote DSM Z values in ellipsoidal heights (~111 m), not orthometric. The fix reads the mean `GeoidSeparation` column from `rtkGPS.csv` (N = −34.430 m for this site) and applies `Z_ortho = Z_ellip − N`, yielding orthometric elevations of 143–148 m consistent with the USGS DEM.

### 5.3 Overlap with Pix4DCatch RTK Extent

The best available Pix4DCatch scan for validation is session `2026-04-24-13-11-52`: it is the only session whose bounding box contains the camera position and has 99.2% rtkFloat/Fixed quality (7,801 / 7,859 samples), providing 1–3 cm horizontal accuracy across a ~82×77 m area.

Results from `validate_georef.py` comparing the photo footprint to the Pix4DCatch scan:

| Metric | Value |
|--------|-------|
| Footprint overlaps Pix4DCatch scan bbox | Yes |
| Footprint overlaps Pix4DCatch camera-path bbox | Yes |
| Footprint centroid | 43.039783°N, −76.082815°W |
| Pix4DCatch RTK centroid | 43.040016°N, −76.082998°W |
| Centroid separation | 30.0 m |
| Nearest Pix4DCatch camera frame to footprint | 7.4 m |

The 30.0 m centroid separation reflects the size of the overall scan area (the iPhone walked ~82 m), not a georeferencing error. The nearest Pix4DCatch frame being 7.4 m from the footprint centroid confirms the scan passed near the camera's field of view.

**Coverage limitation:** The camera footprint (a ~2×2 m area ~1–3 m west of the pole) lies on the western edge of the Pix4DCatch scan where the iPhone walk path did not pass. No DSM data exists for the exact pixels the camera sees. Consequently, a GCP residual table — the quantitative accuracy measurement described in §7.1 — could not be produced from existing data and requires a dedicated field experiment with physical ArUco markers.

### 5.4 DSM Source Comparison

The `scripts/pix4d_to_las_dem.py` script was used to generate a 5 cm DSM from session `2026-04-24-13-11-52` (1,632,803 points, 1628×1688 px). At 44 co-located sample points where both the Pix4DCatch DSM and the USGS 1 m DEM have valid data, the elevation difference was:

| Metric | Value |
|--------|-------|
| Mean (Pix4DCatch − USGS) | −2.07 m |
| Std dev | 0.64 m |
| RMSE | 2.16 m |
| Range | −3.41 to −0.95 m |

The systematic −2 m offset likely reflects a combination of genuine terrain variation (the iPhone walked in a different subarea than the camera location) and residual geoid model uncertainty. The scan-area coverage is only 5.4% of the DSM grid, limiting statistical power. A dedicated scan covering the camera footprint would reduce this to a terrain resolution question rather than a coverage question.

### 5.5 Sources of Residual Error

| Source | Current magnitude | Status |
|--------|-------------------|--------|
| GPS position (EXIF, no RTK) | 2–8 m horizontal | Partially mitigated by Pix4DCatch footprint check |
| Heading (unit config, no IMU calibration or validation) | ~0.5–2° → ~4–17 cm at 5 m | **BNO055 magnetometer calibration not yet performed**; heading not yet validated at site |
| Terrain model resolution (USGS 1 m vs 5 cm Pix4DCatch) | Up to ~5 cm on road | Pix4DCatch DSM generated but camera footprint has no coverage |
| Depth map resolution (ArUco GCP at 192×256 px) | ~2–5 cm per-marker | Inherent limitation; averaged across views |
| GCP residual (empirical accuracy) | Not yet measured | Requires physical ArUco marker experiment (§7.1) |

---

## 6. Results

The following deliverables were produced this semester:

**Software pipeline:** A fully documented Python georeferencing toolkit comprising 15+ modules, a test suite with 9 test files, and 4 reference documents. The pipeline runs end-to-end from raw image to (lat, lon, elev) output for any node whose `unit_config.json` file is configured.

**Camera calibration:** A high-quality calibration (RMS 0.278 px) for the UFO-006 OV5647 camera, stored in `calibration.json` and ready for production use.

**Terrain georeferencing:** A working interactive tool (`georeference_terrain.py`) that correctly intersects camera rays with the USGS DEM and Pix4DCatch point clouds. Point-and-click geographic coordinate retrieval is functional for the Meadowbrook-006 site with the unit config applied.

**Pix4DCatch processing pipeline:** `scripts/pix4d_to_las_dem.py` converts any of the 17 available scan sessions to a LAZ point cloud and 5 cm DEM GeoTIFF in orthometric heights (geoid correction applied automatically from `rtkGPS.csv`), usable as high-accuracy terrain input.

**ArUco GCP framework:** A complete Python module (`aruco_gcp.py`) implementing the full ArUco GCP workflow: detection in the UFONet photo, localization in Pix4DCatch depth maps, and CSV export. The module is wired into the interactive georeferencing tool.

**Unit configuration system:** A portable, per-sensor JSON configuration system (`unit_config.py`) that makes the pipeline deployable to any new sensor node by creating a single configuration file.

**EGU 2026 preparation:** The georeferencing work was completed in advance of the EGU 2026 presentation (session HS5.4.3, 3–8 May, Vienna). The pipeline provides the technical foundation for the georeferenced flood extent outputs described in the presentation.

---

## 7. Future Work (Summer)

### 7.1 End-to-End GCP Validation Experiment

**Priority: High**

The most important outstanding task is a controlled validation experiment using physical ArUco markers. This requires:

1. Printing 6–8 ArUco DICT_4X4_50 markers (15×15 cm) on weather-proof PVC or aluminium
2. Placing them within the UFO-006 field of view at the Meadowbrook site
3. Collecting a Pix4DCatch scan while the markers are on the ground
4. Running `aruco_gcp.py --image photo.jpg --pix4d session/` to extract GCP coordinates
5. Running `georeference_terrain.py --gcps gcps.csv --aruco-dict DICT_4X4_50` to verify the pixel-to-world correspondence and measure residuals

The output will be a residual table (meters) for each GCP, constituting the first empirical accuracy measurement of the full pipeline. The expected residuals are 0.1–0.5 m (dominated by GPS horizontal error and heading uncertainty), but measuring them is essential for honest reporting.

### 7.2 DSM Source Comparison

**Priority: High**

The georef_workplan.md specifies a comparison between two surface models:

- **DSM-A (high accuracy):** Pix4DCatch 5 cm DSM from `pix4d_to_las_dem.py`
- **DSM-B (moderate accuracy):** PlanetScope-derived DSM or USGS 1 m DEM

For the same flood event image and camera pose, the same flood mask boundary pixels are georeferenced with each DSM. The spatial difference directly quantifies how much the terrain model quality affects flood extent accuracy. The comparison metrics are:
- Mean and 90th-percentile boundary displacement between Extent-A and Extent-B
- GCP residuals using each DSM
- Displacement at specific vertical relief features (curbs, drain inlets)

This analysis addresses the research question: *is the additional effort of a ground photogrammetric survey necessary, or does the national DEM suffice given the other error sources?*

### 7.3 GCP-Based Pose Refinement Integration

**Priority: Medium**

The pose refinement function in `gcp.py` (`refine_pose_from_gcps`) is not yet wired into the terrain georeferencing tool. Once the ArUco GCP validation experiment produces reliable GCPs, pose refinement should be integrated:

1. Load GCPs from `--gcps` CSV into `TerrainGeoreferencer`
2. Run `refine_pose_from_gcps` on startup to optimize heading, pitch, roll, and camera position
3. Use the refined pose for all subsequent ray-terrain intersections
4. Report initial vs. refined residuals

This should measurably reduce the ~0.1–0.5 m systematic errors attributable to GPS position and heading bias.

### 7.4 Flood Extent GeoJSON/GeoTIFF Export

**Priority: Medium**

The interactive tool currently exports a CSV of individually clicked points. For flood mapping, the output should be a polygon of the classified flood boundary. The necessary steps are:

1. Load a classified flood mask (binary raster from edge-AI classifier)
2. Georeference every boundary pixel using `pixel_to_gps_terrain` in batch mode
3. Apply a convex hull or alpha shape to the georeferenced boundary points
4. Export as GeoJSON (for QGIS) and/or GeoTIFF (georeferenced raster)

The GeoJSON polygon format is immediately ingestible by QGIS, Mapbox, and any hydraulic model that accepts spatial data, fulfilling the "GIS-compatible output" goal stated in the project overview.

### 7.5 Thermal-Optical Registration

**Priority: Medium**

The FLIR Lepton 3.5 thermal sensor is mounted alongside the OV5647 camera but does not share the same optical axis. Flood pixels detected in the thermal channel need to be transformed to the optical image's coordinate system before georeferencing, because the georeferencing pipeline is calibrated to the optical camera's K and D.

SimpleITK's `Similarity2DTransform` with Mattes Mutual Information has been identified as the registration method. The work to do:

1. Capture baseline image pairs (thermal + optical) of the dry road surface with the node installed
2. Compute the initial thermal-to-optical transform using SimpleITK
3. Save per-node baseline transforms to disk
4. Validate registration accuracy (target: <1 px residual after convergence)
5. Re-registration triggered by IMU displacement should initialize from the baseline transform

### 7.6 Site Survey for Remaining Nodes

**Priority: Medium**

The RTK survey and unit configuration system developed for UFO-006 needs to be replicated for all deployed nodes. For each additional node:

1. Occupy the camera position with the viDoc RTK receiver, recording lat/lon/altitude under NTRIP correction
2. Measure mount height above road surface (tape measure)
3. Determine camera heading from compass + site geometry (initial estimate; refine with GCPs)
4. Create `unit_config_<node_id>.json`
5. Collect a Pix4DCatch scan of the site footprint

This work should be scheduled for all deployments before flood season, since deployment-time surveys are far more efficient than post-hoc corrections.

### 7.7 IMU Calibration and Heading Validation

**Priority: High**

The BNO055 magnetometer calibration has not yet been performed for any deployed node. Without it, the magnetometer operates at calibration level 0–1 (out of 3), which can introduce heading errors well beyond the 2.5° RMS spec quoted for a fully calibrated sensor. This is the single highest-impact action before the next field session.

**Step 1 — Physical magnetometer calibration** (per `docs/GEOREFERENCING_PROCEDURE.md` §3):

1. Take the sensor away from metal structures, motors, and reinforced concrete
2. Move it in a figure-8 pattern through all orientations until BNO055 magnetometer status = 3
3. Save the calibration offsets with `bno055_calibration.py` so they persist across power cycles

**Step 2 — Heading validation against RTK ground truth:**

1. Use two RTK-surveyed features visible in both the camera image and the Pix4DCatch scan
2. Compute the true bearing between them from RTK coordinates
3. Compare to the IMU-derived bearing
4. Apply the residual as a fixed heading correction in the unit config

A systematic validation of this procedure across all nodes would provide the first empirical measure of heading accuracy in deployment conditions.

### 7.8 BNO085 Upgrade Assessment

**Priority: Low**

The BNO085 IMU offers approximately 1.0° RMS heading accuracy (vs. 2.5° for the BNO055). At 5 m range, this reduces the heading contribution to the error budget from ~22 cm to ~8.7 cm. This does not close the gap to the 5 cm target on its own, but combined with RTK-surveyed node position, improved terrain model, and GCP pose refinement, it makes the 5 cm target plausible. A hardware swap assessment should include the cost/availability of BNO085 breakout boards and the firmware changes required on the Raspberry Pi.

### 7.9 Per-Node Calibration Database

**Priority: Low**

As the network grows to multiple nodes, managing calibration files becomes a significant operational burden. A simple file-based database structure should be established:

```
config/
  UFO-006/
    unit_config.json
    calibration.json
    heading_correction.json
    survey_date.txt
  UFO-007/
    unit_config.json
    calibration.json
    ...
```

This structure should be version-controlled and indexed so the processing pipeline can locate the correct calibration file for any node ID without human intervention.

---

## Appendix A: Error Budget

The table below quantifies each error source in the georeferencing pipeline, its unmitigated effect on ground coordinate accuracy at 5 m range, and the mitigation implemented or planned.

| Error Source | Unmitigated Effect | Mitigation | Residual After Mitigation |
|---|---|---|---|
| Camera intrinsic error | ~53 cm corner displacement (k₁ = −0.44) | Checkerboard calibration, RMS 0.278 px | <1 cm |
| GPS position (Quectel, no corrections) | 1–3 m systematic anchor shift | RTK survey at installation (viDoc RTK, 1–3 cm) | 1–3 cm |
| Heading error (BNO055) | ~22 cm at 5 m range (2.5° RMS) | IMU calibration + heading validation from RTK survey | ~8 cm at 5 m (1° residual) |
| Pitch/roll error (BNO055) | ~17 cm per degree at 10 m | IMU calibration | ~2–5 cm (sub-degree residual) |
| Vertical datum mismatch | ~15–34 m apparent elevation error → large horizontal error | `--height-above-ground` + consistent datum | <1 cm |
| Surface model (USGS 1 m DEM) | ~5–15 cm on sloped ground | Pix4DCatch 5 cm DSM (planned) | ~1–2 cm |
| Depth map resolution (ArUco GCPs) | ~2–5 cm per-marker | Multi-view averaging (N≥5 frames) | ~1–2 cm |
| **Combined (current, best case)** | | | **~10–20 cm** |
| **Combined (planned, all mitigations)** | | | **~5–8 cm** |

The binding constraint after all planned mitigations are applied remains heading accuracy. Reaching the 5 cm target requires either a BNO085 upgrade or additional GCPs at each deployment.

---

## Appendix B: File and Module Reference

### Core Modules

| File | Purpose |
|------|---------|
| `camera_geometry.py` | `build_rotation_matrix(heading, pitch, roll)` — ENU ↔ camera rotation |
| `geo_core.py` | Flat-ground ray intersection, camera elevation from DEM |
| `gcp.py` | GCP data class, pose refinement, TPS warp, residual computation |
| `vertical_datum.py` | Ellipsoid/orthometric datum constants and conversion |
| `exif_imu.py` | EXIF GPS and IMU parsing (UserComment `Roll R Pitch P Yaw Y`) |
| `unit_config.py` | Per-unit JSON configuration loader, resolve_* methods |

### Primary Tools

| File | Purpose |
|------|---------|
| `georeference_terrain.py` | Interactive ray-terrain georeferencing (DEM + LAS) |
| `georeference_tool.py` | Interactive flat-ground georeferencing + GCP pose refinement |
| `aruco_gcp.py` | ArUco detection in photo + Pix4DCatch depth map localization |
| `validate_georef.py` | Accuracy validation against Pix4DCatch RTK |
| `camera_calibration.py` | Checkerboard calibration → calibration.json |
| `planet_gcp_match.py` | Automatic GCPs from Planet ortho imagery |
| `planet_gcp_manual.py` | Manual GCPs from Planet ortho imagery |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/pix4d_to_las_dem.py` | Pix4DCatch OPF → LAZ + DEM GeoTIFF + camera poses |
| `scripts/pix4d_to_viewer.py` | Pix4DCatch point cloud → self-contained 3D HTML viewer |
| `scripts/dump_dataset_crs.py` | Print CRS, bounds, nodata for DEM/LAS files |
| `scripts/validate_dsm.py` | Validate generated DSM GeoTIFF |

### Configuration Files

| File | Purpose |
|------|---------|
| `calibration.json` | OV5647 intrinsics (K, D) for UFO-006 |
| `unit_config_UFO006.json` | Sensor-specific parameters for UFO-006 node |

### Data Available

| Dataset | Location | Notes |
|---------|----------|-------|
| Pix4DCatch scans (17 sessions) | `/var/home/manu/UFONet iPhone Data/` | 2024-07 through 2026-04; rtkFloat GPS |
| USGS 1 m DEM (2 tiles) | `USGS_1M_18_x40y477_*.tif`, `USGS_1M_18_x41y477_*.tif` | NY FEMAR2 Central 2018 |
| USGS LAS point cloud | `USGS_LPC_NY_FEMAR2_Central_*.laz` | Two tiles covering Meadowbrook area |
| FEMA DEM (Barry Park) | `BarryPark-FEMA-1M-DEM-*.tif` | Alternative DEM for comparison |
| PlanetScope imagery | `1_psscene_analytic_sr_udm2/` | Feb 2026, surface reflectance |
| UFO-006 field images | `20260426-*.jpg`, `20260227-*.jpg` | NIR-ON and NIR-OFF pairs |

---

*This report documents work completed during the Spring 2026 semester. All code is in the repository at `/var/home/manu/git/UFONet Repos/Georeferencing/`.*
