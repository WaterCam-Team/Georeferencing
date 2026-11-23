# Georeferencing

Camera calibration and pixel→GPS georeferencing for fixed-position and mobile imagery (e.g. flood or field monitoring).

## Quick start

- **Calibrate camera:** `python camera_calibration.py` (checkerboard images → `calibration.json`)
- **Georeference (terrain — primary):** `python georeference_terrain.py photo.jpg --dem dem.tif --lat ... --lon ... --height-above-ground ...` (see [detailed terrain process](docs/GEOREFERENCING_PROCESS_DETAILED.md))
- **Inspect DEM/LAS metadata (CRS, bounds, vertical hints):** `python scripts/dump_dataset_crs.py --dem your.tif --las your.las` (optional `--las-crs EPSG` if the cloud has no CRS)
- **Georeference (flat ground — optional):** `python georeference_tool.py` (horizontal plane; rarely appropriate for real relief)

## Full procedure

**[How the flood map works](docs/HOW_IT_WORKS.md)** — the idea end to end: lens, pose, ground model, drawing the mask, accuracy.

**[Step-by-step procedure for researchers](docs/GEOREFERENCING_PROCEDURE.md)** — camera calibration, BNO055/BNO08x IMU calibration, GCP collection, and accurate photo georeferencing.

**[Detailed process (terrain-first)](docs/GEOREFERENCING_PROCESS_DETAILED.md)** — primary workflow for `georeference_terrain.py`, SU-WaterCam / Vidoc RTK / Pix4Dcatch notes, Planet/Pix4D ortho GCP scripts, and optional flat-ground tool.

**[Accuracy with coarse GPS/IMU](docs/ACCURACY_AND_EXTERNAL_RESOURCES.md)** — how to combine RTK/survey, ortho GCPs, terrain quality, calibration, and time sync to improve results.

## Heading and GCP requirements

- **Heading:** `--heading` > unit-config `heading_deg` (surveyed) > EXIF yaw. EXIF yaw from a magnetometer that wasn't fully calibrated (`CalibMag` below 3) is still used as the best estimate, but marked `exif_yaw_uncalibrated` (logged, and recorded in the exports' `heading_source`). There is no GPS-track fallback and no default of 0 deg: with no heading at all the tools stop with an error.
- **GCPs:** pose refinement needs at least 4 (3 give no redundancy). It reports per-point residuals in metres and a leave-one-out check that flags suspect points without dropping them.
- Flat-ground rays within 2 deg of the horizon or beyond 2000 m are rejected.

Details: [GCPs, heading sources and DEM notes](docs/GEOREFERENCING_PROCEDURE.md#44-heading-sources-ray-limits-and-dem-use).

## Main scripts

| Script | Purpose |
|--------|--------|
| `camera_calibration.py` | Checkerboard calibration → K, D, optional camera height; saves `calibration.json` |
| `camera_geometry.py` | Rotation matrix (ENU ↔ camera), re-exported from `ufonet_geo.geometry`; used by calibration and georeferencing |
| `georeference_tool.py` | Interactive click-to-GPS (flat ground), batch, optional GeoTIFF; uses calibration + IMU log or EXIF |
| `georeference_terrain.py` | Click-to-GPS with DEM and/or LAS/LAZ terrain intersection |
| `planet_gcp_match.py` | Generate a GCP CSV by matching a field photo to a Planet orthorectified GeoTIFF |
| `planet_gcp_manual.py` | Manual fallback: click corresponding points in field photo and Planet GeoTIFF |
| `scripts/dump_dataset_crs.py` | Print CRS, bounds, nodata, and quick stats for rasters and LAS/LAZ before terrain georeferencing |
| `add_imu.py` | Write Roll/Pitch/Yaw to EXIF UserComment for images |
| `gcp.py` | Load and save GCPs, refine the camera pose from GCPs, thin-plate-spline warp |
| `aruco_gcp.py` | Find ArUco markers in a photo and locate the same markers in Pix4DCatch frames |
| `label_gcp.py` | Click-to-label ArUco/GCP markers in a photo by hand |
| `print_aruco_markers.py` | Printable ArUco marker sheets |
| `drone_gcp_match.py` | Generate a GCP CSV by matching a field photo to a drone orthomosaic |
| `gcp_from_pix4d.py` | Interactive RTK coordinate extraction from Pix4DCatch frames |
| `validate_georef.py` | Accuracy validation of a fixed-mount node photo against DEM and Pix4DCatch scan |
| `water_georef_workflow.py` | Interactive end-to-end terrain georeferencing and water-boundary export |
| `flood_map.py` | Flood boundary extraction from a photo (SAM clicks) projected to GPS with DEM ray casting |
| `vertical_datum.py` | Ellipsoid versus orthometric height checks and conversion |
| `exif_imu.py` | EXIF GPS and IMU parsing used by the tools |
| `unit_config.py` | Per-unit sensor configuration loader (schema in `unit_config.example.json`) |
| `calib_filter.py`, `calib_filter_sensor_tree.py` | Pre-filter calibration images (sharpness, board geometry, diversity) |
| `undistort_tool.py` | Undistortion utilities and batch/interactive viewer |
| `planet_scene_pull.py` | Fetch PlanetScope scenes around a photo's EXIF location (API key required) |
| `georeference3d.py`, `georeference_sharp.py`, `reconstruct3d.py`, `laz-viewer.py` | Experimental: 3D reconstruction, splat viewer, structure from motion, point cloud viewer |
| `scripts/flood_extent_export.py` | Batch-project a binary flood mask to a GeoJSON polygon and GeoTIFF |
| `scripts/survey_to_pose.py` | RTK survey plus one photo with ArUco targets to the camera's surveyed pose |
| `scripts/scan_to_pose.py`, `scripts/scan_pose_overlay.py` | Camera pose from an RTK Pix4DCatch scan and one node photo, no targets; overlay and sweep check (see `docs/SCAN_TO_POSE.md`) |
| `scripts/scan_survey_package.py` | From the same scan: point cloud, site image, bare-ground DEM and the API survey manifest; field capture, checks and upload in `docs/SCAN_SITE_SURVEY.md` |
| `scripts/pix4d_to_las_dem.py` | Pix4DCatch scan to LAZ, DEM GeoTIFF and camera-pose CSV (see `docs/PIX4DCATCH_DATA_FORMAT.md`) |
| `scripts/pix4d_to_api_json.py`, `scripts/pix4d_to_viewer.py` | Export a scan to the API dashboard's JSON format, or to an HTML point-cloud viewer |
| `scripts/validate_dsm.py` | Quality checks on a DSM (see `docs/DSM_VALIDATION.md`) |
| `scripts/drone_dsm_ingest.py`, `scripts/odm_process.py` | Accept a drone DSM/DEM from any photogrammetry tool; submit drone images to NodeODM |
| `scripts/flood_export.py` | Compare DSM sources for flood mapping |
| `scripts/compare_mount_configs.py` | Compare EXIF pose with GCP-refined pose across mount configurations |
| `scripts/collect_subdir_jpegs.py` | Flatten JPEGs from a folder tree into one folder |

The full list with roles (library, core, helper, experimental) is the [module map](LLM_CONTEXT.md#module-map). Superseded scripts are in `archive/`.

## Shared package: `ufonet_geo`

The core math lives in the `ufonet_geo/` package. The scripts above use it,
and so does the WaterCam API server (the dashboard's georeferencing and flood map):

| Module | What |
|--------|------|
| `geometry` | rotation matrix, lens model (incl. the undistorted frame's camera matrix), rays, flat ground with the grazing/range guards, ENU ↔ lat/lon |
| `pose` | heading / pitch / roll / position from a unit config and EXIF (the precedence and `CalibMag` rules above) |
| `terrain` | DEM sampling, camera elevation, ray / terrain intersection |
| `ortho` | a fixed camera's flood mask on a north-up ground grid (`build_ground_lookup`, `GroundLookup.render`) |

It needs only numpy; the `dem` extra adds rasterio and pyproj. `camera_geometry.py`,
`geo_core.py`, `unit_config.py`, `exif_imu.py` and `georeference_terrain.py`
re-export from it, so existing imports keep working. Change the math in the
package, not in a copy.

Package version: `0.1.0` (`ufonet_geo.__version__`, requires Python 3.12 or newer; the repo's `.python-version` is 3.13).

How other projects use it:

- The API server pins it by commit in its `requirements.txt`, without needing git in its build:

```
ufonet-geo @ https://github.com/WaterCam-Team/Georeferencing/archive/<commit>.tar.gz
```

- SU-WaterCam vendors a copy of the package under `tools/ufonet_geo/`, at a recorded commit (the nodes do not fetch GitHub at run time). Its README there describes how to refresh the copy from a commit on this repo's main.

Rule: a change to `ufonet_geo/` must keep both consumers working. Before merging, check that the API's tests (including its recorded golden outputs) and SU-WaterCam's tests still pass against the new commit, then update the API's pin and refresh SU-WaterCam's copy.

Every file, with whether it is a library, core tool, helper or experiment: [module map](LLM_CONTEXT.md#module-map).

Keep local data (captures, DEMs, point clouds, site notes) in the gitignored `private/` folder, not the repo root: this repo is public.

## Dependencies

```bash
pip install -r requirements.txt
```

Core: `opencv-python`, `numpy`, `pyproj`, `Pillow`, `piexif`, `scipy`.  
Optional: `rasterio` (DEM), `laspy` (LAS/LAZ), `gdal` (GeoTIFF export), `torch`/`sam2`/`segment-anything` (SHARP / SAM in `georeference_sharp.py`, `flood_map.py`).
