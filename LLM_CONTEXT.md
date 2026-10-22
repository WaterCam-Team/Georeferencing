# Georeferencing — LLM project context

This file summarizes the **Python georeferencing toolkit** in this repository so assistants can navigate code, conventions, and dependencies without re-reading the whole tree.

## Purpose

**Camera calibration** and **pixel → geographic coordinates** for fixed or mobile imagery (e.g. flood/field monitoring). Primary mode: **terrain** (DEM GeoTIFF, LAS/LAZ ray casting). Optional: flat ground (plane intersection) for rare flat-site or legacy workflows.

- Orientation from **IMU logs (CSV)** or **EXIF UserComment**.
- **Ground control points (GCPs)** to refine pose and optional thin-plate spline (TPS) warping in pixel space.
- When onboard **GPS/IMU is coarse**, combine **surveyed positions**, **ortho GCPs** (Planet/Pix4D), and **quality terrain** — see `docs/ACCURACY_AND_EXTERNAL_RESOURCES.md`.

Human-facing workflow: `docs/GEOREFERENCING_PROCEDURE.md` (researcher checklist). Terrain-first process: `docs/GEOREFERENCING_PROCESS_DETAILED.md`. **Accuracy + external resources:** `docs/ACCURACY_AND_EXTERNAL_RESOURCES.md`. Quick commands: `README.md`.

## Runtime

- **Python:** `.python-version` specifies **3.13** (use a matching venv).
- **Install:** `pip install -r requirements.txt`
- **Tests:** `pytest` (see `tests/`).

## Dependencies (summary)

| Tier | Packages |
|------|----------|
| **Core** | `opencv-python`, `numpy`, `pyproj`, `Pillow`, `piexif`, `scipy` |
| **Optional** | `rasterio` (DEM), `laspy[lazrs]` (LAZ), `gdal` (GeoTIFF export) |
| **ML / research** | `torch`, `sam2`, `segment-anything` — used by `georeference_sharp.py`, `flood_map.py` (not required for basic georeferencing) |

## Coordinate and angle conventions (critical)

Defined in `camera_geometry.py` and `docs/GEOREFERENCING_PROCEDURE.md`:

- **World:** ENU — X East, Y North, Z Up.
- **Camera:** X right, Y down, Z forward (into scene).
- **Heading (yaw):** 0° = North, 90° = East.
- **Pitch:** 0° = level; **negative** = looking down.
- **Roll:** 0° = level; positive = right side of camera tilts down.

Shared math for flat ground: `geo_core.py` (pixel ray → ENU → `pyproj` local azimuthal equidistant to lat/lon).

## Data formats

| Artifact | Role |
|----------|------|
| **`calibration.json`** | From `camera_calibration.py`: intrinsics `K`, distortion `D`, `img_size`, RMS, optional `camera_height_m`. |
| **GCP CSV** | `label,pixel_u,pixel_v,lat,lon[,elev_m]` — see `gcp.py`. |
| **IMU CSV** (tools expecting logs) | Timestamped rows with `tilt_deg`, `roll_deg`, `yaw_deg` (and optional `temp_c`) — see procedure doc and `georeference_tool.py`. |
| **EXIF UserComment** | `Roll R Pitch P Yaw Y` (degrees) — used by several scripts including `add_imu.py`. |

## Module map

This is the one complete list of project files; README and the docs link here rather than keeping their own.
**Status:** *library* = imported by other files, no CLI; *core* = supported tool in the main workflow;
*helper* = supported single-purpose utility; *experimental* = research or one-off, may be stale or need heavy deps;
*archived* = do not use.

| File | Status | Responsibility |
|------|--------|----------------|
| `camera_geometry.py` | library | `build_rotation_matrix` — ENU ↔ camera; single source of angle conventions. |
| `geo_core.py` | library | Pixel rays, flat-ground intersection, shared `pixel_to_world_flat`, DEM height helpers. |
| `gcp.py` | library | Load/save GCPs, `refine_pose_from_gcps` / `refine_pose_with_diagnostics` (≥4 GCPs, residuals in metres, leave-one-out flags), TPS warp `fit_tps_warp`. |
| `vertical_datum.py` | library | Ellipsoid vs orthometric (EGM96/EGM2008/NAVD88) checks and `transform_height` — avoids mixing EXIF vs terrain vertical datums. |
| `exif_imu.py` | library | Central EXIF GPS + IMU parsing (used by the georeferencing tools). |
| `unit_config.py` | library | Per-unit sensor configuration loader (mount, heading, pitch/roll resolution); schema in `unit_config.example.json`. |
| `georeference_terrain.py` | core | **Primary** interactive tool: **DEM and/or LAS/LAZ** ray–terrain intersection. |
| `georeference_tool.py` | core | Optional **flat-ground** interactive tool (plane model); also used for GCP pose refine when feeding orientation into terrain. |
| `water_georef_workflow.py` | core | Interactive end-to-end terrain georeferencing and water-boundary export. |
| `validate_georef.py` | core | Accuracy validation for a fixed-mount node photo against GCPs. |
| `camera_calibration.py` | core | Checkerboard calibration, optional live capture, writes `calibration.json`. |
| `aruco_gcp.py` | core | ArUco-marker GCP detection workflow. |
| `label_gcp.py` | core | Manual pixel labelling of ArUco/GCP markers. |
| `planet_gcp_match.py` | core | GCP CSV by feature matching a field photo to a Planet ortho GeoTIFF. |
| `planet_gcp_manual.py` | core | Manual fallback: click correspondences in field photo and Planet GeoTIFF. |
| `drone_gcp_match.py` | core | GCP CSV by feature matching a field photo to a drone orthomosaic. |
| `gcp_from_pix4d.py` | core | Interactive RTK coordinate extraction from Pix4DCatch frames. |
| `calib_filter.py` | helper | Pre-filter calibration images (sharpness, board geometry, diversity). |
| `calib_filter_sensor_tree.py` | helper | Same filter for a tree of per-capture folders (skips LWIR); reuses `calib_filter` helpers. |
| `undistort_tool.py` | helper | Undistortion utilities and batch/interactive viewer. |
| `planet_scene_pull.py` | helper | Fetch PlanetScope scenes around a photo's EXIF location (API key required). |
| `print_aruco_markers.py` | helper | Printable ArUco marker sheets (DICT_4X4_50), output `./markers/`. |
| `add_imu.py` | helper | Write Roll/Pitch/Yaw into EXIF UserComment. |
| `georeference3d.py` | experimental | GPS-anchored 3D scene reconstruction and click-to-GPS. |
| `georeference_sharp.py` | experimental | 3D Gaussian Splat (SHARP) scene viewer (optional heavy deps). |
| `flood_map.py` | experimental | Flood boundary extraction from photos (optional ML: SAM). |
| `reconstruct3d.py` | experimental | Incremental structure from motion. |
| `laz-viewer.py` | experimental | Point cloud viewing helper (hard-coded `lidar.laz`). |
| `syr_lamppost_query.py` | experimental | Query Syracuse open data for candidate mounting poles. |
| `archive/georeference_old.py`, `archive/georeference.py` | archived | Legacy (see `archive/README.md`); use `geo_core` / current tools. |
| `scripts/dump_dataset_crs.py` | helper | CRS, bounds, nodata, vertical hint (raster), CRS + Z range (LAS/LAZ) — use before new sites. |
| `scripts/pix4d_to_las_dem.py` | core | Pix4DCatch OPF scan dir → LAZ + DEM GeoTIFF + camera-pose CSV. See `docs/PIX4DCATCH_DATA_FORMAT.md`. |
| `scripts/pix4d_to_api_json.py` | helper | Export a Pix4DCatch scan to the API dashboard's JSON format. |
| `scripts/pix4d_to_viewer.py` | helper | Self-contained HTML point-cloud viewer for a scan. |
| `scripts/validate_dsm.py` | helper | Quality checks on a DSM from `pix4d_to_las_dem.py` (see `docs/DSM_VALIDATION.md`). |
| `scripts/drone_dsm_ingest.py` | helper | Accept a drone DSM/DEM GeoTIFF from any photogrammetry tool. |
| `scripts/odm_process.py` | helper | Submit drone images to NodeODM and fetch the results. |
| `scripts/flood_export.py` | helper | DSM source comparison for flood mapping. |
| `scripts/flood_extent_export.py` | core | Batch-project binary flood masks to geographic coordinates. |
| `scripts/compare_mount_configs.py` | helper | Compare EXIF pose vs GCP-refined pose across mount configs. |
| `scripts/collect_subdir_jpegs.py` | helper | Flatten JPEGs from a folder tree into one folder. |

Copies elsewhere: SU-WaterCam `tools/` vendors `camera_geometry.py`, `geo_core.py` and a diverged `camera_calibration.py`;
the API ports the math into `app/georef_math.py` and `app/dem_math.py`. A fix here does not reach them yet
(see the shared-package plan in the `documentation` repo, `GEOREF_SHARED_MATH_PLAN.md`).

## Tests

Under `tests/`, one file per module or tool. Run `pytest` from repo root.

## Reference documentation (researchers + LLMs)

| Document | Role |
|----------|------|
| `docs/GEOREFERENCING_PROCEDURE.md` | Step-by-step researcher checklist (calibration, IMU, GCP basics). |
| `docs/GEOREFERENCING_PROCESS_DETAILED.md` | Terrain-first operational workflow. |
| `docs/ACCURACY_AND_EXTERNAL_RESOURCES.md` | Improving accuracy when GPS/IMU is coarse; external resources and script map. |
| `docs/PIX4DCATCH_DATA_FORMAT.md` | OPF format details, coordinate transforms, RTK quality, and Linux processing workflow for viDoc RTK scans. |

## Repository layout note

- **First-party code** is the **root `*.py` files**, `scripts/` (small utilities), `tests/`, and `docs/`.
- **Local data** (captures, DEMs, LAS/LAZ, PDFs, site notes, node configs) lives in the gitignored **`private/`** folder, not the repo root. Tool default outputs (`output/`, `markers/`, `sharp_output/`) are gitignored too.

## Editing guidance for LLMs

- Prefer **`geo_core.py`** + **`camera_geometry.py`** for consistent flat-ground behavior; **`gcp.py`** for GCP/TPS.
- When changing vertical behavior, coordinate with **`vertical_datum.py`** and terrain scripts.
- When improving workflows for **coarse GPS/IMU**, read **`docs/ACCURACY_AND_EXTERNAL_RESOURCES.md`** and keep it in sync with script behavior.
- Keep optional imports (rasterio, laspy, XMP, torch) **gracefully optional** where the existing code already does so.
- Do not add `print()` in contexts that follow project-specific rules (e.g. TickTalkPython SQ constraints) if those files are shared with that environment.

---

*Generated for LLM context. Update when architecture, accuracy strategy, or primary workflows change.*
