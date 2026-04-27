# Georeferencing — LLM project context

This file summarizes the **Python georeferencing toolkit** in this repository so assistants can navigate code, conventions, and dependencies without re-reading the whole tree.

## Purpose

**Camera calibration** and **pixel → geographic coordinates** for fixed or mobile imagery (e.g. flood/field monitoring). Primary mode: **terrain** (DEM GeoTIFF, LAS/LAZ ray casting). Optional: flat ground (plane intersection) for rare flat-site or legacy workflows.

- Orientation from **IMU logs (CSV)** or **EXIF UserComment** (and optional XMP in `georeference.py`).
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
| **Optional** | `rasterio` (DEM), `laspy[lazrs]` (LAZ), `gdal` (GeoTIFF export), `python-xmp-toolkit` (XMP IMU in `georeference.py`) |
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

## Module map (project Python files, repo root)

| File | Responsibility |
|------|----------------|
| `camera_geometry.py` | `build_rotation_matrix` — ENU ↔ camera; single source of angle conventions. |
| `geo_core.py` | Pixel rays, flat-ground intersection, shared `pixel_to_world_flat`, DEM height helpers. |
| `gcp.py` | Load/save GCPs, `refine_pose_from_gcps`, TPS warp `fit_tps_warp`, residuals. |
| `vertical_datum.py` | Ellipsoid vs orthometric (EGM96/EGM2008/NAVD88) checks and `transform_height` — avoids mixing EXIF vs terrain vertical datums. |
| `exif_imu.py` | Central EXIF GPS + IMU parsing (used by `georeference.py` and others). |
| `camera_calibration.py` | Checkerboard calibration, optional live capture, writes `calibration.json`. |
| `undistort_tool.py` | Undistortion utilities and batch/interactive viewer. |
| `calib_filter.py` | Pre-filter calibration images (sharpness, board geometry, diversity). |
| `georeference_terrain.py` | **Primary** interactive tool: **DEM and/or LAS/LAZ** ray–terrain intersection. |
| `georeference_tool.py` | Optional **flat-ground** interactive tool (plane model); also used for GCP pose refine when feeding orientation into terrain. |
| `georeference.py` | Standalone example: EXIF GPS + EXIF/XMP IMU → ground plane. |
| `georeference3d.py` | Extended 3D-oriented workflow (see file for scope). |
| `georeference_old.py` | Legacy; prefer `geo_core` / current tools unless migrating. |
| `georeference_sharp.py` | SHARP / 3D Gaussian–related pipeline (optional heavy deps). |
| `flood_map.py` | Segmentation / mapping helpers (optional ML). |
| `reconstruct3d.py` | 3D reconstruction utilities (see file). |
| `laz-viewer.py` | Point cloud viewing helper. |
| `add_imu.py` | Write Roll/Pitch/Yaw into EXIF UserComment. |
| `calibration-checkerboard.py` | On-device / Pi-oriented checkerboard capture (if used in your workflow). |
| `scripts/dump_dataset_crs.py` | CLI helper: CRS, bounds, nodata, vertical hint (raster), CRS + Z range (LAS/LAZ) — use before new sites. |
| `scripts/pix4d_to_las_dem.py` | Convert Pix4DCatch OPF scan dir → LAZ + DEM GeoTIFF + camera-pose CSV. See `docs/PIX4DCATCH_DATA_FORMAT.md`. |

## Tests

Located under `tests/`: `test_gcp.py`, `test_vertical_datum.py`, `test_flat_ground.py`, `test_pixel_rays.py`, `test_intrinsics_scaling.py`, `test_camera_geometry.py`, `test_terrain_ray.py`, `test_planet_gcp_match.py`. Run `pytest` from repo root.

## Reference documentation (researchers + LLMs)

| Document | Role |
|----------|------|
| `docs/GEOREFERENCING_PROCEDURE.md` | Step-by-step researcher checklist (calibration, IMU, GCP basics). |
| `docs/GEOREFERENCING_PROCESS_DETAILED.md` | Terrain-first operational workflow. |
| `docs/ACCURACY_AND_EXTERNAL_RESOURCES.md` | Improving accuracy when GPS/IMU is coarse; external resources and script map. |
| `docs/PIX4DCATCH_DATA_FORMAT.md` | OPF format details, coordinate transforms, RTK quality, and Linux processing workflow for viDoc RTK scans. |

## Repository layout note

- **First-party code** is the **root `*.py` files**, `scripts/` (small utilities), `tests/`, and `docs/`.
- **`PDAL-2.10.0-src/`** (if present) is **upstream PDAL source** — not part of this project’s application code; do not treat it as the codebase to modify for georeferencing features.
- **Untracked assets** (sample images, DEMs, PDFs, large archives) often appear in the working tree; they are **data**, not library code.

## Editing guidance for LLMs

- Prefer **`geo_core.py`** + **`camera_geometry.py`** for consistent flat-ground behavior; **`gcp.py`** for GCP/TPS.
- When changing vertical behavior, coordinate with **`vertical_datum.py`** and terrain scripts.
- When improving workflows for **coarse GPS/IMU**, read **`docs/ACCURACY_AND_EXTERNAL_RESOURCES.md`** and keep it in sync with script behavior.
- Keep optional imports (rasterio, laspy, XMP, torch) **gracefully optional** where the existing code already does so.
- Do not add `print()` in contexts that follow project-specific rules (e.g. TickTalkPython SQ constraints) if those files are shared with that environment.

---

*Generated for LLM context. Update when architecture, accuracy strategy, or primary workflows change.*
