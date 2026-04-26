# Detailed georeferencing process (terrain-first)

This project’s **primary workflow** for real outdoor scenes is **terrain georeferencing**: intersecting camera rays with a **DEM** and/or **rasterized LAS/LAZ** (`georeference_terrain.py`). A horizontal ground plane (`georeference_tool.py`) is a **secondary** option for rare cases or for **optional pose refinement** from GCPs; it is documented briefly at the end.

**Conventions:** WGS84 `lat`/`lon` for camera position unless you transform explicitly. Angles per `camera_geometry.py` (heading 0° = North, pitch **negative** = down). See [GEOREFERENCING_PROCEDURE.md](GEOREFERENCING_PROCEDURE.md) for calibration, IMU, and GCP basics.

**Coarse onboard GPS/IMU:** See [ACCURACY_AND_EXTERNAL_RESOURCES.md](ACCURACY_AND_EXTERNAL_RESOURCES.md) for RTK, ortho GCPs, terrain choice, and combined workflows.

---

## 1. Environment and inputs (terrain)

### 1.1 Software

```bash
pip install -r requirements.txt
```

Terrain mode requires **`rasterio`** (GeoTIFF DEM) and/or **`laspy`** with a LAZ backend for `.laz` files. Core packages (`opencv-python`, `numpy`, `pyproj`, etc.) are always needed.

### 1.2 Artifacts

| Artifact | Role |
|----------|------|
| `calibration.json` | From `camera_calibration.py`: `K`, `D`, optional `camera_height_m`, `img_size`. |
| Field image(s) | Same camera as calibration; resolution can differ if intrinsics are scaled. |
| Camera `lat`, `lon` | EXIF, Vidoc RTK export, or manual entry. |
| Orientation | EXIF UserComment `Roll R Pitch P Yaw Y` and/or IMU CSV (see procedure doc). |
| **Terrain** | GeoTIFF DEM and/or LAS/LAZ with **known or inferable CRS**; inspect with `scripts/dump_dataset_crs.py` or `gdalinfo`. |
| **Vertical datum** | Often **not** fully embedded in rasters — confirm from **data provider docs**; use `--terrain-vertical-datum` / `--camera-elev-datum` when converting EXIF altitude, or prefer **`--height-above-ground`**. |

### 1.3 SU-WaterCam

- Aligns with **WaterCam** usage: EXIF GPS, UserComment orientation when present, checkerboard calibration to `calibration.json`.
- See sibling **SU-WaterCam** project for hardware-specific capture notes.

### 1.4 Vidoc RTK + Pix4Dcatch

- **RTK:** Use as **survey-grade** `lat`/`lon` (and height checks) for the camera or control points.
- **Pix4Dcatch:** Orthomosaic GeoTIFFs can feed **Planet-style GCP scripts** (`--planet-tif`); point clouds can be **LAS/LAZ** inputs for `--las` if CRS and vertical meaning are correct.

---

## 2. Primary workflow — Terrain (`georeference_terrain.py`)

### 2.1 Purpose

Maps each clicked pixel to **lat**, **lon**, **elevation** (terrain surface), and **slant range**, by ray–surface intersection against your DEM/LAS model.

### 2.2 Inspect terrain data

```bash
python scripts/dump_dataset_crs.py --dem your_dem.tif --las your_cloud.laz
```

Record **horizontal CRS**, **nodata**, and any **vertical hints**. If the DEM has no vertical CRS in metadata, use **provider documentation** (e.g. regional orthometric heights).

### 2.3 Camera elevation (recommended)

Prefer **mount height above ground** so camera elevation matches the **same height system as the terrain** without mixing ellipsoid and orthometric incorrectly:

```bash
python georeference_terrain.py photo.jpg \
  --dem your_dem.tif \
  --lat LAT --lon LON \
  --height-above-ground MOUNT_HEIGHT_M \
  --heading H --pitch P --roll R \
  --calibration ./calibration.json
```

Add `--las your.las` and/or `--las-crs EPSG` when using LiDAR. If CRS is missing from the LAS file, **`--las-crs` is required** for correct placement.

- **`--height-above-ground`:** metres from the ground surface under the camera to the camera. The tool sets camera elevation = **terrain elevation at (lat, lon) + this value**.

If you must use absolute **`--elev`** (e.g. from EXIF), set **`--camera-elev-datum`** and **`--terrain-vertical-datum`** per dataset docs. The script prints **diagnostics** (camera vs local terrain, datum conversion warnings).

### 2.4 Interactive session

- **Left-click:** compute point on terrain; note **slant range** and any `[DIAG]` lines.
- **Right-click:** label; **`S`** save CSV; **`Q`** quit.
- Read startup messages: coverage, elevation sanity, datum hints.

### 2.5 Outputs

- Default CSV: `terrain_georeferenced_points.csv` (or `-o` / `--output-csv`) with `label`, `pixel_u`, `pixel_v`, `lat`, `lon`, `elev_m`, `slant_range_m`.

### 2.6 Using Planet / Pix4D ortho for better orientation (optional)

Automatic or manual GCP CSV generation (same GeoTIFF interface):

```bash
python planet_gcp_match.py --field-image YOUR.jpg --planet-tif ortho.tif \
  --output-csv gcp.csv --pixel-space undistorted --calibration ./calibration.json
# or
python planet_gcp_manual.py --field-image YOUR.jpg --planet-tif ortho.tif \
  --output-csv gcp.csv --pixel-space undistorted --calibration ./calibration.json
```

Those GCPs list **pixel ↔ lat/lon** on the reference ortho. They do **not** replace the terrain model. Typical use:

1. Refine **heading / pitch / roll / camera position** using `georeference_tool.py` + loaded GCPs (**R**), *or* adjust orientation from RTK/GCP spreadsheet by hand.
2. Pass the refined angles and position into **`georeference_terrain.py`** for ray–terrain clicks.

On **sloped** ground, the flat-ground model in `georeference_tool.py` is approximate; treat refined pose as **best-effort** for orientation, then trust **terrain** for final elevations.

### 2.7 Quality checks (terrain)

- After startup: **camera minus local terrain** ≈ **mount height** when using `--height-above-ground`.
- Clicks: **slant range** and horizontal distance should match visual distance to features.
- Persistent **no intersection**: wrong **pitch**, **elevation/datum**, or **terrain extent/CRS**.

---

## 3. Optional — Flat ground (`georeference_tool.py`)

Use only when a **horizontal plane** is a deliberate approximation (e.g. quick map overlay, very flat site, or legacy workflow).

```bash
python georeference_tool.py
```

Configure `IMAGE_PATH`, `CALIB_PATH`, position, height above plane, and orientation in the script (or your wrapper). GCP/TPS features are described in [GEOREFERENCING_PROCEDURE.md](GEOREFERENCING_PROCEDURE.md).

---

## 4. Troubleshooting (terrain-focused)

| Symptom | Check |
|---------|--------|
| No terrain intersection | Pitch sign (negative = down); camera elevation vs DEM; `--las-crs`; ray diagnostics |
| Camera “below” terrain at startup | Use `--height-above-ground` or fix absolute elevation / datum |
| Large horizontal error | Heading, `lat`/`lon`, IMU calibration |
| Vertical mismatch | Ellipsoid vs orthometric; PROJ geoid grids; documentation for DEM |
| LAS wrong | CRS; `--las-crs`; extent vs camera |

---

## 5. Minimal end-to-end (terrain)

1. `camera_calibration.py` → `calibration.json`  
2. `scripts/dump_dataset_crs.py` on DEM/LAS  
3. `georeference_terrain.py` with **`--height-above-ground`**, `--dem` / `--las`, pose, calibration  
4. Optional: `planet_gcp_match.py` / `planet_gcp_manual.py` → refine pose in `georeference_tool.py` → re-run terrain with updated `--heading` / `--pitch` / `--roll` / position  

For IMU and checkerboard detail, see [GEOREFERENCING_PROCEDURE.md](GEOREFERENCING_PROCEDURE.md).
