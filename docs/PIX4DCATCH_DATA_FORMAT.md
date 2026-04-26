# Pix4DCatch / viDoc RTK Data Format and Processing on Linux

## Overview

Field scans collected with a **Pix4DCatch** app on an **iPhone 13 Pro** with a **viDoc RTK** rover.
Data lives under `/var/home/manu/UFONet iPhone Data/`, one subdirectory per scan session named by
timestamp (e.g. `2026-04-09-14-10-48`).

There are 17 scan sessions (2024-07-25 through 2026-04-09). Each is also available as a `.zip`
alongside its extracted directory.

---

## Directory Structure (per scan)

```
<timestamp>/
  manifest.json              # top-level asset manifest
  projectManifest.json       # full capture list + depth map refs
  project.opf                # OPF project root (JSON, lists sub-resources)
  projectDirectories.json    # relative paths to each sub-section

  images/
    Image_000001.jpg         # RGB frames (314 per scan for Apr 2026 sessions)
    DepthMap_000001.tiff     # LiDAR depth map per frame (float32, metres)
    Confidence_000001.tiff   # confidence per pixel (0/1/2; 2=high)

  point_clouds/
    legacy/
      pointcloud.gltf        # GLTF 2.0 — geometry description
      pointcloud.bin         # raw binary: interleaved POSITION+COLOR float32
      camera_list.json       # image IDs used during reconstruction
      input_cameras.json     # per-frame GPS + orientation (abs. coords)
      projected_input_cameras.json  # per-frame position in local scene frame
      scene_reference_frame.json    # CRS + local↔UTM transform

  opf_files/
    camera_list.json
    input_cameras.json       # same as above (canonical OPF location)
    projected_input_cameras.json
    scene_reference_frame.json

  meshes/
    mesh.gltf                # textured mesh (GLTF 2.0)
    mesh.bin

  geolocations/
    rtkGPS.csv               # 10 Hz RTK GPS stream (lat/lon/alt + DOP + quality)
    rtkNMEA.csv              # raw NMEA sentences

  metadata/
    catchLog.ndjson
    projectMetadata.json
    thumbnail.png

  reports/
    report.pdf

  region_of_interest/        # present in some scans
  logs/
```

---

## Coordinate Systems

### Scene reference frame (`scene_reference_frame.json`)

```json
{
  "crs": "PROJCRS[\"NAD83(2011) / UTM zone 18N\" ... EPSG:6347]",
  "base_to_canonical": {
    "shift": [-404503, -4768704, -80],
    "scale": [1, 1, 1],
    "swap_xy": false
  }
}
```

- **Canonical** = absolute NAD83(2011) / UTM Zone 18N (EPSG:6347), metres.
- **Base (local)** = scene-centred frame used inside GLTF and `projected_input_cameras.json`.
- **Transform**: `utm_xyz = local_xyz - shift`
  - Equivalently: add `[+404503, +4768704, +80]` to every local coordinate.
- The local origin sits at approximately UTM E 404503, N 4768704, Z 80 m
  (lat ≈ 43.0651°, lon ≈ -76.1730°, Syracuse NY area).

### Vertical datum

`GPSAltitude` in `rtkGPS.csv` is **ellipsoidal height** (NAD83(2011) ellipsoid).
`ReferenceAltitude` column = orthometric height (NAVD88 via EGM96 geoid separation stored in
`GeoidSeparation` column, ~-34.46 m for this area).

The point cloud Z axis follows the ellipsoidal height convention unless Pix4DCatch applies the
geoid correction internally (verify with `vertical_datum.py` if mixing with DEM sources).

---

## GPS / RTK Quality

From `geolocations/rtkGPS.csv` and per-capture sigmas in `input_cameras.json`:

| Field | Typical value (Apr 2026 scans) |
|-------|-------------------------------|
| `qualityIndicator` | `rtkFloat` (not `rtkFixed`) |
| `HAccuracy` | 0.14–0.30 m |
| `VAccuracy` | 0.22–0.40 m |
| Per-frame sigma H (from OPF) | ~0.10 m |
| Per-frame sigma V (from OPF) | ~0.04 m |
| `hdop` / `vdop` | ~0.89–0.94 / ~0.89–0.95 |

`rtkFloat` means carrier-phase ambiguities are not fully resolved. Expect 10–30 cm horizontal
accuracy in practice, better than phone GPS (~3–5 m) but worse than `rtkFixed` (~2 cm).

---

## Point Cloud Binary Format (GLTF)

`pointcloud.bin` stores interleaved `POSITION` + `COLOR_0`, both `VEC3 float32`:

```
Bytes 0–11   : POSITION (x, y, z) — local frame, metres
Bytes 12–23  : COLOR_0  (r, g, b) — normalised [0, 1]
Stride       : 24 bytes
```

To read N points:
```python
import struct, numpy as np
with open("pointcloud.bin", "rb") as f:
    raw = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 6)
xyz_local = raw[:, :3]   # local coords
rgb_norm  = raw[:, 3:]   # 0.0–1.0
```

Apply transform: `xyz_utm = xyz_local + np.array([404503, 4768704, 80])`.
(Read shift from `scene_reference_frame.json`; do not hardcode for other sites.)

### Typical point count (Apr 2026 scans)
~480 000 points covering a ~28 × 27 × 4 m area per session.

---

## Camera Poses (`input_cameras.json`)

Each capture contains:
```json
{
  "geolocation": {
    "coordinates": [lat, lon, ellipsoidal_alt_m],
    "sigmas": [sigma_h_m, sigma_h_m, sigma_v_m],
    "crs": "EPSG:6318 (NAD83(2011) geographic)"
  },
  "orientation": {
    "type": "yaw_pitch_roll",
    "angles_deg": [yaw, pitch, roll],
    "sigmas_deg": [~2, ~1, ~2]
  }
}
```

**Angle convention** (Pix4D): yaw = clockwise from North, pitch = positive up, roll = positive
right-tilt. **Different from this project's convention** (`camera_geometry.py`: yaw 0°=North,
pitch negative=down, roll positive=right-down). Convert before feeding into `georeference_terrain.py`.

---

## Processing Pipeline on Linux

### Dependencies

```bash
pip install pyopf laspy[lazrs] rasterio scipy numpy
# PDAL (optional, for advanced filtering):
# dnf install PDAL  # Fedora
```

### Script: `scripts/pix4d_to_las_dem.py`

Reads one Pix4DCatch scan directory and produces:
1. `<out_dir>/<scan_name>.laz` — point cloud in NAD83(2011) UTM 18N
2. `<out_dir>/<scan_name>_dem.tif` — 5 cm resolution DEM GeoTIFF
3. `<out_dir>/<scan_name>_camera_poses.csv` — per-frame GPS + orientation, GCP-ready

Usage:
```bash
.venv/bin/python scripts/pix4d_to_las_dem.py \
    2026-04-09-14-10-48 \
    --out-dir output/pix4d \
    --dem-res 0.05
```

**`--dsm-method`** controls how the point cloud is gridded:

| Value | Behaviour |
|-------|-----------|
| `linear` (default) | Delaunay interpolation; fills gaps within convex hull of scan |
| `max` | Highest point per cell; no gap-filling; preserves object surface heights |
| `nearest` / `cubic` | scipy griddata alternatives |

Use `max` when you need to detect the top of a sensor or tall object.  Use `linear`
for terrain intersection in `georeference_terrain.py`.

```bash
# True DSM preserving sensor housing height
.venv/bin/python scripts/pix4d_to_las_dem.py 2026-04-24-13-11-52 --dsm-method max
```

Batch all scans:
```bash
for d in "/var/home/manu/UFONet iPhone Data/"/*/; do
    .venv/bin/python scripts/pix4d_to_las_dem.py "$d" --out-dir output/pix4d
done
```

### Script: `scripts/validate_dsm.py`

Validates a generated DSM for CRS, resolution, elevation range, and camera-pose
consistency.  Run immediately after `pix4d_to_las_dem.py`:

```bash
.venv/bin/python scripts/validate_dsm.py output/pix4d/2026-04-24-13-11-52_dem.tif
```

Exit code 0 = all checks passed.  See [DSM_VALIDATION.md](DSM_VALIDATION.md) for
interpretation of each check and georeferencing validation steps.

---

## Integration with `georeference_terrain.py`

1. Run `scripts/validate_dsm.py` to confirm the DSM is sound.
2. Pass `--dem <scan>_dem.tif` as the terrain surface.
3. Use `<scan>_camera_poses.csv` to supply camera position and heading/pitch/roll.
4. Optionally load `<scan>.laz` in `laz-viewer.py` to visually identify GCP targets.

---

## Limitations and Notes

- `rtkFloat` quality: treat horizontal accuracy as ≥10 cm, not cm-level. Acknowledge in GCP
  validation table (EGU Task 1).
- Depth maps are from iPhone 13 Pro LiDAR (range ~5 m indoors / ~10 m outdoors). Point cloud
  is Pix4DCatch's proprietary dense reconstruction, not raw LiDAR.
- Orientation sigmas are ~1–2°. At 10 m range, 1° pitch error ≈ 17 cm ground error.
- Vertical datum mismatch risk: Pix4DCatch GPS altitude is ellipsoidal; FEMA/NHD DEMs use
  NAVD88. Apply geoid correction (`vertical_datum.py`) when combining sources.
- The `projected_input_cameras.json` orientation uses a **different Euler convention** from
  `input_cameras.json`. Use `input_cameras.json` for absolute yaw/pitch/roll.
