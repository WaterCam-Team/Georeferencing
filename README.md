# Georeferencing

Camera calibration and pixel→GPS georeferencing for fixed-position and mobile imagery (e.g. flood or field monitoring).

## Quick start

- **Calibrate camera:** `python camera_calibration.py` (checkerboard images → `calibration.json`)
- **Georeference (terrain — primary):** `python georeference_terrain.py photo.jpg --dem dem.tif --lat ... --lon ... --height-above-ground ...` (see [detailed terrain process](docs/GEOREFERENCING_PROCESS_DETAILED.md))
- **Inspect DEM/LAS metadata (CRS, bounds, vertical hints):** `python scripts/dump_dataset_crs.py --dem your.tif --las your.las` (optional `--las-crs EPSG` if the cloud has no CRS)
- **Georeference (flat ground — optional):** `python georeference_tool.py` (horizontal plane; rarely appropriate for real relief)

## Full procedure

**[Step-by-step procedure for researchers](docs/GEOREFERENCING_PROCEDURE.md)** — camera calibration, BNO055/BNO08x IMU calibration, GCP collection, and accurate photo georeferencing.

**[Detailed process (terrain-first)](docs/GEOREFERENCING_PROCESS_DETAILED.md)** — primary workflow for `georeference_terrain.py`, SU-WaterCam / Vidoc RTK / Pix4Dcatch notes, Planet/Pix4D ortho GCP scripts, and optional flat-ground tool.

**[Accuracy with coarse GPS/IMU](docs/ACCURACY_AND_EXTERNAL_RESOURCES.md)** — how to combine RTK/survey, ortho GCPs, terrain quality, calibration, and time sync to improve results.

## Main scripts

| Script | Purpose |
|--------|--------|
| `camera_calibration.py` | Checkerboard calibration → K, D, optional camera height; saves `calibration.json` |
| `camera_geometry.py` | Shared rotation matrix (ENU ↔ camera); used by calibration and georeferencing |
| `georeference_tool.py` | Interactive click-to-GPS (flat ground), batch, optional GeoTIFF; uses calibration + IMU log or EXIF |
| `georeference_terrain.py` | Click-to-GPS with DEM and/or LAS/LAZ terrain intersection |
| `planet_gcp_match.py` | Generate a GCP CSV by matching a field photo to a Planet orthorectified GeoTIFF |
| `planet_gcp_manual.py` | Manual fallback: click corresponding points in field photo and Planet GeoTIFF |
| `scripts/dump_dataset_crs.py` | Print CRS, bounds, nodata, and quick stats for rasters and LAS/LAZ before terrain georeferencing |
| `add_imu.py` | Write Roll/Pitch/Yaw to EXIF UserComment for images |

## Dependencies

```bash
pip install -r requirements.txt
```

Core: `opencv-python`, `numpy`, `pyproj`, `Pillow`, `piexif`, `scipy`.  
Optional: `rasterio` (DEM), `laspy` (LAS/LAZ), `gdal` (GeoTIFF export), `python-xmp-toolkit` (XMP IMU in `georeference.py`), `torch`/`sam2`/`segment-anything` (SHARP / SAM in `georeference_sharp.py`, `flood_map.py`).
