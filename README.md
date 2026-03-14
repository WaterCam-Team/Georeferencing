# Georeferencing

Camera calibration and pixel→GPS georeferencing for fixed-position and mobile imagery (e.g. flood or field monitoring).

## Quick start

- **Calibrate camera:** `python camera_calibration.py` (checkerboard images → `calibration.json`)
- **Georeference (flat ground):** `python georeference_tool.py` (interactive click-to-GPS)
- **Georeference (terrain):** `python georeference_terrain.py photo.jpg --dem dem.tif --lat ... --lon ... --elev ...`

## Full procedure

**[Step-by-step procedure for researchers](docs/GEOREFERENCING_PROCEDURE.md)** — camera calibration, BNO055/BNO08x IMU calibration, GCP collection, and accurate photo georeferencing.

## Main scripts

| Script | Purpose |
|--------|--------|
| `camera_calibration.py` | Checkerboard calibration → K, D, optional camera height; saves `calibration.json` |
| `camera_geometry.py` | Shared rotation matrix (ENU ↔ camera); used by calibration and georeferencing |
| `georeference_tool.py` | Interactive click-to-GPS (flat ground), batch, optional GeoTIFF; uses calibration + IMU log or EXIF |
| `georeference_terrain.py` | Click-to-GPS with DEM and/or LAS/LAZ terrain intersection |
| `add_imu.py` | Write Roll/Pitch/Yaw to EXIF UserComment for images |

## Dependencies

```bash
pip install -r requirements.txt
```

Core: `opencv-python`, `numpy`, `pyproj`, `Pillow`, `piexif`, `scipy`.  
Optional: `rasterio` (DEM), `laspy` (LAS/LAZ), `gdal` (GeoTIFF export), `python-xmp-toolkit` (XMP IMU in `georeference.py`), `torch`/`sam2`/`segment-anything` (SHARP / SAM in `georeference_sharp.py`, `flood_map.py`).
