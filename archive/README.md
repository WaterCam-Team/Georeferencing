# Archive

Scripts kept for reference only. Nothing in the repo imports them and they are
not tested or maintained.

| File | What it was | Replaced by |
|---|---|---|
| `georeference_old.py` | Earlier flat-ground click-to-GPS tool (calibration + IMU, optional GeoTIFF export). | `georeference_tool.py` (flat ground, shares `geo_core`) and `georeference_terrain.py` (DEM/LAS, primary). |
| `georeference.py` | Single-image example: camera GPS from EXIF, roll/pitch/yaw from EXIF or XMP, one pixel projected onto a flat ground plane. Needs `python-xmp-toolkit`. | `georeference_tool.py` and `georeference_terrain.py`. |
