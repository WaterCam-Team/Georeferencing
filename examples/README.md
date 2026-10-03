# Examples

- `gcps_example.csv`: example GCP file in the format `georeference_terrain.py --gcps`
  reads (`pixel_x,pixel_y,lat,lon`): a 260-point pixel grid projected to the ground.
  The coordinates are a placeholder (40.7128, -74.0060), not a deployment site.

Real GCP files are location data and are gitignored (`*gcps*.csv`). Tools that write
`./gcps.csv` by default (`aruco_gcp.py`, `camera_calibration.py`, `georeference_tool.py`)
leave their output untracked.
