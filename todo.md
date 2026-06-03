High priority — do before next field session

1. IMU magnetometer calibration (§7.7 Step 1) — figure-8 procedure with
bno055_calibration.py until BNO055 status = 3. Not yet done on any node. Biggest
single impact on heading accuracy.
2. ~~ArUco GCP validation experiment (§7.1)~~ DONE 2026-06-01. First empirical
residual table produced: RMSE 0.86 m (evening session, markers 1 + 3, 13 m range).
Result is ground-truth-limited (Pix4DCatch marker std ~0.8 m). To get below 0.5 m:
RTK-survey the marker positions directly (30-second viDoc occupation per marker).
3. ~~Heading validation (§7.7 Step 2)~~ DONE 2026-06-01. BNO055 has 180° mounting
offset — reports ~265° when true heading is ~75°. Fix: add `heading = (raw + 180) % 360`
to firmware before writing EXIF. Corrected heading written to unit_config_UFO006.json.
Camera position also surveyed (RTK, 2 cm H) and written to unit_config.
4. Fix BNO055 180° firmware heading offset — add `(raw_yaw + 180) % 360` correction
in the orientation logger before EXIF write. Without this every image has yaw=0
(IMU not contributing). After fix, do figure-8 calibration (item 1).
5. Secure camera mount — heading shifted 5.4° overnight between 16:10 and 19:55
sessions. A rigid clamp (not friction fit) and a quick ArUco check at session
start will catch physical disturbance before data collection.

High priority — desk work

6. DSM source comparison (§7.2) — georeference same flood mask boundary with
Pix4DCatch 5 cm DSM vs USGS 1 m DEM; measure boundary displacement. Answers
whether the field photogrammetry effort is necessary.

Medium priority

7. Site survey for remaining nodes (§7.6) — viDoc RTK occupation, mount height
tape measure, Pix4DCatch scan, unit_config_<node>.json for every deployed node
beyond UFO-006.
8. GCP-based pose refinement (§7.3) — get 4+ ArUco markers in frame per image
to enable refine_pose_from_gcps (needs ≥3 for solver); currently limited to 2.
Reposition markers to be more central in the camera's FOV.
9. ~~Flood extent GeoJSON/GeoTIFF export (§7.4)~~ DONE. scripts/flood_export.py
produces GeoJSON + binary GeoTIFF from a binary flood mask via terrain ray-casting.
10. Thermal-optical registration (§7.5) — SimpleITK baseline transform for FLIR
Lepton → OV5647 alignment, per-node baseline saved to disk.

Low priority

11. BNO085 upgrade assessment (§7.8) — check cost/availability of BNO085 breakout
boards and required firmware changes; 1.0° vs 2.5° RMS heading improvement.
12. Per-node calibration database (§7.9) — config/<node_id>/ directory structure
for unit configs, calibration files, and heading corrections as the network
grows.
