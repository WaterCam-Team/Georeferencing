# Camera pose from a Pix4DCatch scan (no surveyed targets)

`scripts/survey_to_pose.py` needs ArUco targets with RTK-surveyed centres. When a site
visit produced only an **RTK-fixed Pix4DCatch scan** of the camera's view (iPhone +
viDoc, say), the scan itself can stand in for the targets: every scan frame has an RTK
pose and a LiDAR depth map, so any feature the node camera also sees gets a surveyed 3D
position. `scripts/scan_to_pose.py` solves the pose from that, and
`scripts/scan_pose_overlay.py` checks the result before it goes to the API.

## What you need

| Input | Notes |
|---|---|
| Pix4DCatch project folder | `opf_files/`, `images/` (`Image_*`, `DepthMap_*`, `Confidence_*`). RTK fixed for most of it. Walk the area the camera sees, at the distances it sees. |
| One photo from the node | Taken by the node in the pose to solve, full resolution (e.g. the reference photo uploaded on the dashboard at deployment). Daylight, similar season to the scan. |
| The node camera's calibration | `K`, `D`, `img_size` (e.g. `calibration.json`, the stock OV5647 calibration). The photo size must match `img_size`. |
| Optional: an RTK check point | A surveyed point at or near the camera (`CAM`, or the mean of points around the mount) to compare the solved position with. |
| Optional: phone readings | Compass/clinometer on the unit: a check only (see "Lessons"). |

Keep all of it under `private/<visit>/`: it holds site coordinates.

## Steps

1. **Collect** the scan, photo and calibration into `private/<visit>/` (`scan/`, `photo.jpg`).
   An RTK CSV with columns `Name, Latitude, Longitude, Ellipsoidal height` is optional.
2. **Solve:**

   ```bash
   ../.venv-geo-api/bin/python -I scripts/scan_to_pose.py \
       --scan private/<visit>/scan --photo private/<visit>/photo.jpg \
       --calibration calibration.json --unit-id <unit> --out-dir private/<visit> \
       --check-csv private/<visit>/rtk.csv --check-point CAM --date <survey date>
   ```

   About 2-3 minutes for a 640-frame scan at `--frame-step 2`. Writes
   `pose_config_<unit>_<date>.json` (API pose config fields), `scan_pose_report_<unit>_<date>.json`
   and `scan_pose_matches_<unit>_<date>.npz`.
3. **Read the report** (printed and in the JSON). Rough acceptance values:
   - PnP inliers: 100+ (hundreds is normal), reprojection RMS and held-out median below ~4 px
     at 2592x1944;
   - bootstrap spread well under 0.5 deg (this is noise only; see "Accuracy");
   - inlier azimuths and ranges make sense for what the photo shows;
   - check point: the camera within a few tenths of a metre of it horizontally, at a
     sensible height above it;
   - `mount_height_m` (camera above the 20th-percentile ground height 0.3-0.8 m around it)
     plausible for the mount. Tape it on site next time if you can.
4. **Check by eye and by score:**

   ```bash
   ../.venv-geo-api/bin/python -I scripts/scan_pose_overlay.py \
       --scan private/<visit>/scan --photo private/<visit>/photo.jpg \
       --calibration calibration.json --pose private/<visit>/pose_config_<unit>_<date>.json \
       --report private/<visit>/scan_pose_report_<unit>_<date>.json \
       --variant roll=flip --sweep 1 --out private/<visit>/overlay.jpg
   ```

   - In the overlay the right half of the photo is replaced by the warped scan: features
     must run straight across the white split line. Compare with the `roll=flip` panel.
   - The sweep must peak at `+0.0` in all three rows (heading, pitch, roll). A row peaking
     off zero means that angle is off by about that much: find out why before uploading.
   - Match scores: ~0.3 is a good match; ~0 means unrelated.
5. **Upload** as a site survey with the point cloud, site image and DEM:
   [SCAN_SITE_SURVEY.md](SCAN_SITE_SURVEY.md). The pose config's `mount_height_m` is the
   camera above the ground at the pole base; the survey needs the height above the water
   plane or the DEM instead, which `scan_survey_package.py` works out. An open survey's base
   pose, not the device pose config, places the node's captures.

## How it works

`scan_to_pose.py`:

1. SIFT features (CLAHE-equalised) in the node photo (20 000) and in every `--frame-step`-th
   scan frame (8 000 each). Scan frames are read in their **stored** orientation
   (`IMREAD_IGNORE_ORIENTATION`), which the depth maps and OPF intrinsics refer to.
2. Per frame: Lowe ratio test (0.75), then a fundamental-matrix RANSAC (3 px) on
   undistorted photo points; frames with fewer than 20 inliers are skipped.
3. Each surviving scan pixel gets its depth (confidence 2 only, 0.3-8 m) and is placed in
   the scene frame with the frame's OPF pose (camera x right / y up / z back,
   `R = Rx(omega) Ry(phi) Rz(kappa)`, as `pix4d_to_las_dem`). A photo feature matched in
   several frames gets the median position and is dropped if the frames disagree by more
   than 10 cm.
4. Full 6-DoF PnP: EPnP inside RANSAC (6 px, 20 000 iterations), then Levenberg-Marquardt
   on the inliers, with the node's `K`, `D`. **Nothing is held fixed**, so a surveyed
   camera point and phone readings remain independent checks. (With targets on flat
   ground near the camera, survey_to_pose fixes the position because a free fit can trade
   position for angle. Scan points are 3D and usually span depth and height, so the free
   fit is well conditioned; check it against the check point anyway.)
5. Scene coordinates are UTM (`utm = local - base_to_canonical.shift`). The rotation is
   taken to the local ENU of `ufonet_geo` **numerically**, by pushing grid east/north
   offsets through UTM -> lat/lon -> `latlon_to_enu` (`grid_to_enu`), so the meridian
   convergence needs no sign convention. Heading (true north), pitch and roll are then
   read off in the `ufonet_geo.geometry.build_rotation_matrix` convention.
6. `mount_height_m` from the depth cloud (step 3 in "Steps"); a 200-fold half-split
   bootstrap gives the noise of the solve and a held-out reprojection error.

`scan_pose_overlay.py` back-projects the best-matched scan frames at full resolution and
projects them through the **pose config** with `build_rotation_matrix`, `latlon_to_enu`
and `project_to_pixels`: the same model the API uses. The match score is the correlation
of blurred gradient magnitude (structure, not colour: the cameras differ) over the region
the scan covers in the config's view, held fixed for every variant so scores compare.

## Accuracy

- The bootstrap spread (typically 0.1 deg, 1-2 cm) is the noise of the matches only.
- The scan's own registration limits the real accuracy: Pix4DCatch frame orientation
  sigmas are about 1 deg and RTK positions 1-3 cm. Expect about +-1 deg in heading and
  pitch for features a few metres away, better when matches span a wide field of view and
  several depths.
- Matches come only from textured surfaces (banks, walls, rocks, vegetation); water and
  sky give none. Low photo coverage (~20%) is normal for a creek view; widely spread
  matches matter more than many.
- The depth over shallow clear water may be the bed, not the surface: don't take the
  water level from the scan without an RTK waterline point.

## Lessons (Nov 2026)

- **Don't trust a phone compass on the unit.** Lying on the enclosure lid, next to the
  battery, panel and steel pole, it read about 100 deg off. A clinometer on the lid did
  agree with the solved pitch to under 1 deg, and its tilt magnitude with the roll.
- **Grid vs true north.** Pix4DCatch scenes are in UTM. West of a zone's central meridian
  (northern hemisphere) grid north lies west of true north, so true azimuth = grid
  azimuth - |convergence| (about 0.7 deg at 1 deg from the meridian). An early version of
  the solve added it instead (a 1.5 deg error) and only the overlay sweep caught it
  (heading row peaking at -1 to -1.5); `grid_to_enu` and `tests/test_scan_to_pose.py`
  now pin this.
- A splat of the whole sparse depth cloud is too speckled to judge alignment or to score;
  warping a few full-resolution frames is what works.
- An ArUco check on the scan frames (`aruco_gcp`) finds false positives in foliage when
  there are no targets: confirm hits by eye.
