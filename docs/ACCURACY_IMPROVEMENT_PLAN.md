# Accuracy Improvement Plan — UFO-Net Georeferencing Pipeline

**Status as of 2026-06-24**  
**Site:** Meadowbrook-006 (UFO-006), Syracuse, NY  
**Current empirical residual:** 0.86 m (2-point sample, ArUco session 2026-06-01, 13.4 m slant range, pose-refined heading 69.48°)  
**Design target:** 5 cm at ground level  
**Immediate binding constraint:** Ground-truth floor — Pix4DCatch marker localization std ≈ 0.7–0.8 m

**Pipeline work completed since 2026-06-12 (no new field data yet):**
- §2.1 BNO055 180° offset — corrected in pipeline via `imu_mount_offset_deg: 180.0` in `unit_config_UFO006.json`; `resolve_heading()` applies mount offset + magnetic declination (−12.5°) + residual correction whenever EXIF IMU yaw is the source. Raw sensor-frame heading preserved in EXIF with `HeadingRawSensorFrame: true`.
- Stable heading averaging — `get_euler_stable()` collects 20 samples at 15 ms intervals, returns circular-mean heading, arithmetic-mean pitch/roll, and Mardia–Jupp circular std dev written to XMP as `HeadingStdDev`.
- XMP quality tags — `CalibSys`, `CalibGyro`, `CalibAccel`, `CalibMag` embedded per image; allows post-hoc filtering of frames captured while magnetometer was uncalibrated.
- Pitch/roll rotation matrix — `resolve_pitch_roll()` applies a 2-D rotation by `imu_mount_offset_deg` to EXIF pitch/roll, correctly handling arbitrary sensor mount angles (not just 0°/180°).
- Two-stage calibration tooling — `bno055_calibration.py` Stage 1 (pre-mount hard-iron offsets) + Stage 2 (post-mount pole-influence validation) now in place; `docs/IMU_CALIBRATION.md` documents the field procedure.

**✅ Fixed (2026-07-15) — EXIF pitch sign convention was inverted.** Confirmed independently on the UFO-006 field photo *and* a controlled RTK-validated backyard test (`imu_mount_offset_deg=0`, no mount-offset formula involved at all): raw EXIF pitch uses the opposite sign from what `camera_geometry.py`/`resolve_pitch_roll()`/`refine_pose_from_gcps()` assume (`0°=level, -90°=down`). As currently written, resolved pitch points the camera *above* the horizon for real ground-facing photos on both units — `validate_georef.py` and `refine_pose_from_gcps()`'s own bounds both independently reject the result. This affects every unit, not just 180°-mount ones, and is upstream of the mount-offset negation formula (§2.1 above is not what's wrong). **Root cause confirmed by code inspection**: SU-WaterCam's capture code (`tools/bno055_imu.py`/`tools/add_metadata.py`) is working as designed — it deliberately writes the BNO055's raw, unmodified hardware Euler pitch straight to EXIF, by documented intent, for this pipeline to correct downstream. The actual gap was that **this repo's own `unit_config.py` `resolve_pitch_roll()`** never added that base sign correction — it only rotated for mount offset, it didn't reconcile the hardware's native pitch polarity with `camera_geometry.py`'s convention. **Fixed** by negating raw EXIF pitch before the existing mount-offset rotation is applied (`unit_config.py`, `resolve_pitch_roll()`); `tests/test_unit_config.py` updated to match; validated against real data — `validate_georef.py` on the UFO-006 Meadowbrook-006 photo now hits 49/49 grid pixels (was 0/49 before the fix). **Definitive root cause confirmed 2026-07-16** against the Bosch datasheet (`Docs/instructions/bst-bno055-ds000.pdf`, Table 3-13 p.32): the BNO055 has two selectable Euler formats, Android vs. Windows, with pitch defined at *opposite sign* between them; `UNIT_SEL` powers on to Android format (0x80) and the vendored `adafruit_bno055` driver never changes it, so the sensor stays opposite of what this pipeline assumes. Roll and Heading/Yaw are explicitly format-independent per the same table — confirmed both by the datasheet and by an empirical roll-isolation test (2026-07-16, no evidence of inversion), so roll is left unchanged. GCP pose refinement checks out with the fix in place (RTK-consistent pose, sub-30cm residuals on cleanly-labeled markers) — see full writeup in `docs/BACKYARD_TEST_2026-07-15.md`.

---

## Overview

The current 0.86 m figure cannot yet be interpreted as the pipeline's true accuracy. It is a 2-point residual, not a proper RMSE, computed after pose refinement to an optimal heading of 69.48° (consistent with the RTK-derived site bearing of 69–75°). The two GCPs used as ground truth were localized via Pix4DCatch depth maps, each with ~0.8 m standard deviation in world position. The pipeline residual and the ground-truth uncertainty are indistinguishable at this noise level.

Correcting the accuracy issues requires work at **three levels** in priority order:

1. **Fix the ground truth** — directly RTK-survey marker positions to eliminate the Pix4DCatch localization floor
2. **Fix known IMU errors** — BNO055 firmware offset, magnetometer calibration, mount stabilization
3. **Expand and use GCPs for pose refinement** — enable the full `refine_pose_from_gcps` path with ≥4 markers

A fourth workstream — **DSM source comparison** — quantifies how much the terrain model choice contributes to total error and is needed to decide whether the Pix4DCatch survey effort is warranted for each future node installation.

**Known measurement gap now closed:** `aruco_gcp.py` was storing WGS84 ellipsoidal altitudes as `elev_m` instead of NAVD88 orthometric heights (geoid undulation N = −34.452 m at this site → 35 m offset). Fixed 2026-06-12. Any GCP CSVs generated before this date have wrong elevation values and should be regenerated. Horizontal (lat/lon) outputs are unaffected.

---

## Level 1 — Fix the Ground Truth (Blocking)

Until this is done, no accuracy number produced by the pipeline is trustworthy, because the error budget is dominated by how well we know where the markers are, not how well the pipeline projects them.

### 1.1 RTK-direct marker position survey

**What:** Occupy each ArUco marker position with the viDoc RTK rover for ≥30 seconds with active NTRIP corrections. Record the averaged (lat, lon, ellipsoidal height) for each marker.

**Why:** Pix4DCatch localizes markers by back-projecting through the phone's depth map and camera pose. Even with RTK-tagged frame positions, depth map noise and the ~30 cm LiDAR range error at 3–5 m propagate into the recovered marker world position. The observed per-marker std of 0.7–0.8 m confirms this. A direct 30-second RTK occupation reduces horizontal uncertainty to 1–3 cm.

**How:**
1. Print new durable markers on PVC or aluminium sheet (15×15 cm minimum). ArUco IDs 1, 3, 6 have field history; add at least two new positions to reach the 4-GCP minimum for pose refinement.
2. Place markers flat on the road surface within the camera's FOV — avoid tilting them, as a tilted marker introduces a height-dependent horizontal offset.
3. Hold the viDoc directly over the centre of each marker; occupy for 30 s minimum; record the RTK-fixed solution (RTK_FIX, not FLOAT). Log horizontal PDOP and number of satellites.
4. Update the GCP CSV: replace the Pix4DCatch-derived lat/lon/elev_m with the RTK values. Tag the source as `rtk_direct` in a new column or filename suffix (e.g. `gcps_rtk_YYYYMMDD.csv`).

**Success criterion:** Per-marker position uncertainty ≤ 3 cm (RTK FIX, PDOP < 2.5). RMSE from the pipeline against these GCPs then reflects pipeline performance, not ground-truth noise.

**Note on 2026-05-29 scan quality:** The Pix4DCatch project log for that session recorded `distanceTraveled: 150.5 m` — implausible for a small backyard. This indicates GPS noise during the scan, which inflates per-frame pose uncertainty and pushes per-marker localization error above the 0.7–0.8 m baseline estimate. GCPs from that session should be treated as lower quality than those from a clean scan.

### 1.2 Re-run residual analysis with RTK ground truth

After Step 1.1, re-run `aruco_gcp.py` detection + the pose-refinement path in `gcp.py` using the RTK-surveyed positions. Produce a new residual table. If RMSE drops significantly from 0.86 m, the prior result was dominated by marker uncertainty. If RMSE remains ~0.8 m or higher, the pipeline has a real systematic error to investigate.

---

## Level 2 — Fix Known IMU Errors

Three independent IMU/mount issues have been identified and confirmed by field data. Each must be corrected before the heading contribution to RMSE can be assessed.

### 2.1 BNO055 180° heading offset ✅ Addressed via pipeline (2026-06-24)

**What:** The BNO055 on UFO-006 reports yaw values that are offset by approximately 180°. During the June 1 evening session, the IMU reported yaw = 0.0° (uninitialized) across all three captures. The morning pose refinement found an optimal heading of 69.48°; the true compass bearing (from RTK survey of two known points) is ~69–75°.

**Root cause confirmed (commit 94cbb5b):** The sensor is mounted rotated 180° from the expected orientation.

**Resolution (pipeline approach, preferred over firmware fix):**
- `unit_config_UFO006.json`: `imu_mount_offset_deg: 180.0`, `imu_magnetic_declination_deg: -12.5`
- `resolve_heading()` applies `(raw_yaw + mount_offset + declination + correction) % 360` automatically whenever EXIF IMU yaw is the heading source
- Raw sensor-frame heading is preserved in EXIF XMP (`HeadingRawSensorFrame: true`) — the correction is applied at processing time, not baked into the image
- `imu_heading_correction_deg: 0.0` in unit config provides a per-node residual field to update after RTK bearing validation (§2.2)

This approach is preferable to a firmware fix because corrections are adjustable without firmware deployment and raw data is preserved for reprocessing.

**Remaining validation:** After §2.2 magnetometer calibration and an RTK bearing check, update `imu_heading_correction_deg` with the observed residual and record the validated heading in the notes field.

### 2.2 BNO055 magnetometer calibration — Tooling complete; field calibration pending

**What:** The full two-stage calibration procedure for the magnetometer has not been completed on UFO-006 or any other node. Without calibration, the BNO055's heading accuracy is 2.5° RMS at best and may be significantly worse near ferromagnetic infrastructure.

**Tooling in place (2026-06-24):** `tools/bno055_calibration.py` implements a two-stage procedure (see `docs/IMU_CALIBRATION.md`):
- **Stage 1** (pre-mount, bench): figure-8 gyroscope/accelerometer/magnetometer calibration; saves hard-iron offsets to `bno055_calibration.json`
- **Stage 2** (post-mount, pole): `--mode mount` validates heading against a known RTK bearing, measures pole ferromagnetic influence, outputs residual for `imu_heading_correction_deg`
- Offsets are automatically reloaded on boot via `_apply_calibration()` — no manual step needed after initial setup

**Field procedure (still required):**
1. Run Stage 1 on the bench before mounting: `python3 bno055_calibration.py`
2. Follow figure-8 instructions until gyro=3, accel≥2, mag=3
3. Mount unit on pole; run Stage 2: `python3 bno055_calibration.py --mode mount --rtk-bearing 70.5`
4. Record the residual output and set `imu_heading_correction_deg` in `unit_config_UFO006.json`
5. Verify: capture a test image and confirm XMP `Yaw` after pipeline correction is within 2° of the RTK-derived bearing

**Expected improvement:** BNO055 specified 2.5° RMS heading after full calibration. At 13 m slant range, 2.5° heading error causes ~57 cm lateral displacement. After calibration + declination correction (already wired in), this drops toward the spec floor.

### 2.3 Camera mount stabilization

**What:** Marker 3's pixel position shifted ~100 px between the morning (16:10) and evening (19:55) sessions on June 1, corresponding to a 5.4° heading shift. The mount is a friction fit and can be disturbed during site visits.

**Fix:**
1. Install a rigid clamp (hose clamp or custom bracket) that prevents rotation without deliberate loosening. The mount should be tool-tight, not hand-tight.
2. Add a quick-check procedure at the start of every field session: photograph the ArUco markers before any other data collection and compare marker pixel positions against the last stable reference. A shift > 5 px in a marker at 13 m slant range indicates a heading change > ~0.1° and should be investigated before proceeding.
3. Record the mount heading to 0.1° precision using the RTK-derived bearing method (not the IMU) after any mount adjustment and update `unit_config_UFO006.json`.

**Impact:** The 5.4° heading shift between sessions is larger than the IMU's 2.5° RMS accuracy. A loose mount makes calibration efforts irrelevant — the heading changes faster than it can be characterized.

---

## Level 3 — Expand GCP Coverage and Enable Pose Refinement

### 3.1 Get ≥4 ArUco markers in frame

**What:** The June 1 sessions detected at most 2 markers (IDs 1 and 3). The `refine_pose_from_gcps` solver in `gcp.py` requires ≥3 GCPs to constrain heading + pitch + roll simultaneously; 4+ provide redundancy and allow residual checking.

**Fix:**
1. Add markers at positions within the camera's FOV that avoid the extreme edges (where residual distortion is highest). Suggested placement: 4 markers arranged roughly in a square within the central 60% of the image, spanning 5–20 m slant range.
2. Avoid placing all markers at the same depth (all at 13 m). Including at least one marker at ~5 m and one at ~20 m provides leverage on both pitch and the camera height estimate.
3. Reposition existing markers 1 and 3 if their current locations are at the image edge or clustered in depth.

**Success criterion:** Consistent detection of ≥4 markers in all three productive captures from a session. This requires stable marker placement (not kicked overnight), adequate lighting, and markers large enough to detect at the farthest range being used.

### 3.2 Run pose refinement with RTK GCPs

After Levels 1 and 2 are complete:
1. Load RTK-surveyed GCPs from Step 1.1 into `georeference_tool.py` via `--gcps gcps_rtk_YYYYMMDD.csv`.
2. Press **R** to run `refine_pose_from_gcps`. Record the optimized heading, pitch, roll, and the per-GCP residuals.
3. If residuals are consistent (no single GCP outlier >3× median), the refined pose is trustworthy. Write the refined angles to `unit_config_UFO006.json`.
4. Re-run the full flood-extent export with the refined pose and report the final GCP RMSE.

**Expected outcome:** With RTK ground truth and ≥4 GCPs, the refined pose should close the gap between the pipeline's geometric accuracy floor (~8 cm from BNO055 heading spec) and the current 0.86 m result.

**Operational dependency:** Achieving 5 cm requires pose refinement to be active at every production deployment, not just during validation. This means ≥4 RTK-surveyed markers must be permanently installed, visible, and detected in every image. If a marker is kicked, obstructed, or fails to detect, the heading falls back to IMU accuracy (~57 cm at 13 m). Plan for marker durability (PVC or aluminium sheet, anchored to pavement) and add a session-start check that verifies ≥4 markers are detected before recording data.

---

## Level 4 — DSM Source Comparison (Desk Work) ✅ Done 2026-07-15 (partial — close-range only)

**What:** Georeference a grid of image points using two terrain models:
- **DSM-A:** Pix4DCatch photogrammetric DSM (`output/pix4d/2026-04-24-13-11-52_dem.tif` — the OPF that actually covers Meadowbrook-006; the `2026-06-01-*` session named in the original plan below covers a different site ~13 km away and was not usable here)
- **DSM-B:** USGS 1 m DEM (`USGS_1M_18_x41y477_NY_FEMAR2_Central_2018_D19.tif` — note: the `x40y477` tile named below does not cover this site; `x41y477` does)

**Why:** The current error budget (georef_workplan.md §4) predicts that DSM choice matters most at road-crown and curb features. Empirically demonstrating the displacement between DSM-A and DSM-B results answers whether the ground-survey effort is necessary for future node installations or whether the national DEM is sufficient given other error sources.

**Result (`scripts/flood_export.py`, Meadowbrook-006/UFO-006 photo, 352-point grid, 2026-07-15):**
mean displacement **0.241 m**, median **0.000 m**, p90 **0.478 m**, max **0.972 m** — all points within the photo's slant range (<10 m, since the mount is only 0.84 m AGL at 33.75° down-tilt). Full detail and interpretation in `docs/DSM_VALIDATION.md` §3. This run was blocked until the EXIF pitch-sign bug was fixed (see `docs/BACKYARD_TEST_2026-07-15.md`) — before the fix, every ray missed the ground on both terrain sources.

**Not yet done:** this photo's footprint doesn't reach the 10–20 m road-crown/curb range the "why" above specifically asks about — that needs a farther/shallower-angle photo from this or another node to actually test the hypothesis. Original steps below kept for reference:

<details>
<summary>Original steps (superseded — see result above)</summary>

1. ~~Process the June 1 scan (`2026-06-01-18-25-23` or another OPF with good coverage) through `scripts/pix4d_to_las_dem.py` to produce `dsm_a.tif`.~~ (wrong site — see note above)
2. Georeference a representative image (or flood mask boundary) twice using `georeference_terrain.py`:
   - Run 1: `--dem dsm_a.tif`
   - Run 2: `--dem USGS_1M_18_x40y477_NY_FEMAR2_Central_2018_D19.tif`
3. Export both outputs as GeoJSON via `scripts/flood_export.py`.
4. In QGIS or geopandas, compute boundary displacement between the two extents at multiple points. Focus on curb-face crossings.
5. Record mean and 90th-percentile displacement. This becomes the "terrain model contribution" row in the error budget table.

</details>

---

## Error Budget: Current vs. Target

| Error source | Current (2026-06-24) | After Level 1 | After Level 2 | After Level 3 | Target |
|---|---|---|---|---|---|
| Ground truth uncertainty | ~0.8 m (Pix4DCatch marker std) | 1–3 cm (RTK direct) | 1–3 cm | 1–3 cm | ≤3 cm |
| Heading error — mount offset | ✅ Corrected in pipeline (+180° via unit_config) | ✅ | ✅ | ✅ | ✅ |
| Heading error — declination | ✅ Corrected in pipeline (−12.5° via unit_config) | ✅ | ✅ | ✅ | ✅ |
| Heading error — mag bias | Unknown (magnetometer uncalibrated) | Unknown | ≤2.5° RMS after cal (~57 cm at 13 m) | ≤2.5° RMS | ≤1° |
| Heading residual correction | 0.0° (not yet validated vs RTK bearing) | 0.0° | Set after §2.2 field cal | ✓ | ✓ |
| Mount stability | 5.4° shift/session (friction fit) | 5.4° shift | <0.1° (rigid clamp) | <0.1° | <0.1° |
| GCP pose refinement | Not enabled (2 GCPs only) | 2 GCPs | 2 GCPs | ≥4 GCPs → pose refined | ≥4 GCPs |
| Node position | RTK-surveyed ✓ (2 cm H) | ✓ | ✓ | ✓ | ✓ |
| Lens distortion | Corrected ✓ (0.278 px RMS cal) | ✓ | ✓ | ✓ | ✓ |
| Terrain model | USGS 1 m DEM — measured contribution 0.24 m mean / 0.48 m p90 at <10 m range (2026-07-15, close-range only; not yet measured at 10-20m/curb range) | USGS 1 m DEM | USGS 1 m DEM | Pix4DCatch DSM | Pix4DCatch DSM |
| IMU data quality metadata | ✅ CalibSys/Mag/HeadingStdDev in XMP | ✅ | ✅ | ✅ | ✅ |

The heading error (2.1 and 2.2) is the binding constraint after ground truth is fixed. At 13 m slant range, 2.5° heading uncertainty = ~57 cm lateral displacement. The IMU spec floor means the 5 cm target cannot be achieved from heading alone without GCP pose refinement or a better IMU (BNO085: 1.0° RMS → ~22 cm, still not 5 cm).

To close to 5 cm, GCP pose refinement (Level 3) with RTK GCPs is required in addition to the IMU fixes. With well-distributed RTK GCPs and a stable mount, the refined pose should reduce the effective heading uncertainty well below 1°.

---

## Recommended Execution Order

### Next field session (required before any desk work is meaningful)

1. Bring viDoc RTK + iPhone, print 4+ ArUco markers on rigid sheets
2. Install rigid clamp on camera mount before doing anything else
3. RTK-survey each marker (≥30 s per marker, record lat/lon/elev_m + PDOP)
4. Photograph markers with camera for ArUco pixel detection
5. Run `bno055_calibration.py` figure-8 magnetometer calibration while unit is mounted
6. Deploy firmware with 180° heading fix (can be done before the session at the desk)

### Desk work (before field session, or immediately after)

1. ~~**First:** Add `heading = (raw_yaw + 180) % 360` correction to orientation firmware (§2.1)~~ ✅ Done via pipeline (unit_config + resolve_heading)
2. ~~**Run DSM comparison using existing data (§Level 4)** — no new field data needed; only remaining desk task~~ ✅ Done 2026-07-15 (close-range only; 10-20m/curb range still open)
3. Update `unit_config_UFO006.json` with `imu_heading_correction_deg` residual after §2.2 field calibration

### After field session

1. Re-run `aruco_gcp.py` + `gcp.py` residual analysis with RTK GCPs
2. Run `refine_pose_from_gcps` with ≥4 RTK GCPs
3. Record refined heading/pitch/roll and per-GCP residuals
4. Update error budget table with new empirical RMSE
5. Evaluate whether the 5 cm target is now achievable or whether BNO085 upgrade is needed

---

## Files to Update as Each Step Completes

| Step | Status | File(s) to update |
|---|---|---|
| Geoid fix | ✅ 2026-06-12 | `aruco_gcp.py` patched; regenerate any GCP CSVs from pre-fix sessions |
| 2.1 Mount offset + declination correction | ✅ 2026-06-24 | `unit_config_UFO006.json` updated; `resolve_heading()` applies corrections |
| Stable heading averaging + XMP quality tags | ✅ 2026-06-24 | `bno055_imu.py`, `add_metadata.py` |
| Pitch/roll rotation matrix | ✅ 2026-06-24 | `unit_config.py` `resolve_pitch_roll()` |
| Two-stage calibration tooling | ✅ 2026-06-24 | `bno055_calibration.py`, `docs/IMU_CALIBRATION.md` |
| 2.2 Magnetometer calibration (field) | ❌ Pending | `bno055_calibration.json` (new, per node); `unit_config_UFO006.json` (`imu_heading_correction_deg`) |
| 2.3 Mount clamp (field) | ❌ Pending | `unit_config_UFO006.json` — add `mount_secured: true` and date in notes |
| 1.1 RTK marker survey (field) | ❌ Pending | New `gcps_rtk_YYYYMMDD.csv` in repo root |
| 3.1–3.2 Pose refinement (field + desk) | ❌ Pending | `unit_config_UFO006.json` — update `heading_deg`, `pitch_deg`, `roll_deg` after refinement |
| Level 4 DSM comparison (desk) | ✅ 2026-07-15 (partial, close-range only) | `docs/DSM_VALIDATION.md` §3 — displacement table added |
| EXIF pitch sign fix (new finding, 2026-07-15) | ✅ 2026-07-15 | `unit_config.py` `resolve_pitch_roll()` (this repo); `tests/test_unit_config.py` updated — see `docs/BACKYARD_TEST_2026-07-15.md` |
| Backyard RTK validation of GCP refinement | ✅ 2026-07-15 (partial — 3/5 markers) | `docs/BACKYARD_TEST_2026-07-15.md` |

---

*This is the authoritative accuracy tracking document. Update the status line and error budget table as each step completes.*
