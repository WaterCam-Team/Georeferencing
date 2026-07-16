# Backyard Test Session — 2026-07-15

## Purpose

Before spending more field time on the site-installed accuracy plan (RTK marker
survey, magnetometer calibration, GCP pose refinement — see
`docs/ACCURACY_IMPROVEMENT_PLAN.md`), a controlled backyard test was run with:
a fresh WaterCam camera/IMU unit, 5 ArUco markers placed flat on the ground,
and RTK-surveyed ground truth for each marker (viDoc RTK + NTRIP, not
Pix4DCatch depth-back-projection). Goal: validate the GCP pose-refinement
workflow end-to-end against real RTK truth, and (if the mount happened to be
rotated 180°) get a controlled read on a suspected IMU pitch/roll sign issue.

BNO055 was mounted with +Y forward (matching the camera's lens direction),
+X right, text/Z up — the nominal/0° orientation, i.e. `imu_mount_offset_deg: 0`.
Camera mount height was **less than 6 ft (1.8288 m)** AGL, per direct
measurement. Mount was **not rigid** (no rigid clamp available for this test);
each session's pose was treated independently rather than assuming a
carried-over calibration (see discussion in-session — this is fine for GCP
refinement, which re-derives pose per photo).

## Data

`Backyard 7-15-26/` (Georeferencing repo root):
- `Images/` — 12 capture sessions (`20260715-HHMMSS/`), each with NIR-OFF,
  NIR-ON, Lepton thermal `.pgm`, and a temperature CSV.
- `Pix4d 2026-07-15-15-05-09/`, `Pix4d 2026-07-15-15-08-03/` — two Pix4DCatch
  photogrammetry scans (not used for this analysis).
- `Yard Points/` — the RTK ground control points, collected via PIX4DCatch's
  GCP-collection feature paired with a **viDoc RTK rover** (`viDoc 2100337`),
  NTRIP corrections from `rtn.dot.ny.gov` (NY State DOT RTN), mountpoint
  `near_msm`, **30 s occupation per point**, reported horizontal sigma
  **≈ 1 cm**. This fully satisfies (and exceeds) the accuracy plan's Level 1.1
  target of ≤3 cm. CRS: EPSG:6318 (NAD83(2011), ellipsoidal height).

### RTK ground truth (5 points)

Exact coordinates redacted here (personal backyard location) — see
`Yard Points/input-control-points.json` in the working tree for the actual
values if needed locally.

| point | lat | lon | elev_m (ellipsoidal) |
|---|---|---|---|
| global_1 | [redacted] | [redacted] | 84.89 |
| global_2 | [redacted] | [redacted] | 84.92 |
| global_3 | [redacted] | [redacted] | 84.97 |
| global_4 | [redacted] | [redacted] | 84.91 |
| global_5 | [redacted] | [redacted] | 84.99 |

Points are just named in RTK-collection order (`global_1`..`5`) with no
automatic link to which physical ArUco marker/ID sat where — that mapping had
to be supplied manually (see below). Only `global_1` has a Pix4DCatch
reference photo, and it's at too oblique an angle to read the marker's ArUco
ID off it.

### Marker layout (confirmed by user, session `20260715-151842`)

Left to right in frame: **global_3, global_1, global_5, global_2, global_4**
(global_4 rightmost). **global_5 is closest to the camera** (1.4 m by RTK,
vs. 4–7 m for the others) and is the only one large/clear enough to show a
legible ArUco pattern by eye. **global_5 only appears in the last 3 of the 12
sessions** — the camera was re-angled partway through the test specifically
to bring it into frame; earlier sessions only show global_1–4.

## Automatic ArUco detection — unreliable at this range/marker size

Ran `aruco_gcp.detect_in_photo()` (`DICT_4X4_50`) across all 12 sessions:
only **one** session (`20260715-151826`) detected anything at all — 2 of 5
markers (IDs 17, 37). The session actually used for analysis (`151842`)
detected **zero** markers with default parameters.

Tried relaxing detector parameters (`minMarkerPerimeterRate=0.01`,
wider adaptive-threshold window, subpixel corner refinement) + 2x image
upscaling: recovered up to 4 raw detections, but two were duplicate/spurious
IDs landing in the fence/foliage background (confirmed by checking their
pixel locations) — relaxing sensitivity further trades missed real markers
for false positives in clutter, not a net win. **Conclusion: for markers this
small/distant, automatic detection is not reliable; manual pixel labeling
(via the existing interactive `georeference_terrain.py`/`georeference_tool.py`
click tools, or equivalent) is the practical path**, at least until markers
are printed larger or placed closer.

Manually-estimated pixel coordinates used for session `151842` (undistorted
image space, K from the *repo's existing* `calibration.json` — **not** this
test camera's own intrinsics, since none exist yet for this unit):

| label | pixel_u | pixel_v |
|---|---|---|
| global_3 | 557 | 741 |
| global_1 | 1073 | 562 |
| global_5 | 1462 | 1095 |
| global_2 | 1511 | 593 |
| global_4 | 2326 | 762 |

## Critical finding: EXIF pitch sign convention is inverted

This is the most important result of the session, found while trying to run
`refine_pose_from_gcps()` and independently corroborated twice:

1. **UFO-006 / Meadowbrook-006 field photo** (`imu_mount_offset_deg: 180`):
   `resolve_pitch_roll()` resolves pitch to **+33.75°** (raw EXIF pitch
   -33.75°, negated per the 180° mount-offset formula). `camera_geometry.py`'s
   convention is `0°=level, -90°=straight down` — so +33.75° means pointed
   *above* the horizon. Confirmed independently via the already-existing
   `validate_georef.py` tool: it finds **zero terrain intersections** for
   this exact photo, for the same reason.
2. **This backyard rig** (`imu_mount_offset_deg: 0`, i.e. pure passthrough,
   no correction formula involved at all): raw EXIF pitch is **+21.44°**.
   Feeding that straight into `refine_pose_from_gcps()` fails outright —
   `scipy.optimize.least_squares` rejects it because the function's own
   bounds require `pitch ∈ [-90, 0]` (camera can only look level-to-down,
   per the same convention). With `bounds=False` forced, every single GCP
   ray misses the ground plane (`pixel_to_world_flat` returns `None` for
   all 5 points) — i.e. **the pipeline's own math agrees the pose is
   physically invalid**, independent of the mount-offset formula entirely.
3. **Negating pitch** in both cases (UFO-006 and this rig) immediately
   produces a physically sensible, RTK-consistent solution (see results
   below).

**Conclusion:** raw EXIF pitch, as currently written by the SU-WaterCam
capture pipeline, uses the *opposite* sign convention from what
`camera_geometry.py` (and everything built on it — `resolve_pitch_roll()`,
`refine_pose_from_gcps()`) assumes. This is upstream of, and independent
from, the `imu_mount_offset_deg` 180° negation formula — it affects every
unit, not just 180°-mounted ones. **Fixed 2026-07-16** (see "Open items"
below for the commit).

**Root cause, confirmed by reading the actual code (2026-07-15):** this is
*not* a mistake in SU-WaterCam's capture code. `tools/bno055_imu.py:171`
(`get_euler_stable()`) takes `sensor.euler[2]` — the Adafruit
`adafruit_bno055` library's raw, unscaled read of BNO055 hardware register
`0x1A` (`_euler` property, `adafruit_bno055.py:894-896`), no transformation
applied. `tools/add_metadata.py:125,153` writes that value straight into
EXIF/XMP `Pitch`, and is explicitly documented as doing so on purpose
(comment at lines 154-156: *"Mount offset, magnetic declination, and pole
correction are applied by the Georeferencing pipeline at processing
time"*) — SU-WaterCam's side is deliberately meant to write the raw
sensor-frame value untouched.

**The actual gap is in the Georeferencing repo's `unit_config.py`
`resolve_pitch_roll()`** — it implements a mount-offset *rotation* (for the
sensor being physically rotated 0°/90°/180°/270° around its mounting axis)
but has no step that reconciles the BNO055's native pitch-sign convention
with `camera_geometry.py`'s assumed convention (`0°=level, -90°=down`).
Since the hardware's own polarity is the opposite of what `camera_geometry.py`
assumes, and nothing anywhere corrects for that *base* mismatch, resolved
pitch comes out backward regardless of mount offset — consistent with
seeing the identical problem in both the 180°-offset case (UFO-006) and the
0°-offset case (this backyard rig).

**Roll checked separately (2026-07-16)** — see "Roll sign convention check"
below. Short version: no evidence of the same inversion; left unchanged.

**Definitive root cause confirmed against the Bosch datasheet (2026-07-16)**
— `Docs/instructions/bst-bno055-ds000.pdf` (BST-BNO055-DS000-18, rev1.8):
Table 3-13 "Rotation angle conventions" (p.32) shows the BNO055 has two
selectable Euler output formats, **Android vs. Windows**, and **pitch is
defined with opposite sign between them** ("turning clockwise decreases
values" in Android format vs. "increases values" in Windows format). The
Page-0 register map (~p.56) shows `UNIT_SEL` powers on to **0x80** (bit 7
set = Android format). The vendored `adafruit_bno055` CircuitPython driver
never writes `UNIT_SEL` (confirmed via `grep` — no references anywhere in
the library), so the sensor stays in its power-on-reset Android format —
the opposite of what `camera_geometry.py` assumes. This is the actual,
documented mechanism behind the bug, not just an empirically-patched
symptom. **Roll and Heading/Yaw are explicitly format-independent** per
the same table (identical convention in both Android and Windows) — this
is why only pitch needed correction, and it's why the roll-isolation test
below found no inversion: the datasheet says there shouldn't be one.

## GCP pose-refinement results (pitch sign corrected)

Camera EXIF: lat/lon redacted (personal backyard location — see the
photo's own EXIF locally if needed), alt=128.0 m (ellipsoidal, GPS — noisy,
not used directly for height),
raw pitch=+21.4375° (negated to -21.4375° for x0), roll=1.4375°,
yaw=169.75°.

**Unconstrained height** (bounds: lon/lat ±0.002°, height 0.1–200 m,
heading any, pitch [-90,0], roll ±180): converges to heading=105.54°,
pitch=-20.29°, roll=-4.23°, **height=2.29 m** — but the user confirmed
actual mount height was **< 6 ft (1.83 m)**, so this solution over-fits by
pushing height too high to compensate for other errors (mismatched
calibration, imprecise pixel labels).

**Height bounded to ≤ 1.8288 m (6 ft)**, re-run with a custom `least_squares`
call (same residual function as `refine_pose_from_gcps`, tighter height
upper bound): converges to **heading=88–90°, pitch≈-16.3° to -16.4°,
roll≈-0.6° to -0.9°, height≈1.50–1.56 m** — physically consistent with the
stated mount height.

Per-marker residuals (refined pose vs. RTK truth), before vs. after a
precise re-estimate of `global_1`/`global_2`'s pixel coordinates (cropped
4x-zoomed re-inspection — see below):

| marker | v1 residual | v2 residual (after re-estimate) |
|---|---|---|
| global_3 | 0.032 m | 0.100 m |
| global_4 | 0.193 m | 0.274 m |
| global_5 | 0.327 m | 0.225 m |
| global_1 | 1.579 m | 1.737 m |
| global_2 | 1.853 m | 1.673 m |

**Re-estimating `global_1`/`global_2`'s pixel coordinates did not meaningfully
change their residuals** (both moved by only a few pixels from the original
estimate) — ruling out pixel-labeling imprecision as the cause. Three markers
consistently fit to 10–30 cm; these same two consistently sit around 1.7 m
regardless. Most likely cause: the borrowed `calibration.json` (a different
camera unit's intrinsics/distortion model — this test rig has no calibration
of its own yet), since lens-distortion residual error is position-dependent
across the frame and could plausibly hit these two markers harder than the
other three. Non-flat ground at those two specific spots is a much less
likely secondary possibility.

**Net conclusion:** once the pitch-sign issue is corrected, the core
GCP-refinement math (`gcp.py`, `geo_core.py`, `camera_geometry.py`) checks
out — it recovers a plausible, RTK-consistent, mount-height-consistent pose,
and sub-meter (often sub-30cm) accuracy on markers with reliable pixel
labels.

## Tooling added this session (Georeferencing repo)

- `scripts/flood_export.py` — Level 4 DSM-source-comparison tool (grid-based
  ray-cast comparison between the Pix4DCatch DSM and the USGS 1m DEM for the
  Meadowbrook-006 site). **Blocked**: the Meadowbrook-006 photo hits the same
  pitch-sign bug (item 1 above) — zero grid points intersect either terrain
  source. Correct site-matched inputs were identified for when this unblocks:
  image `Meadowbrook-006/20260426-090402-NIR-OFF.jpg`, DSM-A
  `output/pix4d/2026-04-24-13-11-52_dem.tif`, DSM-B
  `USGS_1M_18_x41y477_NY_FEMAR2_Central_2018_D19.tif` (not the `2026-06-01-*`
  session/`x40y477` tile the accuracy plan's example names — those cover a
  different, ~13 km-distant site).
- `scripts/compare_mount_configs.py` — designed to compare auto-corrected
  EXIF pose vs. GCP-refined pose across mount configurations (e.g. 0° vs.
  180° `imu_mount_offset_deg`), built on `aruco_gcp.detect_in_photo`/
  `write_gcp_csv` for marker detection. Not actually used for the analysis
  above, since automatic detection failed on this dataset — the ad hoc
  manual-CSV workflow above was used instead. The script itself is unchanged
  and should still work once markers are reliably auto-detectable (larger
  markers, closer range, or once someone points it at a session where
  detection succeeds).

## Roll sign convention check (2026-07-16, desk-only, no new field data)

Tested whether roll has the same sign inversion as pitch, using the existing
backyard RTK data: held heading/pitch/height/camera lat-lon fixed at the
already-validated 5-GCP refined optimum (heading=88.11°, pitch=-16.37°,
height=1.561m), then swept *only* roll and scored against the 3 reliably-
labeled markers (`global_3`, `global_4`, `global_5`).

| roll hypothesis | mean residual (3 reliable markers) |
|---|---|
| Best-fit (free sweep, -15° to 15°) | +0.20° → 0.167 m |
| Auto/unchanged (+1.4375°, current code) | 0.255 m |
| Negated (-1.4375°, pitch-style fix) | 0.289 m |

**Result: roll does not show the same inversion as pitch.** The best-fit
value sits closer to the raw/unchanged convention than to the negated one,
and unchanged fits slightly better than negated. **No code change made** —
current roll handling in `resolve_pitch_roll()` is left as-is.

**Caveat (empirical test alone):** this is a much weaker test than the one
that caught the pitch bug. Pitch's error was categorical (wrong sign meant
no ray could reach the ground at all); roll's actual value here is small
(~1.4°), so its effect on projected position is subtle, and this test (3
noisy points, borrowed camera calibration) only has power to rule out a
*large* inversion, not a smaller one hiding in the noise.

**Superseded by the datasheet finding above:** the Bosch datasheet
confirms roll's convention is identical in both Android and Windows
formats — the same mechanism that inverted pitch categorically does not
apply to roll. Combined with this empirical test finding no inversion
either, roll can be considered settled without needing the deliberate
large-roll field session originally proposed here — though one would still
be a reasonable sanity check if a backyard session happens anyway.

## Open items / next steps

1. ✅ **Fixed** — added a base sign correction (negate raw EXIF pitch before
   the existing mount-offset rotation) in `unit_config.py`
   `resolve_pitch_roll()`. `tests/test_unit_config.py` updated (4 tests'
   expected values changed; all 19 pass). Validated against real data:
   `validate_georef.py` on the UFO-006 Meadowbrook-006 photo now hits
   49/49 grid pixels (footprint ~3×3 m, slant range 1.2–4.8 m — physically
   sensible for the 0.8 m mount height), vs. 0/49 before the fix. Committed
   `a5e930a`, pushed to `origin/main`.
1b. ✅ **Roll checked** (2026-07-16) — see "Roll sign convention check"
   above. No evidence of the same inversion at the small (~1.4°) magnitude
   tested; left unchanged. Not fully conclusive — see caveat above.
2. Get a real `camera_calibration.json` for this backyard test camera instead
   of borrowing another unit's intrinsics — likely the biggest lever left to
   shrink the `global_1`/`global_2` residuals.
3. Precisely re-label all 5 markers (not just 2) once real intrinsics exist,
   to get a clean, fully-trustworthy residual table.
4. Decide whether to capture an additional session with the BNO055 rotated
   180° (Y pointing back toward the mount) on this same rig, now that the
   base sign convention is understood — would directly validate/invalidate
   the `imu_mount_offset_deg=180` negation formula against RTK truth, the
   same way this session validated the 0° case. Also worth deliberately
   testing a large (~20-30°) roll tilt at the same time, for a decisive
   read on item 1b.
5. ✅ Level 4 DSM comparison (`scripts/flood_export.py`) run on
   Meadowbrook-006 (2026-07-16) — see `docs/DSM_VALIDATION.md` §3. Still
   open: mid/far range (10-20m, curb) not yet tested.

## Unrelated fixes from this session (SU-WaterCam repo, already committed)

For continuity — three unrelated production bugs were found and fixed
earlier in this session, in `SU-WaterCam` (all committed and pushed):
- `5c55e22` — SimpleITK image-dimensionality bug in
  `tools/coreg_multiple.py`'s cached-transform reuse path.
- `2a64c9e` — 10 convenience functions in `tools/lora_handler_concurrent.py`
  not honoring `get_lora_handler()`'s documented `None`-return contract.
- `7d9ec14` — same class of bug at `ticktalk_main.py:335`
  (`handler.is_joined()` unguarded against `None`).
