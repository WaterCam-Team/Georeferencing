# DSM Validation and Georeferencing Quality Assessment

Covers two distinct quality checks:
1. **DSM validation** — is the generated surface model geometrically correct?
2. **Georeferencing validation** — does the pixel → (lat, lon, elev) mapping match
   ground truth?

Related docs: [PIX4DCATCH_DATA_FORMAT.md](PIX4DCATCH_DATA_FORMAT.md),
[ACCURACY_AND_EXTERNAL_RESOURCES.md](ACCURACY_AND_EXTERNAL_RESOURCES.md),
[GEOREFERENCING_PROCESS_DETAILED.md](GEOREFERENCING_PROCESS_DETAILED.md)

---

## 1. DSM Validation

### 1.1 Automated script

After running `scripts/pix4d_to_las_dem.py`, validate the output with:

```bash
.venv/bin/python scripts/validate_dsm.py output/pix4d/2026-04-24-13-11-52_dem.tif
```

The script infers the camera poses CSV from the DSM filename.  Supply it
explicitly with `--poses` if needed.

**Example output:**

```
=== DSM Quality Report: 2026-04-24-13-11-52_dem.tif ===

Geometry
  PASS  CRS = EPSG:6347  —  got EPSG:6347
  PASS  Resolution = 0.050 m  —  got 0.0500 × 0.0500 m
        Extent:     81.4 × 84.4 m  (1628 × 1688 px)

Elevation (ellipsoidal)
  PASS  z_min > 100 m  —  got 109.10 m  (datum error if outside expected range)
  PASS  z_max < 125 m  —  got 112.79 m
        Z range:    109.10 – 112.79 m  (mean 111.03 ± 0.61 m)

Coverage
  PASS  Has valid cells  —  755,180 / 2,748,064 cells (27.5% of bounding box)
        Coverage:   27.5% of bounding-box  (remainder is edge padding + scan gaps)

Camera poses
  PASS  All cameras above DSM surface  —  1020 poses checked, 974 outside extent
  PASS  Median height above ground 0.5–10 m  —  4.04 m  (range 2.84–5.97 m)

Overall: PASS  (all checks passed)
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--expected-res` | `0.05` | Expected grid resolution (m) |
| `--z-min` | `100` | Minimum plausible ellipsoidal elevation (m) — adjust for other sites |
| `--z-max` | `125` | Maximum plausible ellipsoidal elevation (m) |

Exit code is 0 on PASS, 1 on any FAIL.

### 1.2 What each check means

| Check | What failure indicates |
|-------|----------------------|
| CRS = EPSG:6347 | `scene_reference_frame.json` CRS not written correctly, or rasterio did not embed it |
| Resolution = 0.05 m | `--dem-res` argument mismatch, or raster written with wrong transform |
| z_min > 100 m | UTM shift applied with wrong sign; large offset (e.g. z ≈ −80) means `−shift` was used instead of `local − shift` |
| z_max < 125 m | Same shift error in the opposite direction, or a stray outlier point |
| Has valid cells | Pipeline failed silently; point cloud was empty |
| All cameras above DSM | Coordinate transform or datum broken — if cameras are ~34 m underground, ellipsoidal/NAVD88 datum was swapped |
| Median height 0.5–10 m | Confirms the scan was collected close to the surface; a ~34 m offset is the geoid separation (datum mix-up); a large negative value is a shift sign error |

### 1.3 Automated tests

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_dsm_validation.py -v
```

Unit tests run without any data on disk.  Integration tests
(`TestDSMProperties`, `TestCameraPosesAboveDSM`) auto-skip when the DSM is
absent.  Generate the DSM first if needed:

```bash
.venv/bin/python scripts/pix4d_to_las_dem.py 2026-04-24-13-11-52
```

### 1.4 max-Z vs linear-interpolated DSM

`pix4d_to_las_dem.py` supports two gridding methods (set with `--dsm-method`):

| Method | Behaviour | Use when |
|--------|-----------|----------|
| `linear` (default) | Delaunay triangulation; fills gaps within convex hull | General-purpose; good for terrain intersection |
| `max` | Highest point in each cell; no gap-filling | True DSM preserving object surfaces; sensor height analysis |

**Key difference:** `linear` smooths spike features (tall objects, sensor housings) by
blending with surrounding points.  `max` records the peak exactly.  For a sensor installed
on a surface you want to detect the top of the sensor — use `max`.

The test `TestMaxVsLinearSynthetic::test_max_captures_spike_linear_smooths_it` demonstrates
this numerically: on a flat 100 m cloud with one 110 m spike, `max` returns 110.00 m in the
spike cell while `linear` returns ≤109.5 m.

### 1.5 Visual inspection in QGIS

1. Load `<scan>_dem.tif` (DSM) and `BarryPark-FEMA-1M-DEM-18TVN080640.tif` (reference).
2. Apply a hillshade render to both (Layer Properties → Symbology → Hillshade).
3. Check that linear features (kerbs, wall edges, drainage channels) align between
   the two layers.  Systematic offset in X or Y indicates a CRS or shift error.
4. Spot-check z values with the Identify tool at recognisable features.  At this site
   the DSM uses ellipsoidal height; the FEMA DEM uses NAVD88 orthometric.  The
   expected separation is ≈34 m (geoid height at Syracuse NY); confirm with
   `vertical_datum.py`.

---

## 2. Georeferencing Validation

Georeferencing maps a pixel (u, v) in a field photo to a geographic coordinate
(lat, lon, elev) by intersecting the camera ray with the DSM.  Errors come from:
camera position, orientation (heading/pitch/roll), intrinsics (K, D), and the DSM.

### 2.1 Check-point method (recommended)

This is the standard geodetic approach: use independent points whose ground-truth
position is known but were **not** used during pose refinement.

**Step 1 — Collect check points**

Pick 5–8 features that are:
- Clearly visible in the field photo (corner of a drain grate, paint mark, kerb corner)
- Identifiable in an orthorectified reference (Planet ortho from `planet_scene_pull.py`,
  or Pix4DCatch orthomosaic)
- Spread across the image, not clustered in one corner

**Step 2 — Record pixel and ground-truth coordinates**

Use `planet_gcp_manual.py` (interactive click) or `planet_gcp_match.py` (automatic ORB) to
produce a GCP CSV:

```
label, pixel_u, pixel_v, lat, lon [, elev_m]
```

**Step 3 — Split into refinement GCPs and check points**

- Use 4–6 GCPs for pose refinement in `georeference_tool.py`
- Hold back 2–3 as check points (do not use them for optimisation)

**Step 4 — Refine pose and run terrain intersection**

```bash
python georeference_tool.py   # load image + GCPs → refine R → note heading/pitch/roll
python georeference_terrain.py --dem <scan>_dem.tif \
    --lat <cam_lat> --lon <cam_lon> \
    --heading <h> --pitch <p> --roll <r> \
    --height-above-ground <measured_m>
```

In the interactive viewer, click each check-point pixel and record the reported
`(lat, lon, elev)`.

**Step 5 — Compute residuals**

Compare the terrain-intersection output to the known check-point positions:

```
horizontal_error_m = haversine(predicted_lat, predicted_lon,
                               truth_lat, truth_lon)
vertical_error_m   = predicted_elev - truth_elev
```

Report: per-point error, RMSE, max error.

**Acceptable thresholds (rtkFloat accuracy):**
- Horizontal RMSE < 0.3 m (consistent with 0.15 m RTK sigma + orientation uncertainty)
- Vertical error < 0.5 m (dominated by geoid/datum and DSM accuracy)

### 2.2 GCP residuals in `georeference_tool.py`

`georeference_tool.py` calls `gcp_residuals()` from `gcp.py` and prints
`rms_m_approx` during pose refinement.  This measures reprojection error on the
**flat-ground model** used internally — it is a proxy for orientation quality, not a
direct check of the terrain-intersection output.

Use it to catch gross errors (rms > 1 m suggests a bad initial heading or wrong GCP),
but always follow up with the check-point method above for an end-to-end validation.

### 2.3 Slant-range sanity check

`georeference_terrain.py` reports `slant_range_m` for every click.  Compare to the
known camera-to-target distance (measured with a tape or from the site survey).  A
factor-of-2 discrepancy usually means a wrong camera height or a datum mismatch.

### 2.4 Visual overlay

Reproject the georeferenced photo onto the Planet ortho in QGIS:
1. Export several clicked points as a GeoJSON.
2. Open both layers; confirmed GCPs should coincide with the corresponding ortho features.
3. Residual scatter gives a visual sense of whether errors are systematic (offset/rotation)
   or random.

Systematic offset → camera position error (fix `--lat`/`--lon`).
Systematic rotation → heading error (re-run pose refinement).
Random scatter → orientation noise + DSM resolution limit.

---

## 3. Summary Checklist

### DSM
- [ ] `validate_dsm.py` reports all PASS
- [ ] CRS confirmed EPSG:6347
- [ ] Elevation range physically plausible (ellipsoidal)
- [ ] Camera poses are above DSM surface with median height 0.5–10 m
- [ ] Hillshade aligns with reference DEM (QGIS visual check)

### Georeferencing
- [ ] At least 4 refinement GCPs with `georeference_tool.py` rms < 0.5 m
- [ ] 2+ independent check points with horizontal error < 0.3 m
- [ ] Slant range matches known camera-to-target distance
- [ ] No systematic offset or rotation visible in QGIS overlay
