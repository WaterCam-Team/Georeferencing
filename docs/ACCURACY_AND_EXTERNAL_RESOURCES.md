# Improving georeferencing accuracy when sensor GPS and IMU are coarse

**Audience:** Researchers and future maintainers. SU-WaterCam and similar devices often provide **coarse** GNSS position and **coarse** orientation (magnetometer-based heading, consumer IMU pitch/roll). This document explains **what errors dominate**, which **external resources** tighten each part of the solution, and **how this repository** ties them together.

**Related docs:** [GEOREFERENCING_PROCEDURE.md](GEOREFERENCING_PROCEDURE.md) (calibration, IMU basics), [GEOREFERENCING_PROCESS_DETAILED.md](GEOREFERENCING_PROCESS_DETAILED.md) (terrain-first workflow).

---

## 1. What is wrong when GPS/IMU is coarse?

Georeferencing maps **pixel → ray → ground intersection → (lat, lon, elev)**. Errors come from:

| Component | Typical coarse-sensor issue | Effect on output |
|-----------|----------------------------|------------------|
| **Camera position (`lat`, `lon`)** | GNSS error 2–15+ m (phone), multipath | Systematic shift of all rays at the anchor |
| **Camera height** | Noisy EXIF altitude; ellipsoid vs orthometric mix | Large **horizontal** error on oblique rays when intersecting terrain |
| **Heading (yaw)** | Magnetometer bias, declination, nearby metal | Rotates the entire solution in plan |
| **Pitch / roll** | IMU drift, misalignment to optical axis | Wrong ray direction; error grows with distance |
| **Intrinsics (`K`, `D`)** | Uncalibrated or wrong resolution | Distortion and scale errors; worse at image edges |
| **Terrain model** | Coarse DEM, wrong CRS, wrong vertical | Wrong `(lat, lon, elev)` along the ray even if pose were perfect |

**Strategy:** Use **independent constraints** (surveyed points, ortho imagery, better elevation data) to **calibrate** or **replace** the weakest parts of the chain. No single fix solves everything; combine resources that match your site and budget.

---

## 2. Resource tiers (what to add beyond onboard GPS/IMU)

### 2.1 Strongest horizontal ground truth — surveyed control

- **RTK GNSS** on a **tripod** at the camera location (or surveyed mark near the mount).
- **Total station / GNSS rover** on **visible features** you can also identify in the image (corners, paint marks, hydrants).

**Improves:** `lat`, `lon` anchor; optionally height if tied to the same vertical datum as your DEM.

**In this repo:** Enter `--lat` / `--lon` (and height strategy) from the survey instead of trusting EXIF alone. For terrain, prefer **`--height-above-ground`** measured with a tape + **terrain at camera** from DEM so vertical stays consistent.

### 2.2 Orthorectified reference imagery — GCPs for pose

- **Planet** ortho GeoTIFFs (downloaded via `planet_scene_pull.py` or provider workflow).
- **Pix4Dcatch** (or similar) **orthomosaics** with embedded CRS — same role as Planet for **pixel ↔ world** correspondences.

**Improves:** Joint estimate of **heading, pitch, roll**, and sometimes **camera position**, by matching **image features** to **known geographic coordinates** on the ortho.

**In this repo:**

1. Generate GCP CSV with **`planet_gcp_match.py`** (automatic ORB matching) or **`planet_gcp_manual.py`** (click correspondences). Output format matches **`gcp.py`**: `label,pixel_u,pixel_v,lat,lon[,elev_m]`.
2. Load GCPs in **`georeference_tool.py`** and run **pose refinement** (`R`). The flat-ground model is approximate on slopes; use refined angles primarily to **feed `georeference_terrain.py`** (see [GEOREFERENCING_PROCESS_DETAILED.md](GEOREFERENCING_PROCESS_DETAILED.md)).
3. With **6+** well-distributed GCPs, optional **TPS warp** (`W`) can improve **local** pixel→geo mapping in the flat tool; for **elevations**, still rely on **terrain** intersection.

**Limitations:** Seasonal change, shadows, and parallax vs oblique field photos can hurt matching — use manual GCPs or crop/reference to similar conditions.

### 2.3 Better terrain — ray intersection quality

- **LiDAR-derived DEM/DSM** or **high-res regional DEM** vs coarse global products.
- **LAS/LAZ** from mobile mapping or UAV (Pix4D export) as **`--las`** in `georeference_terrain.py` when CRS is correct.

**Improves:** **Elevation** at intersection and horizontal placement along the ray (especially on slopes).

**In this repo:** `georeference_terrain.py` with `--dem` and/or `--las`; **`scripts/dump_dataset_crs.py`** before runs; **`--las-crs`** when LAS lacks embedded CRS.

### 2.4 IMU quality and time sync (reduce coarse-IMU harm)

- **Magnetometer calibration** (figure-8) and **mount alignment** (documented in procedure doc).
- **Timestamp alignment** between image capture and IMU CSV so the correct row is used.

**Improves:** Heading and pitch/roll used in `build_rotation_matrix`.

**In this repo:** `georeference_tool.py` IMU log path + image timestamp; EXIF UserComment from **`add_imu.py`** when logs are not available.

### 2.5 Camera calibration — always worth investment

- Checkerboard **variety** (distance, tilt), **same resolution** as field images, RMS **&lt; 1 px** target.

**Improves:** Pixel→ray direction; reduces need to “fix” bad geometry with GCPs alone.

---

## 3. Recommended combined workflow (coarse GPS/IMU)

Use **every applicable** step; skip only what you cannot obtain.

1. **Calibrate** → `calibration.json`.
2. **Terrain data** → DEM/LAS; verify CRS and vertical meaning (`scripts/dump_dataset_crs.py`, provider docs).
3. **Position:** Prefer **RTK/survey** `lat`/`lon` for the camera when possible; else EXIF as starting point.
4. **Height on terrain:** **`--height-above-ground`** + DEM/LAS in `georeference_terrain.py` (avoids blind trust in EXIF altitude vs orthometric DEM).
5. **Orientation:** Start from IMU/EXIF; then **add GCPs** from Planet or Pix4D ortho → **`georeference_tool.py`** refine pose → **copy refined `heading`/`pitch`/`roll`** (and position if adjusted) into **`georeference_terrain.py`**.
6. **Validate:** Click known visible features; compare to ortho/RTK; check terrain diagnostics (slant range, camera vs terrain elevation).

---

## 4. What each script contributes (quick reference)

| Resource | Script / module | Role |
|----------|------------------|------|
| Survey lat/lon | CLI args to `georeference_terrain.py` | Replace coarse EXIF position |
| Ortho GCPs (auto) | `planet_gcp_match.py` | GCP CSV from field + GeoTIFF |
| Ortho GCPs (manual) | `planet_gcp_manual.py` | Reliable correspondences when auto fails |
| Pose refine from GCPs | `georeference_tool.py` + `gcp.py` | Refine heading/pitch/roll/position (flat model); feed outputs to terrain |
| Terrain intersection | `georeference_terrain.py` | Final lat/lon/elev on DEM/LAS |
| Vertical consistency | `vertical_datum.py`, `--height-above-ground` | Reduce ellipsoid/orthometric mix-ups |
| CRS inspection | `scripts/dump_dataset_crs.py` | Avoid wrong LAS CRS or unknown DEM extent |
| Planet download (optional) | `planet_scene_pull.py` | Fetch scenes by EXIF location (API key required) |

---

## 5. Limitations and honesty

- **GCP pose refinement** in `georeference_tool.py` assumes a **flat ground plane** for its internal residual — on **steep terrain**, residuals are not physically exact; still useful to **reduce gross orientation/position errors** before terrain.
- **TPS warp** changes pixel→geo mapping in the flat tool; it does **not** replace a **3D terrain** model for elevation — use **`georeference_terrain.py`** for z.
- **Ortho GCPs** tie you to **that ortho’s geolocation accuracy** (Planet/Pix4D product dependent).

---

## 6. Checklist before publishing results

- [ ] Calibration RMS acceptable; resolution consistent with field images  
- [ ] Terrain CRS and vertical datum documented or justified  
- [ ] Camera position source documented (EXIF vs RTK vs refined)  
- [ ] If GCPs used: count, spread in image, residual summary  
- [ ] If coarse GPS retained: state expected uncertainty band  

---

*Maintainers: update this file when adding new reference-data workflows or changing GCP/terrain integration.*
