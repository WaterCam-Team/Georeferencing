# Site survey from a Pix4DCatch scan: capture, DEM, checks, upload

End-to-end procedure for a node (re)deployment where the site is recorded with an
RTK-fixed **Pix4DCatch scan** (iPhone LiDAR + RTK receiver such as a viDoc) instead of
surveyed targets. It produces everything the API needs to place the node's water maps:

| Product | Made by | Goes to the API as |
|---|---|---|
| camera pose (heading, pitch, roll, position) | `scripts/scan_to_pose.py` | survey `base_pose` + device pose config |
| 3D point cloud of the site | `scripts/scan_survey_package.py` | `PUT /pointcloud/<session>` (via the survey import) |
| top-view site image | `scripts/scan_survey_package.py` | survey site image (map layer) |
| bare-ground DEM | `scripts/scan_survey_package.py` | `POST /dems` (dashboard Georeference -> DEMs) |
| survey manifest | `scripts/scan_survey_package.py` | `API/scripts/import_survey.py` |

Everything with site coordinates stays under `private/<visit>/` (git-ignored); the scripts
refuse to write anywhere else.

## 1. Site capture (in the field)

1. **Finish the mount first.** Aim, tighten and close the box before scanning: the survey
   is valid from the moment the mount is final. Note that time (with time zone).
2. **Node reference photo.** Have the node take a full-resolution photo and upload it as
   the reference photo on the dashboard (it needs network at the site). This photo is what
   the pose is solved from. Daylight, nothing in front of the lens.
3. **Pix4DCatch scan, RTK fixed** for nearly all of it:
   - walk the whole area the camera sees, at the distances it sees (both banks, the water's
     edge, the far bank face), and the ground around the mount;
   - keep the phone 0.5-4 m from surfaces: depth beyond ~4 m is noisy and is not used;
   - water itself gives few or wrong returns (often the bed): scan the edges well.
4. **RTK points** (same receiver): a few around the mount (or one at the lens) as a check
   on the solved position, and **a waterline point** (on the water's edge, at the surface),
   which sets the water height better than the scan can (`--water-height`).
5. **Tape the mount height** (lens above ground) if you can: a check on the solved one.
6. Phone compass on the enclosure: don't bother, it reads far off next to the battery, panel
   and pole. A clinometer on the lid is a useful pitch check.

Copy to `private/<visit>/`: `scan/` (the whole Pix4DCatch project), `photo.jpg` (the
reference photo, downloaded from the dashboard), `rtk.csv` (`Name, Latitude, Longitude,
Ellipsoidal height`), and notes.

## 2. Camera pose

Follow [SCAN_TO_POSE.md](SCAN_TO_POSE.md): `scan_to_pose.py`, then `scan_pose_overlay.py`
(the sweep must peak at 0 in all three rows). It writes `pose_config_<unit>_<date>.json` and
`scan_pose_report_<unit>_<date>.json`, both used below. Don't upload the pose config on its
own yet; step 5 does it with the right ground model.

## 3. Point cloud, site image, DEM, manifest

```bash
../.venv-geo-api/bin/python -I scripts/scan_survey_package.py \
    --scan private/<visit>/scan --photo private/<visit>/photo.jpg --calibration calibration.json \
    --pose private/<visit>/pose_config_<unit>_<date>.json \
    --report private/<visit>/scan_pose_report_<unit>_<date>.json \
    --device-id <api device id> --unit-id <unit> --date <date> \
    --valid-from <YYYY-MM-DDTHH:MM:SS-05:00> --name "<site>, from <date> (RTK site scan)" \
    --session-id <site>-<unit>-scan-<date> --out-dir private/<visit> \
    [--water-height <orthometric m>] [--lens-note "Lens: ..."]
```

About 2 minutes. It writes `private/<visit>/import/`:

- `scan_cloud.json`: depth points within `--radius` (9 m) of the camera, voxel-averaged
  (1.5 cm), in metres from the camera (x east, y up, z south) with the origin in `georef`;
- `site_scan.png`: top view (highest point per 2 cm cell) on a lat/lon grid, bounds in the manifest;
- `dem_<unit>_<date>.tif` and `dem_preview.png`: the DEM (below);
- `reference.jpg`, `manifest.json`, `dem_info.json` (water plane, DEM height at the camera,
  mount heights for both ground models).

**Heights are orthometric** (ellipsoidal minus the scan's `GeoidSeparation`; in the US that
is NAVD88 via GEOID18), the datum of the 3D viewer and `camera_alt_m`.

**The DEM** (`ground_dem()`), 0.1 m cells, scan CRS + NAVD88 (`--vertical-epsg`):

1. each cell takes the 10th-percentile height of its points (the lowest returns);
2. objects narrower than ~1 m (enclosure, pole, plants, stems) are removed with a grey
   opening: a cell more than 0.15 m above the opened surface takes the opened value;
3. cells below the water plane are raised to it, so the water is one flat surface (the
   LiDAR may have seen the bed);
4. holes up to 0.5 m wide are filled from the nearest cell; bigger ones stay nodata.

**The water plane** is `--water-height` (a surveyed waterline point, orthometric) or else the
5th percentile of scan heights in the camera's view 2-5 m out (`--water-window`).

## 4. Checks (read the printout and the images)

| Line | What to look for |
|---|---|
| `scan heights in view` | the percentiles close together (a flat water surface or bed); the p5 is the water plane unless `--water-height` is given |
| `DEM ... filled ... removed ... raised` | the camera cell must have a height (the script stops if not) |
| `DEM along the heading` | from the mount down to the water, flat across it, up the far bank: no spikes or holes in the view |
| `API ground lookup` | the API's own lookup (`ufonet_geo.ortho`), flat vs DEM: the photo rows each maps. The DEM should map at least as much; it reaches up the far bank where a flat plane can't |

Open `dem_preview.png` (water blue, camera red: the water should lie where the photo shows
it) and `site_scan.png` (should look like the site from above). Compare the camera height
above the DEM (`mount_height_m_with_dem` in `dem_info.json`) with the taped height, if any.

## 5. Upload

You need an account with `manage_devices`. The importer asks for the password on the
terminal, so run it in a real terminal (not through a tool with no input).

1. **DEM:** dashboard -> Georeference -> DEMs -> upload `dem_<unit>_<date>.tif` (or
   `POST /dems` with form fields `name` and `file`). Its id: `GET /dems` (any account) lists
   each DEM's `id`, `name` and `vertical_datum` (should be `navd88`).
2. **Manifest with the DEM:** rerun step 3 with `--dem-id <id>`. The base pose then gets
   `map_dem_id` and `mount_height_m` = camera height above the DEM.
3. **Survey import** (`scripts/import_survey.py` is in the API repo, on main):

   ```bash
   python scripts/import_survey.py <abs path>/private/<visit>/import/manifest.json \
       --api <API URL> --email <admin email> --dry-run     # files present?
   python scripts/import_survey.py <abs path>/private/<visit>/import/manifest.json \
       --api <API URL> --email <admin email> [--close-previous]
   ```

   `--close-previous` ends the device's open survey just before this one starts (the old
   mount). It prints the survey id, "Site image: set" and "Point cloud: uploaded".
4. **Device pose config** (the fallback for captures outside every survey): dashboard ->
   Georeference -> the device's pose config, set the same `heading_deg`, `pitch_deg`,
   `roll_deg`, `camera_lat`, `camera_lon`, `mount_height_m` and `map_dem_id` as the base pose
   (or `PUT /devices/{id}/pose-config`, which merges the fields sent).
5. **Pose monitor:** "Accept current orientation", so node and survey start together.
6. **Check** (any account): `GET /devices/{id}/surveys` shows the survey open from
   `valid_from` with the base pose; `/static/surveys/<id>/reference.jpg` and
   `/static/pointclouds/<session>.json` match the local files; the `.meta.json` next to the
   cloud names the device and survey; `GET /dems` lists the DEM.

## Things that are easy to get wrong

- **`mount_height_m` means two things.** Flat ground: camera height above the *water
  plane*. With a DEM: camera height above the *DEM at the camera* (the API places the
  camera at DEM(camera) + mount_height_m). Neither is the pole height. The packager works out
  both; change `map_dem_id` and `mount_height_m` together.
- **An open survey overrides the device pose config.** Captures inside a survey's period
  are placed with the survey's base pose (or a capture pose); the device pose config is
  only the fallback outside surveys. Changing it doesn't change an open survey's maps.
- **A survey's base pose can't be edited.** To change it (another DEM, a surveyed water
  height): delete the survey (dashboard Georeference, or `DELETE /surveys/{id}`; check first
  that it has no water maps you need) and import again. The point cloud is replaced in place.
- **`valid_from` needs a time zone** and must not overlap another survey of the device.
- **The lens calibration** in the manifest must be the node's own record's (the node
  undistorts its masks with that one); the API warns on an open survey if they differ.
- **Water height from the scan is a lower bound** when the LiDAR saw the bed: survey a
  waterline point next visit and re-import with `--water-height`.
