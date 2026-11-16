# How the flood map works: from a camera photo to water on the map

This explains the idea behind the georeferencing, end to end. The step-by-step
procedure for doing it at a site is in [GEOREFERENCING_PROCEDURE.md](GEOREFERENCING_PROCEDURE.md);
the code is the shared `ufonet_geo` package in this repository.

## The idea

A camera unit sees a patch of ground. The unit labels each pixel of its photo as water or
not water, and sends that mask to the server. To put the mask on a map, the server needs to
know, for each pixel, which point on the ground it shows. That depends on three things:

1. **The lens:** how the camera bends light (its calibration).
2. **The camera's pose:** where it is and which way it points (position, heading, pitch,
   roll).
3. **The shape of the ground** it looks at: a flat plane, the water surface, or a terrain
   model.

With those, each patch of ground around the camera can be projected into the image to see
which pixel shows it, and the mask says whether that patch is water. Most of the work, and
most of the error, is in getting the pose and the ground right.

## 1. The lens (calibration)

- **What it is:** a camera matrix (focal length and image centre, in pixels) and distortion
  coefficients, which describe how the wide lens curves straight lines (barrel distortion).
- **How it's measured:** photograph a printed checkerboard many times, tilted, near and far,
  in all corners of the frame. OpenCV fits the matrix and the coefficients from where the
  corners land (procedure, section 2).
- **Generic or the unit's own:** a generic calibration for the camera model ships with the
  unit software. Units of the same model are close but not identical, so a unit's own
  calibration places water more accurately, most of all near the edges of the frame.
- **The server must use the same calibration as the unit.** The unit removes the lens
  distortion before it labels the pixels, so a mismatch shifts the map.

## 2. Which pixels are water (on the unit)

1. The unit takes its photos (colour; on five-band units also near-infrared and thermal,
   aligned to the same frame).
2. It removes the lens distortion, keeping every original pixel (the result has curved
   black borders).
3. A segmentation model (SegFormer, run with ONNX Runtime) labels each pixel.
4. The mask goes to the server as a small compressed bitmap, over LoRa or IP.

## 3. The camera's pose

Six numbers:

- **position:** latitude, longitude, and the camera's height above the ground it looks at;
- **heading:** the compass direction the lens points, relative to true north;
- **pitch:** how far it tilts down;
- **roll:** how far the image is rotated about the lens axis.

The pose is the largest source of error, and there are several ways to find it:

### 3a. The unit's own sensors: not good enough on their own

Each photo carries the unit's GPS position and IMU orientation. A consumer GPS can be
several metres off, and a magnetometer heading tens of degrees off near metal or before it
is calibrated: enough to put the water on the wrong side of a creek. The server uses these
only for values a survey doesn't provide, and shows a warning when it does.

### 3b. Surveyed markers

1. Lay printed ArUco markers on the ground in the camera's view, spread from near to far
   and left to right.
2. Survey each marker's centre with an RTK GNSS receiver (about 1 cm).
3. Take a photo with the unit and find each marker's centre in it. The crossing point of the
   marker's diagonals is its true centre, even when it is seen at an angle.
4. Solve for the pose that makes each surveyed marker project onto its pixel: the
   "perspective-n-point" problem, solved by least squares (`scripts/survey_to_pose.py`).
5. **Check it honestly:** solve again leaving one marker out, predict where that marker
   should be, and measure the miss; repeat for each marker. This held-out error is the
   accuracy figure. The fit's own residual always looks better.

### 3c. Natural tie points from a site scan

A phone scan of the site with an RTK receiver records many photos, each with its position,
orientation and a depth map. Features seen both in the unit's photo and in a scan photo
(stone edges, posts) get real-world coordinates from the scan, and the same
perspective-n-point solve then uses tens to hundreds of natural points instead of a few
markers. In testing this gave held-out errors of a few centimetres. It only works while the
scene still looks as it did during the scan: changed vegetation stops the matching.

### 3d. Later photos of the same camera

If the camera only turned and didn't move, a later photo is related to an earlier one by a
pure rotation, which can be fitted from matched features. Applying it to the earlier pose
gives the new orientation, and it shows when a camera was re-aimed. After a remount that
moves the camera, a new survey is needed.

### Surveys on the server

A survey (the pose, its accuracy and the period it applies to) is imported to the server.
Each capture is placed with the survey that covers its time; outside every survey the unit's
configured pose is used, and the dashboard marks the result as unsurveyed.

## 4. Where the rays land: the ground

Each pixel defines a ray from the camera. Where it meets the ground depends on the model:

- **Flat plane:** level ground at the camera's height below it. Right for a lawn, wrong for
  creek banks.
- **Water surface:** water is flat, so water pixels can be projected onto a plane at the
  water's height. An error in that height moves the far water edge several times as much,
  so the waterline is best surveyed directly.
- **Terrain model (DEM):** a height grid of the site, needed for banks and floodplains.
  The rays are traced onto it, and ground hidden behind higher ground is left out.

## 5. Drawing the mask on the map

1. **A grid** of cells is laid north-up on the ground around the camera, sized to what the
   camera sees and out to a maximum range.
2. **Each cell is projected into the image** with the pose and the lens. Cells are dropped
   when they fall outside the frame, are seen too close to the horizon to place reliably, lie
   beyond the range, or are hidden by terrain. This table of cell to pixel is computed once
   per pose.
3. **For each new mask,** every cell takes its pixel's value: water, dry, or not seen. That
   is one array lookup, so it is fast.
4. **The dashboard** draws the grid as an image overlay on the map and reports the water area.
   It can export the extent as GeoJSON or GeoTIFF.

## How accurate it is, and what limits it

- With a pose from many tie points and a surveyed water height, water edges land within tens
  of centimetres on the map in testing; with a few markers, the pose check is in the
  centimetre-to-decimetre range.
- **Segmentation** is often the weakest link: a model that overcalls water at the edges, or
  noise from changing light, moves the edge more than the geometry does.
- **A bumped camera** silently invalidates its pose. Each unit watches its orientation and
  warns when it has tilted or turned (the dashboard shows it), and a new survey fixes it.
- **The ground model:** a wrong water height or an inaccurate terrain model shifts the far
  edge most.

## Where things are

| What | Where |
|---|---|
| Projection, pose, terrain code | `ufonet_geo/` (`geometry.py`, `pose.py`, `ortho.py`, `terrain.py`) |
| Turning a survey into a pose | `scripts/survey_to_pose.py` |
| Site procedure | [GEOREFERENCING_PROCEDURE.md](GEOREFERENCING_PROCEDURE.md) |
| Accuracy notes | [ACCURACY_AND_EXTERNAL_RESOURCES.md](ACCURACY_AND_EXTERNAL_RESOURCES.md) |
