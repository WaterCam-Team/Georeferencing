#!/usr/bin/env python3
"""
georeference_sharp.py — GPS-anchored 3D Gaussian Splat scene viewer in a web browser

Uses Apple's SHARP model (Sharp Monocular View Synthesis in Less Than a Second,
github.com/apple/ml-sharp) to turn field photographs into a photorealistic 3D
Gaussian Splat scene, registered to real-world GPS coordinates using each photo's
EXIF GPS and IMU metadata.  The result is a web-based interactive viewer where
you can click anywhere in the 3D scene to get GPS coordinates.

USAGE:
    python georeference_sharp.py photo.jpg --output-dir out/
    python georeference_sharp.py photos_dir/ --dem dem1.tif dem2.tif --serve
    python georeference_sharp.py photos_dir/ --filter "*NIR-OFF*" --output-dir out/

    Then open:  http://localhost:8000/viewer.html
    (or run with --serve to start the server automatically)

    IMPORTANT: The viewer will NOT work when opened as file:// (e.g. double-clicking
    viewer.html). You must serve the output directory over HTTP and open the URL
    in the browser. Use:  cd <output_dir>  then  python -m http.server 8000

CONTROLS (browser viewer):
    Left drag     — orbit
    Scroll        — zoom
    Right drag    — pan
    Click         — get GPS coordinates of clicked point
    Add to boundary  — accumulate flood boundary points
    Export GeoJSON   — download flood polygon

INSTALLATION:
    uv pip install -e ~/git/ml-sharp          # SHARP (already done)
    uv pip install rasterio                   # for DEM support

METADATA EXPECTED IN EACH IMAGE:
    - GPS lat/lon/alt in standard EXIF GPSInfo tags
    - IMU orientation in EXIF UserComment: "Roll R Pitch P Yaw Y"

OUTPUT (in --output-dir):
    viewer.html      open this in a browser via local HTTP server
    scene/splats.ply merged world-space 3DGS point cloud
    scene_meta.json  GPS origin + camera list
"""

import argparse
import fnmatch
import http.server
import json
import os
import sys
import threading
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from pyproj import Proj
import torch

from camera_geometry import build_rotation_matrix
from exif_imu import read_gps_imu_from_exif

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: EXIF / IMU METADATA READING
# ─────────────────────────────────────────────────────────────────────────────

def read_image_metadata(path):
    """Read GPS and IMU from image EXIF. Returns dict with lat/lon/alt/roll/pitch/yaw."""
    return read_gps_imu_from_exif(path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CAMERA INTRINSICS
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics(calib_path=None, img_w=None, img_h=None,
                    focal_mm=4.44, sensor_w_mm=4.614):
    """Return (K, calib_size). K is 3×3."""
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            d = json.load(f)
        K = np.array(d["K"], dtype=np.float64)
        print(f"[CALIB] Loaded {calib_path}  (RMS={d.get('rms', '?')} px)")
        return K, d.get("img_size")
    if img_w is None:
        raise ValueError("Need image size or --calib to build intrinsics")
    px_per_mm = img_w / sensor_w_mm
    fx = fy = focal_mm * px_per_mm
    K = np.array([[fx, 0, img_w / 2.0],
                  [0, fy, img_h / 2.0],
                  [0,  0,          1.0]], dtype=np.float64)
    print(f"[CALIB] Nominal intrinsics  fx={fx:.1f} px")
    return K, None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: IMAGE LOADING WITH FILTER
# ─────────────────────────────────────────────────────────────────────────────

def load_images(paths, name_filter="*NIR-OFF*"):
    """Load images + EXIF metadata. Returns list of {path, img(BGR), meta, w, h}."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []
    for p in [Path(x) for x in paths]:
        if not p.exists():
            print(f"  [WARN] Not found: {p}  (quote paths with spaces)")
            continue
        if p.is_dir():
            found = sorted(x for x in p.rglob("*")
                           if x.suffix.lower() in exts
                           and fnmatch.fnmatch(x.name, name_filter))
            if not found:
                print(f"  [WARN] No files matching {name_filter!r} under {p}")
            image_paths.extend(found)
        elif p.suffix.lower() in exts:
            image_paths.append(p)

    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [WARN] Could not read {p.name}")
            continue
        h, w = img.shape[:2]
        meta = read_image_metadata(str(p))
        images.append({"path": str(p), "img": img, "meta": meta, "w": w, "h": h})
        gps_s = f"{meta['lat']:.5f}" if meta["lat"] else "no GPS"
        imu_s = (f"R={meta['roll_deg']:.1f} P={meta['pitch_deg']:.1f} "
                 f"Y={meta['yaw_deg']:.1f}") if meta["yaw_deg"] is not None else "no IMU"
        print(f"  {p.name}  {w}×{h}  {gps_s}  {imu_s}")

    print(f"[LOAD] {len(images)} images")
    return images


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SHARP INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_sharp_model(checkpoint, device):
    """Load SHARP model from checkpoint or auto-download."""
    from sharp.models import create_predictor, PredictorParams
    from sharp.cli.predict import DEFAULT_MODEL_URL

    predictor = create_predictor(PredictorParams())
    if checkpoint and Path(checkpoint).exists():
        print(f"[SHARP] Loading checkpoint from {checkpoint}")
        state_dict = torch.load(checkpoint, weights_only=True, map_location=device)
    else:
        print(f"[SHARP] Downloading model weights (~2.5 GB, one-time)…")
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True, map_location=device)
    predictor.load_state_dict(state_dict)
    return predictor.eval().to(device)


def run_sharp(predictor, image_bgr, K, device):
    """
    Run SHARP on one image.
    Returns Gaussians3D in OpenCV camera space (X right, Y down, Z forward), metric.
    """
    from sharp.cli.predict import predict_image

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    f_px = float(K[0, 0])
    return predict_image(predictor, image_rgb, f_px, device)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: COORDINATE TRANSFORM — CAMERA SPACE → THREE.JS WORLD
# ─────────────────────────────────────────────────────────────────────────────
#
# Coordinate systems:
#   SHARP/OpenCV camera: X right, Y down, Z forward
#   World ENU:           X East, Y North, Z Up
#   Three.js Y-up:       X East, Y Up, Z -North
#
# build_rotation_matrix returns R (world ENU → camera, 3×3).
# Camera-to-world: R.T.
# ENU → Three.js: M = [[1,0,0],[0,0,1],[0,-1,0]]
# Combined 3×4 transform for apply_transform:
#   T_linear = M @ R.T    (3×3)
#   T_offset = M @ cam_enu_3d
# ─────────────────────────────────────────────────────────────────────────────

_M_ENU2THREE = np.array([[1,  0,  0],
                          [0,  0,  1],
                          [0, -1,  0]], dtype=np.float32)


def gps_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt=0.0):
    """Single GPS position → ENU (metres) relative to origin."""
    proj = Proj(proj="aeqd", lat_0=origin_lat, lon_0=origin_lon, datum="WGS84")
    e, n = proj(lon, lat)
    u = (alt or 0.0) - (origin_alt or 0.0)
    return np.array([e, n, u], dtype=np.float32)


def make_cam_to_threejs_transform(R_world_to_cam, cam_enu):
    """
    Build 3×4 float32 torch.Tensor for apply_transform().
    Combines camera→ENU rotation + ENU→Three.js convention change.
    """
    R_cam_to_three = (_M_ENU2THREE @ R_world_to_cam.T).astype(np.float32)
    t_three = (_M_ENU2THREE @ cam_enu.astype(np.float32))
    T = np.hstack([R_cam_to_three, t_three.reshape(3, 1)])   # (3, 4)
    return torch.from_numpy(T)


def transform_to_world(gaussians, R_world_to_cam, cam_enu):
    """Transform SHARP Gaussians from camera OpenCV space to Three.js world."""
    from sharp.utils.gaussians import apply_transform
    T = make_cam_to_threejs_transform(R_world_to_cam, cam_enu)
    return apply_transform(gaussians, T)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MERGE AND EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def filter_gaussians(gaussians, min_opacity=0.05, max_gaussians=300_000):
    """Remove low-opacity Gaussians and subsample to max_gaussians (keeps highest-opacity)."""
    from sharp.utils.gaussians import Gaussians3D
    ops = gaussians.opacities[0]          # (N,)
    mask = ops >= min_opacity
    if mask.sum() > max_gaussians:
        # Keep the max_gaussians highest-opacity ones within the mask
        vals = ops.clone()
        vals[~mask] = -1.0
        keep_idx = vals.topk(max_gaussians).indices
        mask = torch.zeros_like(ops, dtype=torch.bool)
        mask[keep_idx] = True
    n_in  = gaussians.opacities.shape[1]
    n_out = int(mask.sum())
    print(f"         filter: {n_in:,} → {n_out:,} Gaussians "
          f"(opacity≥{min_opacity}, max={max_gaussians:,})")
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, mask, :],
        singular_values=gaussians.singular_values[:, mask, :],
        quaternions=gaussians.quaternions[:, mask, :],
        colors=gaussians.colors[:, mask, :],
        opacities=gaussians.opacities[:, mask],
    )


def merge_gaussians(gs_list):
    """Concatenate multiple Gaussians3D along the N dimension."""
    from sharp.utils.gaussians import Gaussians3D
    return Gaussians3D(
        mean_vectors=torch.cat([g.mean_vectors for g in gs_list], dim=1),
        singular_values=torch.cat([g.singular_values for g in gs_list], dim=1),
        quaternions=torch.cat([g.quaternions for g in gs_list], dim=1),
        colors=torch.cat([g.colors for g in gs_list], dim=1),
        opacities=torch.cat([g.opacities for g in gs_list], dim=1),
    )


def sample_positions(gaussians, n_samples=15_000):
    """Return flat list [x,y,z, x,y,z, …] of subsampled Gaussian centres for JS raycasting."""
    pos = gaussians.mean_vectors[0].detach().cpu().float().numpy()  # (N, 3)
    N = len(pos)
    stride = max(1, N // n_samples)
    sampled = pos[::stride]   # (n_samples, 3)
    return sampled.flatten().tolist()


def save_world_ply(path, gaussians, f_px, img_h, img_w):
    """Save merged world-space Gaussians as a standard 3DGS .ply file."""
    from sharp.utils.gaussians import save_ply
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(gaussians, f_px, (img_h, img_w), path)
    n = gaussians.mean_vectors.shape[1]
    print(f"[PLY] Saved {n:,} Gaussians → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: OPTIONAL DEM (for camera altitude metadata)
# ─────────────────────────────────────────────────────────────────────────────

def load_dem(paths):
    """
    Load DEM GeoTIFF tiles. Returns get_elevation(lon, lat) → float | None.
    Used to record ground elevation in scene_meta.json.
    """
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        raise ImportError("DEM support requires: uv pip install rasterio")

    sources = []
    for p in paths:
        src = rasterio.open(p)
        crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
        xfm = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        sources.append((src, xfm, src.nodata))
        print(f"[DEM] {Path(p).name}  {src.width}×{src.height} px  "
              f"res={src.res[0]:.1f} m  crs={crs_str}")

    def get_elevation(lon, lat):
        import rasterio.windows
        for src, xfm, nodata in sources:
            x, y = xfm.transform(lon, lat)
            l, b, r, t = src.bounds
            if not (l <= x <= r and b <= y <= t):
                continue
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                continue
            val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1)).flat[0]
            if (nodata is not None and val == nodata) or np.isnan(val):
                continue
            return float(val)
        return None

    return get_elevation


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: HTML VIEWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def write_meta_json(path, origin_lat, origin_lon, origin_alt, cameras,
                    sample_pts=None, init_cam_pos=None, init_cam_look=None):
    """Write scene_meta.json with GPS origin, camera list, and sample positions."""
    data = {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "origin_alt": origin_alt,
        "cameras": cameras,
        "sample_positions": sample_pts or [],   # flat [x,y,z,…] for JS raycasting
        "init_cam_pos":  init_cam_pos  or [0, 30, 80],
        "init_cam_look": init_cam_look or [0,  5, -20],
    }
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))   # compact — sample_pts can be large
    kb = path.stat().st_size / 1024
    print(f"[META] {path}  ({kb:.0f} KB)")


def generate_html(ply_filename):
    """Return self-contained HTML for the 3DGS browser viewer."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHARP 3D Scene Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #111; color: #eee; font-family: monospace; overflow: hidden; }}
  #container {{ width: 100vw; height: 100vh; position: relative; }}
  #panel {{
    position: fixed; top: 12px; right: 12px; width: 280px;
    background: rgba(0,0,0,0.78); border: 1px solid #444;
    border-radius: 8px; padding: 14px; z-index: 100;
    backdrop-filter: blur(4px);
  }}
  #panel h3 {{ margin-bottom: 8px; color: #7cf; font-size: 13px; }}
  #coords {{ font-size: 12px; line-height: 1.7; color: #cfc; min-height: 60px; }}
  .btn {{
    margin-top: 6px; width: 100%; padding: 6px; border: none;
    border-radius: 4px; cursor: pointer; font-size: 12px; font-family: monospace;
  }}
  #btn-copy  {{ background: #226; color: #aaf; }}
  #btn-add   {{ background: #262; color: #afa; }}
  #btn-clear {{ background: #422; color: #faa; }}
  #btn-export{{ background: #443; color: #ffa; }}
  #boundary-list {{
    margin-top: 8px; font-size: 11px; color: #999;
    max-height: 120px; overflow-y: auto;
  }}
  #hint {{
    position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.6); padding: 6px 14px; border-radius: 4px;
    font-size: 11px; color: #888; pointer-events: none; z-index: 100;
  }}
  #loading {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 16px; color: #7cf; text-align: center; z-index: 200;
    background: rgba(0,0,0,0.7); padding: 20px 30px; border-radius: 8px;
  }}
</style>
</head>
<body>
<div id="container"></div>
<div id="panel">
  <h3>SHARP GPS Viewer</h3>
  <div id="coords">Click a point in the scene…</div>
  <button class="btn" id="btn-copy"   onclick="copyCoords()">Copy coords</button>
  <button class="btn" id="btn-add"    onclick="addBoundary()">Add to boundary</button>
  <button class="btn" id="btn-clear"  onclick="clearBoundary()">Clear boundary</button>
  <button class="btn" id="btn-export" onclick="exportGeoJSON()">Export GeoJSON</button>
  <div id="boundary-list"></div>
</div>
<div id="hint">Left drag: orbit &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; Right drag: pan &nbsp;|&nbsp; Click: GPS</div>
<div id="loading">Loading 3D scene…<br><small style="color:#888;font-size:11px">Large scenes may take a moment</small></div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "@mkkellogg/gaussian-splats-3d": "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

// ── Require HTTP (file:// cannot load scene_meta.json or .ply due to CORS) ───
if (window.location.protocol === 'file:') {{
  document.getElementById('loading').innerHTML =
    '<span style="color:#f88">Cannot open as file://</span><br><br>' +
    '<span style="color:#ccc;font-size:12px">Run a local server from the <b>output directory</b>:<br>' +
    '<code style="color:#7cf">cd &lt;output_dir&gt;</code><br>' +
    '<code style="color:#7cf">python -m http.server 8000</code><br>' +
    'Then open: <code style="color:#7cf">http://localhost:8000/viewer.html</code></span>';
  throw new Error('Open via http:// (see message on page)');
}}

// ── Base URL for assets (same directory as this HTML) ───────────────────────
const baseUrl = new URL(window.location.href);

// ── Load metadata ─────────────────────────────────────────────────────────────
let meta = null;
try {{
  const res = await fetch(new URL('scene_meta.json', baseUrl));
  if (!res.ok) throw new Error(`scene_meta.json: HTTP ${{res.status}}`);
  meta = await res.json();
}} catch (e) {{
  document.getElementById('loading').innerHTML =
    `<span style="color:#f88">Failed to load scene_meta.json:<br>${{e.message}}</span><br>` +
    `<small style="color:#888">Serve this folder via HTTP: cd &lt;output_dir&gt; && python -m http.server 8000</small>`;
  throw e;
}}

// ── Three.js world (X=East, Y=Up, Z=-North) → GPS ────────────────────────────
function worldToGPS(wx, wy, wz) {{
  const e = wx, n = -wz, u = wy;
  const R = 6371000;
  const olat = meta.origin_lat * Math.PI / 180;
  return {{
    lat: meta.origin_lat + (n / R) * (180 / Math.PI),
    lon: meta.origin_lon + (e / (R * Math.cos(olat))) * (180 / Math.PI),
    alt: meta.origin_alt + u
  }};
}}

// ── GaussianSplats3D viewer (self-driven mode) ────────────────────────────────
const container = document.getElementById('container');
const icp = meta.init_cam_pos  || [0, 30, 80];
const icl = meta.init_cam_look || [0,  5, -20];

const viewer = new GaussianSplats3D.Viewer({{
  rootElement: container,
  cameraUp: [0, 1, 0],
  initialCameraPosition: icp,
  initialCameraLookAt: icl,
  splatAlphaRemovalThreshold: 5,
  dynamicScene: false,
}});

const sceneUrl = new URL('scene/{ply_filename}', baseUrl).href;
viewer.addSplatScene(sceneUrl, {{
  splatAlphaRemovalThreshold: 5,
  showLoadingUI: false,
}}).then(() => {{
  viewer.start();
  document.getElementById('loading').style.display = 'none';
}}).catch(err => {{
  console.error('Scene load failed:', err);
  document.getElementById('loading').innerHTML =
    `<span style="color:#f88">Scene load error:<br>${{err.message}}</span><br>` +
    `<small style="color:#888">1) Open DevTools (F12) → Console for details<br>` +
    `2) Serve from output dir: <code>cd &lt;out_dir&gt; && python -m http.server 8000</code><br>` +
    `3) Open <code>http://localhost:8000/viewer.html</code> (not file://)</small>`;
}});

// ── State ─────────────────────────────────────────────────────────────────────
let lastGPS = null;
let boundary = [];

function updateCoords(wx, wy, wz) {{
  const gps = worldToGPS(wx, wy, wz);
  lastGPS = gps;
  const e = wx.toFixed(1), n = (-wz).toFixed(1), u = wy.toFixed(1);
  document.getElementById('coords').innerHTML =
    `Lat: <b>${{gps.lat.toFixed(7)}}</b><br>` +
    `Lon: <b>${{gps.lon.toFixed(7)}}</b><br>` +
    `Alt: <b>${{gps.alt.toFixed(1)}} m</b><br>` +
    `<span style="color:#888;font-size:10px">ENU E=${{e}} N=${{n}} U=${{u}} m</span>`;
}}

window.copyCoords = () => {{
  if (!lastGPS) return;
  navigator.clipboard.writeText(`${{lastGPS.lat.toFixed(7)}}, ${{lastGPS.lon.toFixed(7)}}`);
}};
window.addBoundary = () => {{
  if (!lastGPS) return;
  boundary.push({{ ...lastGPS }});
  renderBoundaryList();
}};
window.clearBoundary = () => {{ boundary = []; renderBoundaryList(); }};

function renderBoundaryList() {{
  const el = document.getElementById('boundary-list');
  el.innerHTML = boundary.length === 0 ? '' :
    `<b style="color:#aaa">${{boundary.length}} pt(s):</b><br>` +
    boundary.map((p, i) => `${{i+1}}. ${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}`).join('<br>');
}}

window.exportGeoJSON = () => {{
  if (boundary.length < 3) {{ alert('Need ≥3 boundary points.'); return; }}
  const coords = boundary.map(p => [p.lon, p.lat]);
  coords.push(coords[0]);
  const gj = {{ type:'FeatureCollection', features:[{{ type:'Feature',
    geometry:{{ type:'Polygon', coordinates:[coords] }},
    properties:{{ source:'SHARP viewer', n_points:boundary.length }} }}] }};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(gj,null,2)], {{type:'application/json'}}));
  a.download = 'flood_boundary.geojson';  a.click();
}};

document.addEventListener('keydown', e => {{
  if (e.key==='g'||e.key==='G') window.exportGeoJSON();
  if (e.key==='c'||e.key==='C') window.clearBoundary();
}});

// ── Click → GPS ───────────────────────────────────────────────────────────────
// Primary: viewer.rayCast()  Fallback: nearest sample point to the view ray
container.addEventListener('click', async (e) => {{
  // 1. GaussianSplats3D rayCast
  try {{
    const hit = await viewer.rayCast(e.clientX, e.clientY);
    if (hit && hit.intersection) {{
      const p = hit.intersection;
      updateCoords(p.x, p.y, p.z);
      return;
    }}
  }} catch (_) {{}}

  // 2. Nearest sample point (pure-JS, no Three.js scene integration needed)
  const pts = meta.sample_positions;
  if (!pts || !pts.length || !viewer.camera) return;

  const W = container.clientWidth, H = container.clientHeight;
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(
    new THREE.Vector2((e.clientX / W) * 2 - 1, -(e.clientY / H) * 2 + 1),
    viewer.camera
  );
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;

  let minD2 = 400, bx = null, by = 0, bz = 0;  // 20 m threshold
  for (let i = 0; i < pts.length; i += 3) {{
    const px = pts[i], py = pts[i+1], pz = pts[i+2];
    const t = (px-ro.x)*rd.x + (py-ro.y)*rd.y + (pz-ro.z)*rd.z;
    if (t < 0) continue;
    const cx = ro.x + t*rd.x, cy = ro.y + t*rd.y, cz = ro.z + t*rd.z;
    const d2 = (px-cx)**2 + (py-cy)**2 + (pz-cz)**2;
    if (d2 < minD2) {{ minD2 = d2; bx = px; by = py; bz = pz; }}
  }}
  if (bx !== null) updateCoords(bx, by, bz);
}});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="SHARP 3D Gaussian Splat scene viewer with GPS coordinates in browser")
    ap.add_argument("inputs", nargs="+",
                    help="Image files or directories")
    ap.add_argument("--filter", default="*NIR-OFF*",
                    help="Filename glob for directory search (default: *NIR-OFF*)")
    ap.add_argument("--calib", default=None,
                    help="Camera calibration JSON (optional but recommended)")
    ap.add_argument("--height", type=float, default=5.0,
                    help="Camera mount height above ground in metres (default 5.0)")
    ap.add_argument("--dem", nargs="+", default=None,
                    help="DEM GeoTIFF files for ground elevation metadata")
    ap.add_argument("--output-dir", default="sharp_output",
                    help="Output directory (default: sharp_output/)")
    ap.add_argument("--checkpoint", default=None,
                    help="Local SHARP model weights .pt file (auto-downloads if omitted)")
    ap.add_argument("--device", default=None,
                    choices=["cpu", "mps", "cuda"],
                    help="Torch device (default: auto-detect)")
    ap.add_argument("--serve", action="store_true",
                    help="Start local HTTP server and open browser after generating")
    ap.add_argument("--port", type=int, default=8000,
                    help="HTTP server port (default: 8000)")
    args = ap.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[DEVICE] {device}")

    # ── Load images ───────────────────────────────────────────────────────────
    print("[LOAD] Reading images…")
    images = load_images(args.inputs, name_filter=args.filter)
    if not images:
        sys.exit("[ERROR] No images loaded.")

    # Skip images without GPS or IMU
    valid = [im for im in images
             if im["meta"]["lat"] is not None and im["meta"]["yaw_deg"] is not None]
    skipped = [Path(im["path"]).name for im in images if im not in valid]
    if skipped:
        print(f"[SKIP] {len(skipped)} image(s) without GPS+IMU: "
              + ", ".join(skipped[:5]) + ("…" if len(skipped) > 5 else ""))
    if not valid:
        sys.exit("[ERROR] No images with GPS + IMU metadata found.")
    images = valid
    print(f"[USE]  {len(images)} images with GPS+IMU")

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    first = images[0]
    K, calib_size = load_intrinsics(args.calib, first["w"], first["h"])

    # Scale K to image resolution if calibration was for a different size
    if calib_size is not None:
        cw, ch = calib_size
        if cw != first["w"] or ch != first["h"]:
            sx = first["w"] / cw
            sy = first["h"] / ch
            K = K.copy()
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy

    # ── ENU origin = first image GPS ──────────────────────────────────────────
    m0 = images[0]["meta"]
    origin_lat = m0["lat"]
    origin_lon = m0["lon"]
    origin_alt = m0["alt"] or 0.0
    print(f"[ORIGIN] {origin_lat:.6f}°N, {origin_lon:.6f}°E, alt={origin_alt:.1f} m")

    # ── Optional DEM ──────────────────────────────────────────────────────────
    get_elevation = None
    if args.dem:
        get_elevation = load_dem(args.dem)

    # ── Load SHARP model ──────────────────────────────────────────────────────
    predictor = load_sharp_model(args.checkpoint, device)

    # ── Per-image: SHARP inference + geo-transform ────────────────────────────
    all_gaussians = []
    camera_meta = []
    f_px_first = float(K[0, 0])
    img_h_first, img_w_first = images[0]["h"], images[0]["w"]

    for i, im in enumerate(images):
        name = Path(im["path"]).name
        m = im["meta"]
        print(f"[SHARP] {i+1}/{len(images)}  {name}…")

        # Run SHARP
        gaussians_cam = run_sharp(predictor, im["img"], K, device)
        n_g = gaussians_cam.mean_vectors.shape[1]
        print(f"         → {n_g:,} Gaussians")

        # Camera ENU position (GPS alt + mount height is camera height above ground)
        cam_enu = gps_to_enu(m["lat"], m["lon"],
                              (m["alt"] or origin_alt) + args.height,
                              origin_lat, origin_lon, origin_alt)

        # Camera rotation matrix (world ENU → camera)
        R = build_rotation_matrix(m["yaw_deg"], m["pitch_deg"], m["roll_deg"])

        # Transform to Three.js world space, then filter low-opacity Gaussians
        gaussians_world = transform_to_world(gaussians_cam, R, cam_enu)
        gaussians_world = filter_gaussians(gaussians_world,
                                           min_opacity=0.05, max_gaussians=300_000)
        all_gaussians.append(gaussians_world)

        # Ground elevation for metadata
        ground_elev = None
        if get_elevation:
            ground_elev = get_elevation(m["lon"], m["lat"])

        camera_meta.append({
            "image_name": name,
            "lat": m["lat"],
            "lon": m["lon"],
            "alt": m["alt"],
            "heading": m["yaw_deg"],
            "pitch": m["pitch_deg"],
            "roll": m["roll_deg"],
            "enu_e": float(cam_enu[0]),
            "enu_n": float(cam_enu[1]),
            "enu_u": float(cam_enu[2]),
            "ground_elev_m": ground_elev,
        })

    # ── Merge and save ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    scene_dir = out_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    print("[MERGE] Combining Gaussians from all images…")
    merged = merge_gaussians(all_gaussians)
    ply_path = scene_dir / "splats.ply"
    save_world_ply(ply_path, merged, f_px_first, img_h_first, img_w_first)

    # ── Compute initial viewer camera position from first camera ──────────────
    import math as _math
    c0 = camera_meta[0]
    cx, cy, cz = c0["enu_e"], c0["enu_u"], -c0["enu_n"]   # Three.js coords
    h_rad = _math.radians(c0["heading"])
    fwd_x, fwd_z = _math.sin(h_rad), -_math.cos(h_rad)    # forward in Three.js
    init_cam_pos  = [cx - fwd_x * 30, cy + 15, cz - fwd_z * 30]
    init_cam_look = [cx + fwd_x * 20, cy + 2,  cz + fwd_z * 20]

    # ── Subsample Gaussian centres for click-to-GPS fallback ─────────────────
    print("[SAMPLE] Extracting sample positions for click-to-GPS…")
    sample_pts = sample_positions(merged, n_samples=15_000)

    # ── Write metadata and HTML ───────────────────────────────────────────────
    meta_path = out_dir / "scene_meta.json"
    write_meta_json(meta_path, origin_lat, origin_lon, origin_alt, camera_meta,
                    sample_pts=sample_pts,
                    init_cam_pos=init_cam_pos,
                    init_cam_look=init_cam_look)

    html_content = generate_html("splats.ply")
    viewer_path = out_dir / "viewer.html"
    viewer_path.write_text(html_content)
    print(f"[HTML] {viewer_path}")

    # ── Instructions / serve ──────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"  Output: {out_dir.resolve()}")
    print()
    print("  To view (must use HTTP, not file://):")
    print(f"    Use  --serve  so the script runs a server with the right headers")
    print(f"    (SharedArrayBuffer needs Cross-Origin-Opener-Policy / Embedder-Policy).")
    print(f"    Or:  cd {out_dir.resolve()}")
    print(f"         python -m http.server {args.port}")
    print(f"    Then open http://localhost:{args.port}/viewer.html")
    print("  If the scene does not render: open DevTools (F12) → Console for errors.")
    print("─" * 60)

    if args.serve:
        os.chdir(out_dir.resolve())

        # SharedArrayBuffer (used by the Gaussian Splats decoder) requires
        # cross-origin isolation. COEP: credentialless allows CDN scripts
        # to load while still enabling SharedArrayBuffer (Chrome 110+).
        class _CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
                super().end_headers()

            def log_message(self, *a): pass

        server = http.server.HTTPServer(("", args.port), _CrossOriginIsolatedHandler)
        url = f"http://localhost:{args.port}/viewer.html"
        print(f"\n[SERVE] {url}  (Ctrl-C to stop)")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
