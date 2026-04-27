"""
Generate a self-contained HTML 3D point-cloud viewer for a Pix4DCatch scan.

Reads the opf_format or legacy point cloud, subsamples if needed, and writes
a single portable HTML file using Three.js (loaded from CDN).

Usage:
    .venv/bin/python scripts/pix4d_to_viewer.py <scan_dir> [options]

Options:
    --out PATH        Output HTML file (default: <scan_name>_viewer.html)
    --max-points N    Maximum points to include (default: 300000)
    --point-size F    Initial point size in world units (default: 0.02)
"""

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_opf_format(scan_dir: Path):
    """Load positions + RGB from point_clouds/opf_format/."""
    pcl_dir = scan_dir / "point_clouds" / "opf_format"
    gltf_path = pcl_dir / "pcl.gltf"

    with open(gltf_path) as f:
        gltf = json.load(f)

    acc = gltf["accessors"]
    n_points = acc[0]["count"]

    # Positions: accessor 0 → bufferView 0 → positions.glbin
    pos_bv = gltf["bufferViews"][acc[0]["bufferView"]]
    pos_buf = gltf["buffers"][pos_bv["buffer"]]
    pos_path = pcl_dir / pos_buf["uri"]
    offset = pos_bv.get("byteOffset", 0)
    pos_raw = np.frombuffer(
        pos_path.read_bytes()[offset: offset + pos_bv["byteLength"]],
        dtype=np.float32,
    ).reshape(n_points, 3)

    # Colors: accessor 2 → bufferView 2 → colors.glbin (VEC4 uint8)
    col_bv = gltf["bufferViews"][acc[2]["bufferView"]]
    col_buf = gltf["buffers"][col_bv["buffer"]]
    col_path = pcl_dir / col_buf["uri"]
    offset = col_bv.get("byteOffset", 0)
    col_raw = np.frombuffer(
        col_path.read_bytes()[offset: offset + col_bv["byteLength"]],
        dtype=np.uint8,
    ).reshape(n_points, 4)

    # Apply node transform (column-major glTF → numpy row-major)
    # Converts Pix4D Z-up local frame to Three.js Y-up: x'=x, y'=z, z'=-y
    node = gltf["nodes"][0]
    if "matrix" in node:
        m = np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
        ones = np.ones((n_points, 1), dtype=np.float64)
        pts_h = np.hstack([pos_raw.astype(np.float64), ones])
        transformed = (m @ pts_h.T).T[:, :3].astype(np.float32)
    else:
        transformed = pos_raw

    return transformed, col_raw[:, :3]  # drop alpha


def _load_legacy(scan_dir: Path):
    """Load positions + RGB from point_clouds/legacy/ (interleaved float32 xyz+rgb)."""
    legacy = scan_dir / "point_clouds" / "legacy"
    with open(legacy / "pointcloud.gltf") as f:
        gltf = json.load(f)

    n_points = gltf["accessors"][0]["count"]
    stride = gltf["bufferViews"][0]["byteStride"]

    raw = np.frombuffer(
        (legacy / "pointcloud.bin").read_bytes()[: n_points * stride],
        dtype=np.float32,
    ).reshape(n_points, stride // 4)

    xyz = raw[:, :3]
    rgb = (raw[:, 3:6] * 255).clip(0, 255).astype(np.uint8)
    return xyz, rgb


def load_pointcloud(scan_dir: Path):
    opf = scan_dir / "point_clouds" / "opf_format" / "pcl.gltf"
    if opf.exists():
        print("  Using opf_format point cloud")
        return _load_opf_format(scan_dir)
    legacy = scan_dir / "point_clouds" / "legacy" / "pointcloud.gltf"
    if legacy.exists():
        print("  Using legacy point cloud")
        return _load_legacy(scan_dir)
    raise FileNotFoundError(f"No point cloud found under {scan_dir / 'point_clouds'}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; overflow: hidden; font-family: system-ui, sans-serif; }}
  canvas {{ display: block; }}

  #ui {{
    position: absolute; top: 16px; left: 16px;
    background: rgba(13,17,23,0.85);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 14px 18px;
    color: #e6edf3; font-size: 13px; line-height: 1.8;
    backdrop-filter: blur(6px); user-select: none;
    min-width: 220px;
  }}
  #ui h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 8px;
            color: #79c0ff; letter-spacing: 0.02em; }}
  #ui .stat {{ color: #8b949e; }}
  #ui label {{ display: flex; align-items: center; gap: 8px; margin-top: 10px; }}
  #ui input[type=range] {{ flex: 1; accent-color: #79c0ff; }}
  #ui select {{ background: #161b22; color: #e6edf3;
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 4px; padding: 2px 6px; font-size: 12px; }}
  #hint {{
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    color: rgba(255,255,255,0.3); font-size: 12px; pointer-events: none;
  }}
</style>
</head>
<body>
<div id="ui">
  <h3>{title}</h3>
  <div class="stat">{n_points:,} points &nbsp;·&nbsp; {extent}</div>
  <label>Point size
    <input type="range" id="sizeSlider" min="0.002" max="0.15"
           step="0.001" value="{point_size}">
  </label>
  <label>Color
    <select id="colorMode">
      <option value="rgb">Scan color</option>
      <option value="height">Height</option>
      <option value="flat">Flat white</option>
    </select>
  </label>
</div>
<div id="hint">Drag to rotate &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Right-drag to pan</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// ── decode base64 binary arrays ──────────────────────────────────────────
function b64toF32(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}
function b64toU8(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}}

const posB64 = `{pos_b64}`;
const colB64 = `{col_b64}`;

const posArr = b64toF32(posB64);
const colArr = b64toU8(colB64);
const N = posArr.length / 3;

// ── scene setup ──────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0d1117);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.001, 500);
camera.position.set(0, 8, 14);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.1;
controls.maxDistance = 80;

// ── build geometry ───────────────────────────────────────────────────────
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));

// Build normalized RGB Float32 from uint8
const rgbArr = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {{
  rgbArr[i * 3]     = colArr[i * 3]     / 255;
  rgbArr[i * 3 + 1] = colArr[i * 3 + 1] / 255;
  rgbArr[i * 3 + 2] = colArr[i * 3 + 2] / 255;
}}

// Height colormap (viridis-like)
function heightColor(y, yMin, yRange) {{
  const t = Math.max(0, Math.min(1, (y - yMin) / yRange));
  const r = Math.max(0, 1.5 * t - 0.5);
  const g = Math.max(0, Math.sin(Math.PI * t));
  const b = Math.max(0, 1 - 1.5 * t);
  return [r, g, b];
}}
let yMin = Infinity, yMax = -Infinity;
for (let i = 0; i < N; i++) {{ const y = posArr[i*3+1]; if (y < yMin) yMin=y; if (y > yMax) yMax=y; }}
const yRange = yMax - yMin;
const heightArr = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {{
  const [r, g, b] = heightColor(posArr[i*3+1], yMin, yRange);
  heightArr[i*3]=r; heightArr[i*3+1]=g; heightArr[i*3+2]=b;
}}

geometry.setAttribute('color', new THREE.BufferAttribute(rgbArr, 3));

const material = new THREE.PointsMaterial({{
  vertexColors: true,
  size: {point_size},
  sizeAttenuation: true,
}});
const cloud = new THREE.Points(geometry, material);
scene.add(cloud);

// Axes (small, at origin)
scene.add(new THREE.AxesHelper(0.5));

// Centre camera on bounding box
geometry.computeBoundingBox();
const bbox = geometry.boundingBox;
const centre = new THREE.Vector3();
bbox.getCenter(centre);
controls.target.copy(centre);
const size = new THREE.Vector3();
bbox.getSize(size);
const maxDim = Math.max(size.x, size.y, size.z);
camera.position.copy(centre).add(new THREE.Vector3(0, maxDim * 0.6, maxDim * 1.1));
camera.near = maxDim * 0.001;
camera.far  = maxDim * 20;
camera.updateProjectionMatrix();

// ── UI wiring ────────────────────────────────────────────────────────────
document.getElementById('sizeSlider').addEventListener('input', e => {{
  material.size = parseFloat(e.target.value);
}});

document.getElementById('colorMode').addEventListener('change', e => {{
  if (e.target.value === 'rgb') {{
    geometry.setAttribute('color', new THREE.BufferAttribute(rgbArr, 3));
    material.vertexColors = true; material.color.set(0xffffff);
  }} else if (e.target.value === 'height') {{
    geometry.setAttribute('color', new THREE.BufferAttribute(heightArr, 3));
    material.vertexColors = true; material.color.set(0xffffff);
  }} else {{
    material.vertexColors = false; material.color.set(0xffffff);
  }}
  material.needsUpdate = true;
}});

// ── render loop ──────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def generate_viewer(scan_dir: Path, out_path: Path,
                    max_points: int, point_size: float) -> None:
    print(f"\nProcessing: {scan_dir.name}")
    xyz, rgb = load_pointcloud(scan_dir)
    n_total = len(xyz)
    print(f"  Points loaded: {n_total:,}")

    # Subsample
    if n_total > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, max_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        rgb = rgb[idx]
        print(f"  Subsampled to: {len(xyz):,}")

    n_points = len(xyz)

    # Extent description
    size = xyz.max(axis=0) - xyz.min(axis=0)
    extent = f"{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} m"

    # Base64 encode
    pos_b64 = base64.b64encode(xyz.astype(np.float32).tobytes()).decode()
    col_b64 = base64.b64encode(rgb.astype(np.uint8).tobytes()).decode()

    # Split into 76-char lines so the string doesn't hit JS engine limits
    def wrap(s, w=76):
        return "\n".join(s[i:i+w] for i in range(0, len(s), w))

    title = scan_dir.name
    html = _HTML_TEMPLATE.format(
        title=title,
        n_points=n_points,
        extent=extent,
        point_size=point_size,
        pos_b64=wrap(pos_b64),
        col_b64=wrap(col_b64),
    )

    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Written: {out_path}  ({size_mb:.1f} MB)")
    print(f"  Open in browser: file://{out_path.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", help="Pix4DCatch scan directory")
    ap.add_argument("--out", default=None,
                    help="Output HTML file (default: <scan_name>_viewer.html)")
    ap.add_argument("--max-points", type=int, default=300_000,
                    help="Maximum points (default: 300000)")
    ap.add_argument("--point-size", type=float, default=0.02,
                    help="Initial point size in metres (default: 0.02)")
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: not a directory: {scan_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else Path(f"{scan_dir.name}_viewer.html")

    generate_viewer(scan_dir, out_path, args.max_points, args.point_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
