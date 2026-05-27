"""
Export a Pix4DCatch scan to the JSON format consumed by the UFONet API dashboard.

Writes two files to --out-dir:
  <session_id>.json       -- point cloud data (pos_b64 + col_b64, ~5 MB)
  <session_id>.meta.json  -- lightweight sidecar (points, scene_dims)

The session ID is the scan directory name (e.g. 2026-04-26-14-07-06).

Usage:
    python scripts/pix4d_to_api_json.py <scan_dir> --out-dir ../API/static/pointclouds/
    python scripts/pix4d_to_api_json.py <scan_dir> --out-dir . --max-points 150000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np


def _load_opf_format(scan_dir: Path):
    pcl_dir = scan_dir / "point_clouds" / "opf_format"
    with open(pcl_dir / "pcl.gltf") as f:
        gltf = json.load(f)

    acc = gltf["accessors"]
    n_points = acc[0]["count"]

    pos_bv = gltf["bufferViews"][acc[0]["bufferView"]]
    pos_buf = gltf["buffers"][pos_bv["buffer"]]
    offset = pos_bv.get("byteOffset", 0)
    pos_raw = np.frombuffer(
        (pcl_dir / pos_buf["uri"]).read_bytes()[offset: offset + pos_bv["byteLength"]],
        dtype=np.float32,
    ).reshape(n_points, 3)

    col_bv = gltf["bufferViews"][acc[2]["bufferView"]]
    col_buf = gltf["buffers"][col_bv["buffer"]]
    offset = col_bv.get("byteOffset", 0)
    col_raw = np.frombuffer(
        (pcl_dir / col_buf["uri"]).read_bytes()[offset: offset + col_bv["byteLength"]],
        dtype=np.uint8,
    ).reshape(n_points, 4)

    node = gltf["nodes"][0]
    if "matrix" in node:
        m = np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
        ones = np.ones((n_points, 1), dtype=np.float64)
        pts_h = np.hstack([pos_raw.astype(np.float64), ones])
        xyz = (m @ pts_h.T).T[:, :3].astype(np.float32)
    else:
        xyz = pos_raw

    return xyz, col_raw[:, :3]


def _load_legacy(scan_dir: Path):
    legacy = scan_dir / "point_clouds" / "legacy"
    with open(legacy / "pointcloud.gltf") as f:
        gltf = json.load(f)
    n_points = gltf["accessors"][0]["count"]
    stride = gltf["bufferViews"][0]["byteStride"]
    raw = np.frombuffer(
        (legacy / "pointcloud.bin").read_bytes()[: n_points * stride],
        dtype=np.float32,
    ).reshape(n_points, stride // 4)
    return raw[:, :3], (raw[:, 3:6] * 255).clip(0, 255).astype(np.uint8)


def load_pointcloud(scan_dir: Path):
    if (scan_dir / "point_clouds" / "opf_format" / "pcl.gltf").exists():
        print("  Using opf_format point cloud")
        return _load_opf_format(scan_dir)
    if (scan_dir / "point_clouds" / "legacy" / "pointcloud.gltf").exists():
        print("  Using legacy point cloud")
        return _load_legacy(scan_dir)
    raise FileNotFoundError(f"No point cloud found under {scan_dir / 'point_clouds'}")


def export(scan_dir: Path, out_dir: Path, max_points: int) -> None:
    session_id = scan_dir.name
    print(f"\nProcessing: {session_id}")

    xyz, rgb = load_pointcloud(scan_dir)
    n_total = len(xyz)
    print(f"  Points loaded: {n_total:,}")

    if n_total > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, max_points, replace=False)
        idx.sort()
        xyz, rgb = xyz[idx], rgb[idx]
        print(f"  Subsampled to: {len(xyz):,}")

    n_points = len(xyz)
    size = xyz.max(axis=0) - xyz.min(axis=0)
    scene_dims = [round(float(size[0]), 1), round(float(size[1]), 1), round(float(size[2]), 1)]

    pos_b64 = base64.b64encode(xyz.astype(np.float32).tobytes()).decode()
    col_b64 = base64.b64encode(rgb.astype(np.uint8).tobytes()).decode()

    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"{session_id}.json"
    data_path.write_text(json.dumps({
        "session": session_id,
        "points": n_points,
        "scene_dims": scene_dims,
        "pos_b64": pos_b64,
        "col_b64": col_b64,
    }), encoding="utf-8")
    print(f"  Data:  {data_path}  ({data_path.stat().st_size / 1e6:.1f} MB)")

    meta_path = out_dir / f"{session_id}.meta.json"
    meta_path.write_text(json.dumps({
        "points": n_points,
        "scene_dims": scene_dims,
    }, indent=2), encoding="utf-8")
    print(f"  Meta:  {meta_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", help="Pix4DCatch scan directory")
    ap.add_argument("--out-dir", default=".", help="Output directory (default: current dir)")
    ap.add_argument("--max-points", type=int, default=300_000,
                    help="Maximum points to include (default: 300000)")
    args = ap.parse_args()

    if args.max_points < 1:
        print("ERROR: --max-points must be >= 1", file=sys.stderr)
        sys.exit(1)

    scan_dir = Path(args.scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: not a directory: {scan_dir}", file=sys.stderr)
        sys.exit(1)

    export(scan_dir, Path(args.out_dir).resolve(), args.max_points)
    print("\nDone.")


if __name__ == "__main__":
    main()
