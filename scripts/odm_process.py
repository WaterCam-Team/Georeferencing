"""
odm_process.py
==============
Submit a directory of drone images to a NodeODM REST API, poll for completion,
and download the DSM + orthophoto GeoTIFFs.

After downloading, prints the `drone_dsm_ingest.py` command to run next.

NodeODM can be run locally with Docker:
    docker run -p 3000:3000 opendronemap/nodeodm

Usage
-----
    python scripts/odm_process.py images/ \\
        [--url http://localhost:3000] \\
        [--out-dir ./odm_output] \\
        [--options dsm=true orthophoto-resolution=3 dem-resolution=3] \\
        [--poll-interval 30] \\
        [--poll-timeout 3600]

Requires: requests (pip install requests)
No additional dependencies beyond what planet_scene_pull.py already needs.
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
import time
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import requests
except ImportError:
    sys.exit("Install requests: pip install requests")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

_DEFAULT_OPTIONS = {
    "dsm": "true",
    "orthophoto-resolution": "3",
    "dem-resolution": "3",
}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"


# ---------------------------------------------------------------------------
# NodeODM REST helpers
# ---------------------------------------------------------------------------

def _api(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def create_task(base_url: str, options: dict, name: str = "drone") -> str:
    """Create a new task and return its UUID."""
    resp = requests.post(
        _api(base_url, "/task/new/init"),
        json={"name": name, "options": [{"name": k, "value": v} for k, v in options.items()]},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("uuid") or data.get("id")
    if not task_id:
        raise RuntimeError(f"NodeODM did not return a task ID: {data}")
    return task_id


def upload_images(base_url: str, task_id: str, image_paths: list[Path]) -> None:
    """Upload image files to the task."""
    total = len(image_paths)
    url = _api(base_url, f"/task/new/upload/{task_id}")
    for i, img in enumerate(image_paths, 1):
        print(f"  Uploading [{i}/{total}] {img.name} ...", end="\r", flush=True)
        content_type, _ = mimetypes.guess_type(img.name)
        if not content_type:
            content_type = "application/octet-stream"
        with open(img, "rb") as f:
            resp = requests.post(
                url,
                files={"images": (img.name, f, content_type)},
                timeout=120,
            )
            resp.raise_for_status()
    print(f"  Uploaded {total} image(s).{' ' * 20}")


def commit_task(base_url: str, task_id: str) -> None:
    """Signal NodeODM to start processing."""
    resp = requests.post(
        _api(base_url, f"/task/new/commit/{task_id}"),
        timeout=30,
    )
    resp.raise_for_status()


def poll_task(base_url: str, task_id: str,
              poll_interval: int, poll_timeout: int) -> bool:
    """
    Poll task status until completed or failed.
    Returns True on success, False on failure.
    """
    url = _api(base_url, f"/task/{task_id}/info")
    deadline = time.monotonic() + poll_timeout
    last_pct = -1

    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            info = resp.json()
        except Exception as exc:
            print(f"\n  [WARN] Poll error: {exc}")
            time.sleep(poll_interval)
            continue

        status_code = info.get("status", {}).get("code", 0)
        pct = info.get("progress", 0)

        # status codes: 10=queued, 20=running, 30=failed, 40=completed
        if pct != last_pct:
            print(f"  Progress: {pct:.0f}%  (status {status_code})   ", end="\r", flush=True)
            last_pct = pct

        if status_code == 40:
            print(f"\n  {PASS}  Processing complete ({pct:.0f}%)")
            return True
        if status_code == 30:
            print(f"\n  {FAIL}  Task failed: {info.get('status', {}).get('msg', 'unknown')}")
            return False

        time.sleep(poll_interval)

    print(f"\n  {FAIL}  Timed out after {poll_timeout}s")
    return False


def download_asset(base_url: str, task_id: str,
                   asset_name: str, out_path: Path) -> bool:
    """
    Download a named NodeODM asset into out_path.

    NodeODM returns either a ZIP archive (directory assets like orthophoto,
    odm_dem) or a direct file depending on version and asset type.  This
    function handles both: it downloads to a temp file, then tries ZIP
    extraction; if the response is not a valid ZIP it is treated as a direct
    GeoTIFF and moved to out_path unchanged.
    """
    url = _api(base_url, f"/task/{task_id}/download/{asset_name}")
    print(f"  Downloading {asset_name} ...", end=" ", flush=True)

    tmp_path = out_path.parent / f"{out_path.stem}.tmp"
    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
    except Exception as exc:
        print(f"FAILED ({exc})")
        tmp_path.unlink(missing_ok=True)
        return False

    # Try ZIP first; fall back to treating the download as a direct .tif
    if zipfile.is_zipfile(tmp_path):
        try:
            with zipfile.ZipFile(tmp_path) as zf:
                tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                if not tif_names:
                    print(f"FAILED (no .tif inside ZIP)")
                    tmp_path.unlink(missing_ok=True)
                    return False
                with zf.open(tif_names[0]) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"FAILED (ZIP extract: {exc})")
            tmp_path.unlink(missing_ok=True)
            return False
    else:
        # Direct file response — replace final path atomically (works on Windows too)
        tmp_path.replace(out_path)

    print(f"→ {out_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_options(raw: list[str]) -> dict:
    """Parse 'key=value' strings into a dict."""
    opts = dict(_DEFAULT_OPTIONS)
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"--options items must be key=value, got: {item!r}"
            )
        k, _, v = item.partition("=")
        opts[k.strip()] = v.strip()
    return opts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit drone images to NodeODM and download DSM + orthophoto."
    )
    p.add_argument("images_dir", help="Directory containing drone JPEG/TIFF images")
    p.add_argument(
        "--url", default="http://localhost:3000",
        help="NodeODM base URL (default: http://localhost:3000)"
    )
    p.add_argument(
        "--out-dir", default="./odm_output",
        help="Directory for downloaded DSM and orthophoto (default: ./odm_output)"
    )
    p.add_argument(
        "--options", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="ODM task options as key=value pairs "
             "(default: dsm=true orthophoto-resolution=3 dem-resolution=3)"
    )
    p.add_argument(
        "--poll-interval", type=int, default=30, metavar="SEC",
        help="Seconds between status polls (default: 30)"
    )
    p.add_argument(
        "--poll-timeout", type=int, default=3600, metavar="SEC",
        help="Maximum seconds to wait for processing (default: 3600)"
    )
    p.add_argument(
        "--name", default="drone",
        help="Task name shown in NodeODM UI (default: drone)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"[ERR] images directory not found: {images_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not image_paths:
        print(f"[ERR] No image files found in {images_dir}", file=sys.stderr)
        return 2

    print(f"{INFO}  Found {len(image_paths)} image(s) in {images_dir}")

    try:
        options = parse_options(args.options)
    except ValueError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 2
    print(f"{INFO}  ODM options: {options}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Create task ---
    print(f"\nConnecting to NodeODM at {args.url} ...")
    try:
        task_id = create_task(args.url, options, name=args.name)
    except Exception as exc:
        print(f"[ERR] Could not create task: {exc}", file=sys.stderr)
        print("      Is NodeODM running?  docker run -p 3000:3000 opendronemap/nodeodm",
              file=sys.stderr)
        return 1

    print(f"  Task ID: {task_id}")

    # --- Upload ---
    print(f"\nUploading images ...")
    try:
        upload_images(args.url, task_id, image_paths)
    except Exception as exc:
        print(f"[ERR] Upload failed: {exc}", file=sys.stderr)
        return 1

    # --- Commit ---
    print("Starting processing ...")
    try:
        commit_task(args.url, task_id)
    except Exception as exc:
        print(f"[ERR] Commit failed: {exc}", file=sys.stderr)
        return 1

    # --- Poll ---
    print(f"Polling (interval {args.poll_interval}s, timeout {args.poll_timeout}s) ...")
    ok = poll_task(args.url, task_id, args.poll_interval, args.poll_timeout)
    if not ok:
        return 1

    # --- Download ---
    print("\nDownloading outputs ...")
    dsm_path = out_dir / "dsm.tif"
    ortho_path = out_dir / "orthophoto.tif"

    dsm_ok = download_asset(args.url, task_id, "odm_dem", dsm_path)
    ortho_ok = download_asset(args.url, task_id, "orthophoto", ortho_path)

    print()
    if dsm_ok:
        ingest_script = _REPO_ROOT / "scripts" / "drone_dsm_ingest.py"
        print(f"Next step — ingest the DSM:")
        print(f"  python {ingest_script} {dsm_path} --out-dir {out_dir}")
    if ortho_ok:
        gcp_script = _REPO_ROOT / "drone_gcp_match.py"
        print(f"\nOptional — generate GCPs from the orthophoto:")
        print(f"  python {gcp_script} \\")
        print(f"      --field-image <your_field_photo.jpg> \\")
        print(f"      --ortho-tif {ortho_path} \\")
        print(f"      --output-csv ./drone_gcps.csv")

    return 0 if (dsm_ok or ortho_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
