"""
planet_scene_pull.py

Pull PlanetScope scenes covering the GPS location embedded in a photo's EXIF data.
Requires: Pillow, requests, python-dateutil

Planet API docs: https://developers.planet.com/docs/apis/data/
"""

import os
import sys
import json
import argparse
import math
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
except ImportError:
    sys.exit("Install Pillow: pip install Pillow")

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    sys.exit("Install python-dateutil: pip install python-dateutil")


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------

def _to_decimal(dms, ref):
    """Convert DMS tuple from EXIF to signed decimal degrees."""
    deg, mn, sec = dms
    deg, mn, sec = float(deg), float(mn), float(sec)
    decimal = deg + mn / 60 + sec / 3600
    if ref in ("S", "W"):
        decimal *= -1
    return decimal


def extract_gps(image_path):
    """
    Return (lat, lon, datetime_utc) from EXIF.
    datetime_utc may be None if no timestamp is embedded.
    """
    img = Image.open(image_path)
    exif_raw = img._getexif()
    if not exif_raw:
        raise ValueError(f"No EXIF data found in {image_path}")

    exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}

    gps_raw = exif.get("GPSInfo")
    if not gps_raw:
        raise ValueError("No GPSInfo tag in EXIF.")

    gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

    lat = _to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
    lon = _to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])

    dt = None
    for tag in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
        raw = exif.get(tag)
        if raw:
            try:
                dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                break
            except ValueError:
                pass

    return lat, lon, dt


# ---------------------------------------------------------------------------
# Planet Data API
# ---------------------------------------------------------------------------

PLANET_API_BASE = "https://api.planet.com/data/v1"


def build_search_filter(lat, lon, radius_m, date_start, date_end, cloud_cover_max=0.20):
    """
    Construct a Planet API AndFilter combining:
      - geometry (point with buffer approximated as bounding box)
      - date range
      - cloud cover <= cloud_cover_max
    """
    import math
    lat_deg = radius_m / 111_320
    lon_deg = radius_m / (111_320 * math.cos(math.radians(lat)))

    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [lon - lon_deg, lat - lat_deg],
            [lon + lon_deg, lat - lat_deg],
            [lon + lon_deg, lat + lat_deg],
            [lon - lon_deg, lat + lat_deg],
            [lon - lon_deg, lat - lat_deg],
        ]]
    }

    return {
        "type": "AndFilter",
        "config": [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": aoi
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": date_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lte": date_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": cloud_cover_max}
            }
        ]
    }


def search_scenes(api_key, lat, lon, radius_m, date_start, date_end,
                  item_types=None, limit=10, cloud_cover_max=0.20):
    """
    Run a quick-search against the Planet Data API.
    Returns a list of scene feature dicts.
    """
    if item_types is None:
        item_types = ["PSScene"]

    filt = build_search_filter(lat, lon, radius_m, date_start, date_end,
                               cloud_cover_max=cloud_cover_max)

    payload = {
        "item_types": item_types,
        "filter": filt,
    }

    url = f"{PLANET_API_BASE}/quick-search"
    resp = requests.post(
        url,
        json=payload,
        auth=(api_key, ""),
        params={"_page_size": limit},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("features", [])


# ---------------------------------------------------------------------------
# Post-search filtering and scoring
# ---------------------------------------------------------------------------

def _parse_month_range(spec):
    """
    Parse 'START-END' month range string into a set of month numbers (1–12).
    Handles wrap-around: '11-2' -> {11, 12, 1, 2}.
    Returns None if spec is None.
    """
    if spec is None:
        return None
    parts = spec.strip().split("-")
    if len(parts) != 2:
        raise ValueError(
            f"--month-range must be START-END (e.g. 5-9 or 11-2), got: {spec!r}"
        )
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"--month-range months must be integers, got: {spec!r}"
        )
    if not (1 <= start <= 12 and 1 <= end <= 12):
        raise ValueError(
            f"--month-range month numbers must be 1–12, got: {spec!r}"
        )
    if start <= end:
        return set(range(start, end + 1))
    # wrap-around (e.g. Nov–Feb)
    return set(range(start, 13)) | set(range(1, end + 1))


def _month_range_from_photo(photo_dt, window=1):
    """
    Derive a ±window-month window from the photo capture month (wrapping).
    """
    months = set()
    for delta in range(-window, window + 1):
        m = ((photo_dt.month - 1 + delta) % 12) + 1
        months.add(m)
    return months


def filter_scenes(scenes, *, month_range=None, sun_elevation_min=None):
    """
    Return a filtered copy of scenes.

    month_range: set of allowed month numbers (1–12), or None to skip
    sun_elevation_min: minimum sun elevation in degrees, or None to skip
    """
    out = []
    for s in scenes:
        props = s.get("properties", {})

        if month_range is not None:
            acquired_str = props.get("acquired", "")
            try:
                acquired_dt = parse_date(acquired_str)
            except Exception:
                continue
            if acquired_dt.month not in month_range:
                continue

        if sun_elevation_min is not None:
            sun_el = props.get("sun_elevation")
            if sun_el is None or float(sun_el) < sun_elevation_min:
                continue

        out.append(s)
    return out


def score_and_sort_scenes(scenes, photo_dt=None):
    """
    Return a sorted copy of scenes.

    Primary key: cloud_cover ascending (lower is better).
    Secondary key: absolute date distance from photo_dt ascending (when available).
    Scenes with missing cloud_cover are ranked last.
    """
    def _key(s):
        props = s.get("properties", {})
        cloud = props.get("cloud_cover")
        cloud_key = float(cloud) if cloud is not None else 1.0

        date_key = 0.0
        if photo_dt is not None:
            acquired_str = props.get("acquired", "")
            try:
                acquired_dt = parse_date(acquired_str)
                if acquired_dt.tzinfo is None:
                    acquired_dt = acquired_dt.replace(tzinfo=timezone.utc)
                # Normalise photo_dt to tz-aware so subtraction never raises
                # TypeError when the caller passes a naive datetime.
                ref_dt = photo_dt if photo_dt.tzinfo is not None \
                    else photo_dt.replace(tzinfo=timezone.utc)
                date_key = abs((acquired_dt - ref_dt).total_seconds())
            except Exception:
                date_key = float("inf")

        return (cloud_key, date_key)

    return sorted(scenes, key=_key)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_scene_summary(scenes):
    if not scenes:
        print("No scenes found matching the criteria.")
        return

    print(f"\nFound {len(scenes)} scene(s) (sorted: cloud cover asc, date proximity asc):\n")
    for i, s in enumerate(scenes, 1):
        props = s.get("properties", {})
        sid = s.get("id", "unknown")
        acquired = props.get("acquired", "unknown")
        cloud = props.get("cloud_cover", "N/A")
        sun_el = props.get("sun_elevation", "N/A")
        satellite = props.get("satellite_id", "N/A")
        item_type = props.get("item_type", "N/A")
        gsd = props.get("gsd", "N/A")

        best_tag = " [best]" if i == 1 else ""
        print(f"  [{i}]{best_tag}")
        print(f"    ID:           {sid}")
        print(f"    Type:         {item_type}")
        print(f"    Acquired:     {acquired}")
        print(f"    Cloud cover:  {cloud:.0%}" if isinstance(cloud, float) else f"    Cloud cover:  {cloud}")
        print(f"    Sun elevation:{sun_el:.1f}°" if isinstance(sun_el, float) else f"    Sun elevation:{sun_el}")
        print(f"    Satellite:    {satellite}")
        print(f"    GSD (m):      {gsd}")
        print()


def save_results(scenes, out_path):
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"Full results saved to {out_path}")


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------

def request_activation(api_key, scene_id, asset_type="ortho_analytic_4b"):
    """Activate a scene asset for download."""
    url = f"{PLANET_API_BASE}/item-types/PSScene/items/{scene_id}/assets"
    assets = requests.get(url, auth=(api_key, ""), timeout=30).json()

    if asset_type not in assets:
        available = list(assets.keys())
        print(f"Asset type '{asset_type}' not available. Available: {available}")
        return None

    activation_url = assets[asset_type]["_links"]["activate"]
    r = requests.post(activation_url, auth=(api_key, ""), timeout=30)
    if r.status_code in (202, 204):
        print(f"Activation requested for {scene_id} / {asset_type}.")
    elif r.status_code == 200:
        print(f"Asset already active.")
    else:
        print(f"Activation returned status {r.status_code}: {r.text}")

    return assets[asset_type]


def download_scene(api_key, scene_id, out_dir, asset_type="ortho_analytic_4b",
                   poll_timeout=300, poll_interval=15):
    """
    Poll activation status and download once ready.
    poll_timeout: maximum seconds to wait (default 300)
    poll_interval: seconds between polls (default 15)
    """
    import time

    url = f"{PLANET_API_BASE}/item-types/PSScene/items/{scene_id}/assets"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
    max_attempts = max(1, math.ceil(poll_timeout / poll_interval))
    print(f"Polling activation for {scene_id} (timeout {poll_timeout}s)...")
    for attempt in range(max_attempts):
        assets = requests.get(url, auth=(api_key, ""), timeout=30).json()
        status = assets.get(asset_type, {}).get("status", "unknown")
        print(f"  [{attempt+1}/{max_attempts}] Status: {status}")

        if status == "active":
            location = assets[asset_type]["location"]
            fname = out_dir / f"{scene_id}_{asset_type}.tif"
            print(f"Downloading to {fname} ...")
            with requests.get(location, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
            print("Download complete.")
            return fname

        if attempt < max_attempts - 1:
            time.sleep(poll_interval)

    print("Asset did not activate within the polling window. Try again later.")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_api_key(raw):
    """
    Accept either a literal API key string or a path to a file containing one.
    """
    if raw is None:
        return None
    p = Path(raw)
    if p.exists() and p.is_file():
        return p.read_text().strip()
    return raw.strip()


def prompt_scene_selection(scenes):
    """
    Print a numbered list of scenes and ask the user to pick one.
    Returns the selected scene dict, or None if the user skips.
    """
    print("Select a scene to download (or press Enter to skip):\n")
    for i, s in enumerate(scenes, 1):
        props = s.get("properties", {})
        acquired = props.get("acquired", "unknown")
        cloud = props.get("cloud_cover", "N/A")
        cloud_str = f"{cloud:.0%}" if isinstance(cloud, float) else str(cloud)
        sun_el = props.get("sun_elevation", "N/A")
        sun_str = f"{sun_el:.1f}°" if isinstance(sun_el, float) else str(sun_el)
        print(f"  [{i}] {s['id']}  acquired={acquired}  cloud={cloud_str}  sun={sun_str}")

    print()
    while True:
        raw = input("Enter number (or Enter to skip): ").strip()
        if raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(scenes):
            return scenes[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(scenes)}.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pull PlanetScope scenes covering the GPS location in a photo."
    )
    p.add_argument("image", help="Path to a geotagged JPEG/TIFF")
    p.add_argument(
        "--api-key",
        default=os.environ.get("PL_API_KEY"),
        help="Planet API key, path to a key file, or set PL_API_KEY env var",
    )
    p.add_argument(
        "--radius", type=float, default=500,
        help="Search radius in metres around the GPS point (default: 500)"
    )
    p.add_argument(
        "--days-before", type=int, default=7,
        help="Search N days before photo capture date (default: 7)"
    )
    p.add_argument(
        "--days-after", type=int, default=7,
        help="Search N days after photo capture date (default: 7)"
    )
    p.add_argument(
        "--date-start",
        help="Override start date (ISO format, e.g. 2024-06-01). "
             "Used when photo has no EXIF timestamp."
    )
    p.add_argument(
        "--date-end",
        help="Override end date (ISO format)."
    )
    p.add_argument(
        "--cloud-cover-max", type=float, default=0.20, metavar="FRAC",
        help="Maximum cloud cover fraction 0–1 (default: 0.20)"
    )
    p.add_argument(
        "--sun-elevation-min", type=float, default=30.0, metavar="DEG",
        help="Minimum sun elevation in degrees; scenes below threshold are dropped "
             "(default: 30.0). Pass 0 to disable."
    )
    p.add_argument(
        "--month-range", default=None, metavar="M-M",
        help="Restrict to scenes acquired in these months, e.g. 5-9 (May–Sep) or "
             "11-2 (Nov–Feb). Omit to auto-derive ±1 month from photo date."
    )
    p.add_argument(
        "--item-types", nargs="+", default=["PSScene"], metavar="TYPE",
        help="Planet item type(s) to search (default: PSScene). "
             "Example: --item-types PSScene SkySatScene"
    )
    p.add_argument(
        "--limit", type=int, default=10,
        help="Max scenes to return from API (default: 10)"
    )
    p.add_argument(
        "--save-json", metavar="FILE",
        help="Save full scene metadata to a JSON file"
    )
    p.add_argument(
        "--download", metavar="SCENE_ID",
        help="Activate and download a specific scene ID without prompting"
    )
    _download_mode = p.add_mutually_exclusive_group()
    _download_mode.add_argument(
        "--auto-best", action="store_true",
        help="After filtering and ranking, automatically download the best scene "
             "(lowest cloud cover, closest date)"
    )
    _download_mode.add_argument(
        "--interactive", action="store_true",
        help="After searching, prompt to select a scene for download"
    )
    p.add_argument(
        "--download-dir", default="./planet_downloads",
        help="Directory for downloaded scenes (default: ./planet_downloads)"
    )
    p.add_argument(
        "--asset-type", default="ortho_analytic_4b",
        help="Planet asset type to download (default: ortho_analytic_4b)"
    )
    p.add_argument(
        "--poll-timeout", type=int, default=300, metavar="SEC",
        help="Maximum seconds to wait for asset activation (default: 300)"
    )
    p.add_argument(
        "--poll-interval", type=int, default=15, metavar="SEC",
        help="Seconds between activation status polls (default: 15)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key:
        sys.exit(
            "No API key found. Pass --api-key <key or file>, or set PL_API_KEY.\n"
            "Sign up at https://www.planet.com/account"
        )
    args.api_key = api_key

    # --- Extract GPS from photo ---
    print(f"Reading EXIF from: {args.image}")
    lat, lon, photo_dt = extract_gps(args.image)
    print(f"  GPS:  {lat:.6f}, {lon:.6f}")
    if photo_dt:
        print(f"  Date: {photo_dt.isoformat()}")
    else:
        print("  Date: not found in EXIF")

    # --- Resolve date range ---
    if args.date_start:
        date_start = parse_date(args.date_start).replace(tzinfo=timezone.utc)
    elif photo_dt:
        date_start = photo_dt - timedelta(days=args.days_before)
    else:
        sys.exit(
            "No capture date in EXIF and no --date-start provided. "
            "Supply --date-start YYYY-MM-DD to continue."
        )

    if args.date_end:
        date_end = parse_date(args.date_end).replace(tzinfo=timezone.utc)
    elif photo_dt:
        date_end = photo_dt + timedelta(days=args.days_after)
    else:
        date_end = date_start + timedelta(days=args.days_before + args.days_after)

    print(f"  Search window: {date_start.date()} to {date_end.date()}")
    print(f"  Radius: {args.radius} m")
    print(f"  Cloud cover max: {args.cloud_cover_max:.0%}")

    # --- Resolve month range ---
    try:
        month_range = _parse_month_range(args.month_range)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    if month_range is None and photo_dt is not None:
        month_range = _month_range_from_photo(photo_dt, window=1)
        month_names = sorted(month_range)
        print(f"  Month filter (auto ±1 month): {month_names}")
    elif month_range is not None:
        print(f"  Month filter (explicit): {sorted(month_range)}")
    else:
        print("  Month filter: none")

    # --- Resolve sun elevation ---
    sun_elevation_min = args.sun_elevation_min if args.sun_elevation_min > 0 else None
    if sun_elevation_min is not None:
        print(f"  Sun elevation min: {sun_elevation_min}°")
    else:
        print("  Sun elevation filter: disabled")

    print()

    # --- Search ---
    print("Querying Planet Data API...")
    scenes = search_scenes(
        api_key=args.api_key,
        lat=lat,
        lon=lon,
        radius_m=args.radius,
        date_start=date_start,
        date_end=date_end,
        item_types=args.item_types,
        limit=args.limit,
        cloud_cover_max=args.cloud_cover_max,
    )
    print(f"  API returned {len(scenes)} scene(s) within search window.")

    # --- Filter ---
    scenes = filter_scenes(scenes, month_range=month_range,
                           sun_elevation_min=sun_elevation_min)
    if month_range is not None or sun_elevation_min is not None:
        print(f"  {len(scenes)} scene(s) after month/sun filters.")

    # --- Sort ---
    scenes = score_and_sort_scenes(scenes, photo_dt=photo_dt)

    print_scene_summary(scenes)

    if args.save_json:
        save_results(scenes, args.save_json)

    # --- Download: explicit ID > auto-best > interactive > skip ---
    scene_to_download = None

    if args.download:
        scene_to_download = args.download
    elif args.auto_best:
        if scenes:
            scene_to_download = scenes[0]["id"]
            print(f"--auto-best: selecting {scene_to_download}")
        else:
            print("--auto-best: no scenes available to download.")
    elif args.interactive and scenes:
        selected = prompt_scene_selection(scenes)
        if selected:
            scene_to_download = selected["id"]

    if scene_to_download:
        request_activation(args.api_key, scene_to_download, args.asset_type)
        download_scene(
            args.api_key, scene_to_download, args.download_dir, args.asset_type,
            poll_timeout=args.poll_timeout, poll_interval=args.poll_interval,
        )


if __name__ == "__main__":
    main()
