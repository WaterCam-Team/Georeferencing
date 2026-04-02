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
    # Pillow may return IFDRational objects; coerce to float
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

    # Try to get capture datetime from standard EXIF
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


def build_search_filter(lat, lon, radius_m, date_start, date_end):
    """
    Construct a Planet API AndFilter combining:
      - geometry (point with buffer approximated as bounding box)
      - date range
      - cloud cover < 20%
    """
    # Rough degree offset for radius_m at the given latitude
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
                "config": {"lte": 0.20}
            }
        ]
    }


def search_scenes(api_key, lat, lon, radius_m, date_start, date_end,
                  item_types=None, limit=10):
    """
    Run a quick-search against the Planet Data API.
    Returns a list of scene feature dicts.
    """
    if item_types is None:
        item_types = ["PSScene"]  # PlanetScope 4-/8-band

    filt = build_search_filter(lat, lon, radius_m, date_start, date_end)

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


def print_scene_summary(scenes):
    if not scenes:
        print("No scenes found matching the criteria.")
        return

    print(f"\nFound {len(scenes)} scene(s):\n")
    for s in scenes:
        props = s.get("properties", {})
        sid = s.get("id", "unknown")
        acquired = props.get("acquired", "unknown")
        cloud = props.get("cloud_cover", "N/A")
        satellite = props.get("satellite_id", "N/A")
        item_type = props.get("item_type", "N/A")
        gsd = props.get("gsd", "N/A")

        print(f"  ID:          {sid}")
        print(f"  Type:        {item_type}")
        print(f"  Acquired:    {acquired}")
        print(f"  Cloud cover: {cloud:.0%}" if isinstance(cloud, float) else f"  Cloud cover: {cloud}")
        print(f"  Satellite:   {satellite}")
        print(f"  GSD (m):     {gsd}")
        print()


def save_results(scenes, out_path):
    with open(out_path, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"Full results saved to {out_path}")


# ---------------------------------------------------------------------------
# Asset download (optional)
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


def download_scene(api_key, scene_id, out_dir, asset_type="ortho_analytic_4b"):
    """
    Poll activation status and download once ready.
    Planet assets typically activate within 1-5 minutes for recent imagery.
    """
    import time

    url = f"{PLANET_API_BASE}/item-types/PSScene/items/{scene_id}/assets"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Polling activation for {scene_id}...")
    for attempt in range(20):
        assets = requests.get(url, auth=(api_key, ""), timeout=30).json()
        status = assets.get(asset_type, {}).get("status", "unknown")
        print(f"  [{attempt+1}/20] Status: {status}")

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

        time.sleep(15)

    print("Asset did not activate within the polling window. Try again later.")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_api_key(raw):
    """
    Accept either a literal API key string or a path to a file containing one.
    Strips whitespace and newlines either way.
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
        print(f"  [{i}] {s['id']}  acquired={acquired}  cloud={cloud_str}")

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
        "--limit", type=int, default=10,
        help="Max scenes to return (default: 10)"
    )
    p.add_argument(
        "--save-json", metavar="FILE",
        help="Save full scene metadata to a JSON file"
    )
    p.add_argument(
        "--download", metavar="SCENE_ID",
        help="Activate and download a specific scene ID without prompting"
    )
    p.add_argument(
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
    print(f"  Radius: {args.radius} m\n")

    # --- Search ---
    print("Querying Planet Data API...")
    scenes = search_scenes(
        api_key=args.api_key,
        lat=lat,
        lon=lon,
        radius_m=args.radius,
        date_start=date_start,
        date_end=date_end,
        limit=args.limit,
    )

    print_scene_summary(scenes)

    if args.save_json:
        save_results(scenes, args.save_json)

    # --- Download: explicit ID, interactive selection, or skip ---
    scene_to_download = None

    if args.download:
        scene_to_download = args.download
    elif args.interactive and scenes:
        selected = prompt_scene_selection(scenes)
        if selected:
            scene_to_download = selected["id"]

    if scene_to_download:
        request_activation(args.api_key, scene_to_download, args.asset_type)
        download_scene(args.api_key, scene_to_download, args.download_dir, args.asset_type)


if __name__ == "__main__":
    main()
