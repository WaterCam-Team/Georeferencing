"""
query_syr_lampposts.py

Queries the City of Syracuse Open Data Portal (data.syr.gov) for city-owned
lamppost and streetlight pole data, filtering for dedicated (non-power-pole)
city infrastructure suitable for UFO-Net sensor mounting.

The portal runs on Socrata (now Tyler Technologies). Two access methods are tried:
  1. Socrata Discovery API  -- searches the catalog for relevant datasets
  2. Direct dataset fetch   -- if a known dataset ID is supplied or discovered

As of April 2026, a dedicated streetlight pole layer has NOT been confirmed as
publicly available on data.syr.gov. This script will report what it finds and
fall back gracefully if nothing is present. If the dataset becomes available,
or if you obtain a dataset ID from DPW directly, supply it via --dataset-id.

Usage:
    # Search catalog and auto-discover datasets:
    python query_syr_lampposts.py

    # Fetch a known dataset by Socrata 4x4 ID and filter to a bounding box:
    python query_syr_lampposts.py --dataset-id XXXX-XXXX --bbox 43.03,43.09,-76.20,-76.10

    # Save output to GeoJSON:
    python query_syr_lampposts.py --dataset-id XXXX-XXXX --output poles.geojson

Requirements: requests (pip install requests)
"""

import argparse
import json
import sys
import requests

DOMAIN = "data.syr.gov"
CATALOG_URL = f"https://api.us.socrata.com/api/catalog/v1"
SODA_BASE = f"https://{DOMAIN}/resource"

# Keywords to match against dataset names and descriptions in the catalog
SEARCH_TERMS = [
    "streetlight",
    "street light",
    "lamppost",
    "lamp post",
    "light pole",
    "lighting",
    "smart light",
    "LED pole",
    "vertical asset",
]

# Field names commonly used in streetlight datasets across US open data portals.
# The script will try these in order when looking for pole-type / ownership fields.
POLE_TYPE_FIELDS = [
    "pole_type", "poletype", "support_type", "mount_type",
    "structure_type", "asset_type", "ownership", "owner",
    "install_type", "light_type",
]

# Values that indicate a dedicated city-owned standard (not a shared wood pole)
CITY_OWNED_VALUES = [
    "city", "municipal", "dedicated", "standard", "mast arm",
    "concrete", "steel", "aluminum", "decorative", "pedestrian",
    "park", "owned",
]

# Values that indicate a shared National Grid wood pole -- exclude these
EXCLUDE_VALUES = [
    "wood", "wooden", "utility", "national grid", "power pole",
    "joint", "shared", "co-located", "collocated",
]


def search_catalog(query_term: str) -> list[dict]:
    """Search the Socrata catalog for datasets matching a term on data.syr.gov."""
    params = {
        "domains": DOMAIN,
        "q": query_term,
        "limit": 20,
    }
    try:
        r = requests.get(CATALOG_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except requests.RequestException as e:
        print(f"  Catalog search error for '{query_term}': {e}", file=sys.stderr)
        return []


def fetch_dataset(dataset_id: str, soql_where: str = None,
                  limit: int = 50000) -> list[dict]:
    """Fetch rows from a Socrata dataset by its 4x4 identifier."""
    url = f"{SODA_BASE}/{dataset_id}.json"
    params = {"$limit": limit}
    if soql_where:
        params["$where"] = soql_where
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"  Dataset fetch error ({dataset_id}): {e}", file=sys.stderr)
        return []


def bbox_to_soql(bbox_str: str) -> str | None:
    """
    Convert a 'lat_min,lat_max,lon_min,lon_max' string to a Socrata
    within_box() SoQL filter.  Tries common geometry column names.
    """
    try:
        lat_min, lat_max, lon_min, lon_max = [float(x) for x in bbox_str.split(",")]
    except ValueError:
        print("Bad --bbox format. Expected: lat_min,lat_max,lon_min,lon_max",
              file=sys.stderr)
        return None
    # Socrata within_box(geom_col, nwLat, nwLon, seLat, seLon)
    for col in ("location", "point", "geo_point", "geometry", "the_geom", "geom"):
        return (
            f"within_box({col}, {lat_max}, {lon_min}, {lat_min}, {lon_max})"
        )


def is_city_owned(row: dict) -> bool:
    """
    Return True if the row appears to represent a city-owned dedicated
    light standard rather than a shared National Grid wood pole.

    Strategy:
      - If a pole_type / ownership field exists, check its value.
      - If no such field exists, accept the row (can't tell, don't exclude).
    """
    for field in POLE_TYPE_FIELDS:
        val = row.get(field, "")
        if not val:
            continue
        val_lower = str(val).lower()
        # Explicit exclusion
        if any(excl in val_lower for excl in EXCLUDE_VALUES):
            return False
        # Explicit inclusion
        if any(inc in val_lower for inc in CITY_OWNED_VALUES):
            return True
    # No pole-type field found -- return True (include with caveat)
    return True


def rows_to_geojson(rows: list[dict]) -> dict:
    """Convert a list of Socrata rows to a GeoJSON FeatureCollection."""
    features = []
    for row in rows:
        # Try to find coordinates -- Socrata stores them in several ways
        coords = None
        for loc_field in ("location", "point", "geo_point", "the_geom", "geom"):
            loc = row.get(loc_field)
            if isinstance(loc, dict):
                # Socrata Point type: {"type": "Point", "coordinates": [lon, lat]}
                if loc.get("type") == "Point":
                    coords = loc["coordinates"]  # [lon, lat]
                    break
                # Socrata location object: {"latitude": "...", "longitude": "..."}
                lat = loc.get("latitude") or loc.get("lat")
                lon = loc.get("longitude") or loc.get("lon") or loc.get("lng")
                if lat and lon:
                    coords = [float(lon), float(lat)]
                    break

        if not coords:
            # Try flat lat/lon columns
            lat = row.get("latitude") or row.get("lat") or row.get("y")
            lon = row.get("longitude") or row.get("lon") or row.get("lng") or row.get("x")
            if lat and lon:
                try:
                    coords = [float(lon), float(lat)]
                except ValueError:
                    pass

        geometry = (
            {"type": "Point", "coordinates": coords} if coords else None
        )
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": row,
        })

    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(
        description="Query data.syr.gov for city-owned lamppost/streetlight data."
    )
    parser.add_argument(
        "--dataset-id",
        help="Socrata 4x4 dataset ID (e.g. ab12-cd34). "
             "If omitted, the catalog is searched automatically.",
    )
    parser.add_argument(
        "--bbox",
        help="Bounding box filter: lat_min,lat_max,lon_min,lon_max. "
             "Example: 43.03,43.09,-76.20,-76.10",
    )
    parser.add_argument(
        "--output",
        help="Output file path. Supports .geojson or .json (default: print to stdout).",
    )
    parser.add_argument(
        "--all-poles",
        action="store_true",
        help="Return all poles without filtering for city-owned dedicated standards.",
    )
    args = parser.parse_args()

    dataset_id = args.dataset_id

    # --- Step 1: Discover dataset if no ID provided ---
    if not dataset_id:
        print(f"Searching data.syr.gov catalog for streetlight/lamppost datasets...")
        found = {}
        for term in SEARCH_TERMS:
            results = search_catalog(term)
            for item in results:
                uid = item.get("resource", {}).get("id")
                name = item.get("resource", {}).get("name", "")
                desc = item.get("resource", {}).get("description", "")
                if uid and uid not in found:
                    found[uid] = {"name": name, "description": desc}
                    print(f"  Found: [{uid}] {name}")

        if not found:
            print(
                "\nNo streetlight or lamppost datasets found on data.syr.gov.\n"
                "\nThis is expected as of April 2026 -- the city's streetlight\n"
                "inventory from the 2019-2020 smart lighting project does not\n"
                "appear to be published as open data.\n"
                "\nNext steps:\n"
                "  1. Contact Syracuse DPW / Office of Digital Services and\n"
                "     request the streetlight asset inventory (pole type per\n"
                "     location). Cite the 2019-2020 smart lighting project.\n"
                "  2. If a dataset ID is provided, re-run with --dataset-id.\n"
                "  3. Alternatively, check the Onondaga County GIS portal:\n"
                "     https://spatial.vhb.com/onondaga/\n"
            )
            sys.exit(0)

        # If exactly one dataset, use it; if multiple, prompt
        if len(found) == 1:
            dataset_id = list(found.keys())[0]
            print(f"\nUsing dataset: {dataset_id}")
        else:
            print(f"\nMultiple datasets found. Use --dataset-id to select one:")
            for uid, meta in found.items():
                print(f"  {uid}: {meta['name']}")
            sys.exit(0)

    # --- Step 2: Fetch the dataset ---
    print(f"\nFetching dataset {dataset_id} from {DOMAIN}...")
    soql_where = bbox_to_soql(args.bbox) if args.bbox else None
    rows = fetch_dataset(dataset_id, soql_where=soql_where)

    if not rows:
        print("No rows returned. The dataset may be empty or the query failed.")
        sys.exit(1)

    print(f"Total rows fetched: {len(rows)}")

    # --- Step 3: Filter for city-owned dedicated standards ---
    if args.all_poles:
        filtered = rows
        print("Returning all poles (--all-poles flag set, no ownership filter applied).")
    else:
        filtered = [r for r in rows if is_city_owned(r)]
        excluded = len(rows) - len(filtered)
        print(f"After filtering for city-owned dedicated standards: {len(filtered)} poles "
              f"({excluded} excluded as likely wood/shared poles).")

        # Report which fields were used for filtering
        type_fields_present = [
            f for f in POLE_TYPE_FIELDS if any(f in row for row in rows)
        ]
        if type_fields_present:
            print(f"Pole-type fields found in dataset: {type_fields_present}")
        else:
            print(
                "WARNING: No pole-type or ownership field found in dataset.\n"
                "         All poles returned -- manual verification still needed.\n"
                "         Common fields to look for: " + ", ".join(POLE_TYPE_FIELDS)
            )

    # --- Step 4: Output ---
    geojson = rows_to_geojson(filtered)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"Output written to {args.output}")
    else:
        print(json.dumps(geojson, indent=2))


if __name__ == "__main__":
    main()
