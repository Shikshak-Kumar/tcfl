import httpx
import json
import asyncio

# Global semaphore placeholder to be lazily initialized on the running event loop
_overpass_semaphore = None

def _get_semaphore():
    global _overpass_semaphore
    if _overpass_semaphore is None:
        _overpass_semaphore = asyncio.Semaphore(1)
    return _overpass_semaphore


async def get_osm_pois(lat: float, lon: float, radius_deg: float = 0.05) -> dict:
    """
    Fetches Points of Interest (POIs) from OpenStreetMap via the Overpass API
    within a bounding box defined by (lat, lon) and radius_deg.
    Returns a summarized count of key POI categories.
    """

    # Calculate bounding box
    min_lat = lat - radius_deg
    max_lat = lat + radius_deg
    min_lon = lon - radius_deg
    max_lon = lon + radius_deg

    overpass_url = "https://overpass-api.de/api/interpreter"

    # Query for standard traffic generators/attractors
    query = f"""
    [out:json][timeout:50];
    (
      nwr["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["leisure"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["shop"]({min_lat},{min_lon},{max_lat},{max_lon});
      nwr["office"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center tags;
    """

    try:
        # Increased timeout to 60s to handle slow Overpass responses
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Retry logic for 429 (Rate Limit) and 504 (Gateway Timeout)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await client.post(overpass_url, data={"data": query})
                    
                    if response.status_code in [429, 504]:
                        wait_time = (attempt * 3) + 2  # 2, 5, 8 seconds
                        reason = "Rate limited" if response.status_code == 429 else "Gateway Timeout"
                        print(f"[OSM API] {reason} (code {response.status_code}). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    break
                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    print(f"[OSM API] Timeout error: {e}. Retry {attempt+1}/{max_retries}...")
                    await asyncio.sleep(2)
                    continue
            else:
                print(f"[OSM API] Failed after {max_retries} retries.")
                return {}

        # Categorize the results and store coordinates
        poi_summary = {
            "healthcare": [],  # Hospitals, clinics, pharmacies
            "education": [],  # Schools, universities
            "commercial": [],  # Shops, malls, markets
            "office": [],  # Business districts, offices
            "leisure": [],  # Parks, stadiums, entertainment
            "food_dining": [],  # Restaurants, cafes, fast_food
            "public_service": [],  # Police, fire, civic
        }

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            poi_lat = element.get("lat", 0.0)
            poi_lon = element.get("lon", 0.0)

            if poi_lat == 0.0 and poi_lon == 0.0:
                center = element.get("center", {})
                poi_lat = center.get("lat", 0.0)
                poi_lon = center.get("lon", 0.0)

            if poi_lat == 0.0 and poi_lon == 0.0:
                continue

            poi_data = {
                "lat": poi_lat,
                "lon": poi_lon,
                "name": tags.get("name", "Unknown"),
            }

            amenity = tags.get("amenity", "")
            shop = tags.get("shop", "")
            leisure = tags.get("leisure", "")
            office = tags.get("office", "")

            if amenity in ["hospital", "clinic", "doctors", "pharmacy", "dentist"]:
                poi_summary["healthcare"].append(poi_data)
            elif amenity in ["school", "university", "college", "kindergarten"]:
                poi_summary["education"].append(poi_data)
            elif amenity in ["restaurant", "cafe", "fast_food", "bar", "pub", "food_court"]:
                poi_summary["food_dining"].append(poi_data)
            elif amenity in ["police", "fire_station", "courthouse", "townhall", "library"]:
                poi_summary["public_service"].append(poi_data)
            elif shop:
                poi_summary["commercial"].append(poi_data)
            elif office:
                poi_summary["office"].append(poi_data)
            elif leisure or amenity in ["cinema", "theatre", "stadium", "sports_centre", "park"]:
                poi_summary["leisure"].append(poi_data)

        return poi_summary

    except Exception as e:
        print(f"[OSM API] Failed to fetch data: {e}")
        return {}


async def detect_priority_for_intersection(
    lat: float, lon: float, radius_km: float = 1.0
) -> dict:
    """
    Query OSM within radius_km of (lat, lon) and assign a priority tier.
    """
    radius_deg = radius_km / 111.0
    pois = await get_osm_pois(lat, lon, radius_deg=radius_deg)

    # Check for Tier 1 — Healthcare
    healthcare = pois.get("healthcare", [])
    if healthcare:
        return {
            "tier": 1,
            "tier_label": "Hospital Priority",
            "tier_emoji": "🏥",
            "detected_pois": [p.get("name", "Unknown") for p in healthcare[:5]],
            "poi_count": len(healthcare),
            "all_pois": pois,
        }

    # Check for Tier 2 — Education
    education = pois.get("education", [])
    if education:
        return {
            "tier": 2,
            "tier_label": "School Priority",
            "tier_emoji": "🏫",
            "detected_pois": [p.get("name", "Unknown") for p in education[:5]],
            "poi_count": len(education),
            "all_pois": pois,
        }

    # Tier 3 — Normal
    return {
        "tier": 3,
        "tier_label": "Normal",
        "tier_emoji": "📍",
        "detected_pois": [],
        "poi_count": 0,
        "all_pois": pois,
    }


async def detect_pois_for_intersections(
    intersections: list, radius_km: float = 1.0, check_cancellation=None
) -> list:
    """
    For a list of intersections, detect POI priority for each in parallel.
    check_cancellation: optional async function that returns True if we should stop.
    """
    tasks = []
    sem = _get_semaphore()

    for ix in intersections:

        async def wrapped_detect(lat, lon, r):
            if check_cancellation and await check_cancellation():
                return None
            async with sem:
                if check_cancellation and await check_cancellation():
                    return None
                return await detect_priority_for_intersection(lat, lon, r)

        tasks.append(wrapped_detect(ix["lat"], ix["lon"], radius_km))

    priorities = await asyncio.gather(*tasks)

    results = []
    for ix, priority in zip(intersections, priorities):
        if priority is None:
            continue
        results.append({**ix, **priority})
        print(
            f"[OSM] {ix.get('name', 'Pin')}: Tier {priority['tier']} ({priority['tier_label']}) — "
            f"{priority['poi_count']} POIs detected"
        )
    return results


if __name__ == "__main__":
    import argparse
    from utils.tomtom_api import CITY_COORDINATES

    parser = argparse.ArgumentParser(description="Test OSM Overpass API Endpoint")
    parser.add_argument(
        "--city",
        type=str,
        default="Delhi",
        choices=list(CITY_COORDINATES.keys()),
        help="City to test",
    )
    parser.add_argument("--lat", type=float, help="Latitude override")
    parser.add_argument("--lon", type=float, help="Longitude override")
    parser.add_argument("--radius", type=float, default=0.05, help="Radius in degrees")

    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        print(
            f"Testing OSM POI fetch for Custom Coordinates ({lat}, {lon}) with radius {args.radius}..."
        )
    else:
        lat, lon = CITY_COORDINATES[args.city]
        print(
            f"Testing OSM POI fetch for {args.city} ({lat}, {lon}) with radius {args.radius}..."
        )

    pois = get_osm_pois(lat, lon, radius_deg=args.radius)
    print("\nPOI Summary:")
    print(json.dumps(pois, indent=2))
