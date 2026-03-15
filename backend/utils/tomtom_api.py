import urllib.request
import json
import ssl
import os
import time
from typing import Dict, Tuple, Optional, List

# In-memory cache for API responses to avoid rate limits
_flow_cache = {}  # key: (lat, lon), value: (timestamp, data)
_incidents_cache = {}  # key: (lat, lon), value: (timestamp, data)
CACHE_TTL = 300  # 5 minutes


def get_api_key() -> str:
    """Retrieves the TomTom API key from the environment variable."""
    api_key = os.environ.get("TOMTOM_API_KEY")
    if not api_key:
        # Try loading .env if not found in environment
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("TOMTOM_API_KEY")
        except ImportError:
            pass
            
    if not api_key:
        raise ValueError("TOMTOM_API_KEY environment variable is missing. Please set it in your .env file or deployment settings.")
    return api_key


def get_real_time_flow(
    api_key: str, lat: float, lon: float
) -> Optional[Dict[str, float]]:
    """
    Fetches real-time traffic flow data from the TomTom API for a given location.

    Args:
        api_key: TomTom API key
        lat: Latitude of the point
        lon: Longitude of the point

    Returns:
        A dictionary containing traffic metrics like currentSpeed, freeFlowSpeed,
        and congestion_factor, or None if the request failed.
    """
    # Check cache first
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.time()
    if cache_key in _flow_cache:
        cached_time, cached_data = _flow_cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={api_key}&point={lat},{lon}"

    # Create an unverified SSL context if needed (common in some local dev environments)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

            if "flowSegmentData" in data:
                flow_data = data["flowSegmentData"]
                current_speed = flow_data.get("currentSpeed", 1.0)
                free_flow_speed = flow_data.get("freeFlowSpeed", 1.0)

                # Prevent division by zero
                current_speed = max(1.0, current_speed)
                free_flow_speed = max(1.0, free_flow_speed)

                # Congestion factor: How much slower is the traffic compared to free flow?
                # 1.0 = free flow, >1.0 = congested
                # We cap it at 3.0 to prevent absurd simulation states
                congestion_factor = min(3.0, free_flow_speed / current_speed)

                result = {
                    "currentSpeed": current_speed,
                    "freeFlowSpeed": free_flow_speed,
                    "currentTravelTime": flow_data.get("currentTravelTime", 0),
                    "freeFlowTravelTime": flow_data.get("freeFlowTravelTime", 0),
                    "congestion_factor": congestion_factor,
                    "confidence": flow_data.get("confidence", 1.0),
                }
                _flow_cache[cache_key] = (now, result)
                return result
            else:
                print(
                    f"Warning: 'flowSegmentData' not found in TomTom response for {lat},{lon}"
                )
                return None

    except Exception as e:
        print(f"Error fetching TomTom data for {lat},{lon}: {e}")
        return None


# Dictionary mapping major cities to representative coordinates for simulation
CITY_COORDINATES = {
    "Delhi": (28.42494, 77.10221),  # e.g., near commercial hubs
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
}


def get_incidents(
    api_key: str, lat: float, lon: float, radius_deg: float = 0.05
) -> List[Dict]:
    """
    Fetches real-time traffic incidents from the TomTom API in a bounding box around the center point.

    Args:
        api_key: TomTom API key
        lat: Latitude of the center point
        lon: Longitude of the center point
        radius_deg: The radius in degrees to form the bounding box (approx 5.5km per 0.05 deg)

    Returns:
        A list of incident dictionaries.
    """
    # bbox format: minLon,minLat,maxLon,maxLat
    min_lon = lon - radius_deg
    min_lat = lat - radius_deg
    max_lon = lon + radius_deg
    max_lat = lat + radius_deg

    url = (
        f"https://api.tomtom.com/maps/orbis/traffic/incidentDetails?"
        f"apiVersion=1&key={api_key}&bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&fields={{incidents{{type,geometry{{type,coordinates}},properties{{iconCategory}}}}}}"
        f"&language=en-GB&t=1111&timeValidityFilter=present"
    )

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    # Check cache first
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.time()
    if cache_key in _incidents_cache:
        cached_time, cached_data = _incidents_cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    incidents_list = []

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

            if "incidents" in data:
                incidents_list = data["incidents"]

            _incidents_cache[cache_key] = (now, incidents_list)
            return incidents_list

    except Exception as e:
        print(f"Error fetching TomTom incidents for {lat},{lon}: {e}")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test TomTom API Endpoints")
    parser.add_argument("--flow", action="store_true", help="Test Real-Time Flow API")
    parser.add_argument("--incidents", action="store_true", help="Test Incidents API")
    parser.add_argument(
        "--city",
        type=str,
        default="Delhi",
        choices=list(CITY_COORDINATES.keys()),
        help="City to test",
    )
    parser.add_argument("--lat", type=float, help="Latitude override")
    parser.add_argument("--lon", type=float, help="Longitude override")

    args = parser.parse_args()

    api_key = get_api_key()

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        print(f"Testing for Custom Coordinates ({lat}, {lon})")
    else:
        lat, lon = CITY_COORDINATES[args.city]
        print(f"Testing for {args.city} ({lat}, {lon})")

    if args.flow or not (args.flow or args.incidents):
        print("\n--- Testing flowSegmentData ---")
        flow = get_real_time_flow(api_key, lat, lon)
        if flow:
            print(json.dumps(flow, indent=2))
        else:
            print("Failed to fetch flow.")

    if args.incidents or not (args.flow or args.incidents):
        print("\n--- Testing incidentDetails ---")
        incidents = get_incidents(api_key, lat, lon)
        print(f"Found {len(incidents)} active incidents.")
        if incidents:
            print("Sample incident:", json.dumps(incidents[0], indent=2))
