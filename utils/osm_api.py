import requests
import json

def get_osm_pois(lat: float, lon: float, radius_deg: float = 0.05) -> dict:
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
    # using nodes and ways for the requested keys
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["leisure"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["shop"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["office"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center tags;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Categorize the results and store coordinates
        poi_summary = {
            "healthcare": [],    # Hospitals, clinics, pharmacies
            "education": [],     # Schools, universities
            "commercial": [],    # Shops, malls, markets
            "office": [],        # Business districts, offices
            "leisure": [],       # Parks, stadiums, entertainment
            "food_dining": [],   # Restaurants, cafes, fast_food
            "public_service": [] # Police, fire, civic
        }
        
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            
            # Extract coordinates (ways have centers if we used center modifier, but nodes have lat/lon directly)
            # Since we just used basic out, ways won't have lat/lon directly, but nodes will.
            # To be safe, try to get them, or default to 0.0
            poi_lat = element.get('lat', 0.0)
            poi_lon = element.get('lon', 0.0)
            
            # If it's a way and doesn't have lat/lon, we can try to average its nodes but that requires another query.
            # For this simulation's purposes, we'll store the node's lat/lon or skip if missing.
            if poi_lat == 0.0 and poi_lon == 0.0:
                 continue
                 
            poi_data = {"lat": poi_lat, "lon": poi_lon, "name": tags.get('name', 'Unknown')}
            
            amenity = tags.get('amenity', '')
            shop = tags.get('shop', '')
            leisure = tags.get('leisure', '')
            office = tags.get('office', '')
            
            if amenity in ['hospital', 'clinic', 'doctors', 'pharmacy', 'dentist']:
                poi_summary["healthcare"].append(poi_data)
            elif amenity in ['school', 'university', 'college', 'kindergarten']:
                poi_summary["education"].append(poi_data)
            elif amenity in ['restaurant', 'cafe', 'fast_food', 'bar', 'pub', 'food_court']:
                poi_summary["food_dining"].append(poi_data)
            elif amenity in ['police', 'fire_station', 'courthouse', 'townhall', 'library']:
                poi_summary["public_service"].append(poi_data)
            elif shop:
                poi_summary["commercial"].append(poi_data)
            elif office:
                poi_summary["office"].append(poi_data)
            elif leisure or amenity in ['cinema', 'theatre', 'stadium', 'sports_centre', 'park']:
                poi_summary["leisure"].append(poi_data)
                
        return poi_summary

    except Exception as e:
        print(f"[OSM API] Failed to fetch data: {e}")
        return {}

if __name__ == "__main__":
    import argparse
    from utils.tomtom_api import CITY_COORDINATES
    
    parser = argparse.ArgumentParser(description="Test OSM Overpass API Endpoint")
    parser.add_argument("--city", type=str, default="Delhi", choices=list(CITY_COORDINATES.keys()), help="City to test")
    parser.add_argument("--lat", type=float, help="Latitude override")
    parser.add_argument("--lon", type=float, help="Longitude override")
    parser.add_argument("--radius", type=float, default=0.05, help="Radius in degrees")
    
    args = parser.parse_args()
    
    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        print(f"Testing OSM POI fetch for Custom Coordinates ({lat}, {lon}) with radius {args.radius}...")
    else:
        lat, lon = CITY_COORDINATES[args.city]
        print(f"Testing OSM POI fetch for {args.city} ({lat}, {lon}) with radius {args.radius}...")
    
    pois = get_osm_pois(lat, lon, radius_deg=args.radius)
    print("\nPOI Summary:")
    print(json.dumps(pois, indent=2))
