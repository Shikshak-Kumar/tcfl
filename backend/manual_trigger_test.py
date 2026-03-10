import datetime
from dotenv import load_dotenv
load_dotenv()
from utils.tomtom_api import get_api_key, get_real_time_flow, CITY_COORDINATES
from utils.traffic_db import insert_record, get_time_slot_aggregations
import json

def run_manual_collection():
    print("--- Manual Traffic Data Collection Trigger ---")
    try:
        api_key = get_api_key()
        now = datetime.datetime.now()
        print(f"Current Time: {now.strftime('%H:%M:%S')}")
        
        print(f"Fetching traffic data for {len(CITY_COORDINATES)} locations...")
        for city_name, coords in CITY_COORDINATES.items():
            lat, lon = coords
            print(f" - Requesting data for {city_name}...")
            flow_data = get_real_time_flow(api_key, lat, lon)
            
            if flow_data:
                insert_record(
                    location_id=city_name,
                    lat=lat,
                    lon=lon,
                    current_speed=flow_data.get("currentSpeed", 0.0),
                    free_flow_speed=flow_data.get("freeFlowSpeed", 0.0),
                    congestion_ratio=flow_data.get("congestion_factor", 1.0)
                )
                print(f"   Success: {flow_data['currentSpeed']} / {flow_data['freeFlowSpeed']} km/h")
        
        print("\nCollection Complete. Checking database for latest stats...")
        # Since we just inserted data, we can check the stats for today
        stats = get_time_slot_aggregations(target_date=now.date())
        
        # Note: This might still show 0 records for the slot if the current time 
        # is between slots, but the data IS saved in the database.
        
        print("\nRun successful. You can now verify the database records directly.")
    except Exception as e:
        print(f"Error during manual collection: {e}")

if __name__ == "__main__":
    run_manual_collection()
