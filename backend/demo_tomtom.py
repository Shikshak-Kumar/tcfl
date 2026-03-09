import numpy as np
import time
import argparse
import os
from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
from utils.tomtom_api import CITY_COORDINATES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live TomTom Traffic Simulation Demo")
    parser.add_argument("--city", type=str, default="Delhi", choices=list(CITY_COORDINATES.keys()), help="City to simulate")
    parser.add_argument("--target-pois", type=str, default=None, help="Comma-separated POIs (e.g. healthcare,education)")
    
    args = parser.parse_args()
    
    # Parse target_pois if provided
    target_pois_list = None
    if args.target_pois:
        target_pois_list = [p.strip() for p in args.target_pois.split(",")]
        
    api_key = os.environ.get("TOMTOM_API_KEY")
    if not api_key:
        raise ValueError("TOMTOM_API_KEY environment variable is missing. Please set it in your .env file or deployment settings.")
    lat, lon = CITY_COORDINATES[args.city]
    
    print("============================================================")
    print("DEMO: REAL-TIME TRAFFIC SIMULATION (TOMTOM API)")
    print("============================================================")
    print(f"Simulating traffic for {args.city} ({lat}, {lon})")
    if target_pois_list:
        print(f"Targeting specific POIs: {target_pois_list}")
    
    # Initialize with Real Time pattern
    env = TomTomTrafficEnvironment(
        sumo_config_path="demo_config", 
        tomtom_api_key=api_key,
        lat=lat,
        lon=lon,
        max_vehicles=1000, 
        traffic_pattern="real_time",
        target_pois=target_pois_list
    )
    
    state = env.reset()
    done = False
    step = 0
    
    print(f"{'Step':<10} | {'Base Rate':<15} | {'Multiplier':<15} | {'Actual Rate':<15} | {'Queue Length':<15}")
    print("-" * 78)
    
    try:
        while not done and step < 50:
            if step % 30 < 15:
                action = 0 # NS Green
            else:
                action = 2 # EW Green
                
            next_state, reward, done, info = env.step(action)
            step += 1
            
            if step % 10 == 0:
                rate = env.arrival_rate
                base = env.base_arrival_rate
                mult = env.congestion_multiplier
                total_q = sum(env.lane_queues.values())
                
                print(f"{step:<10} | {base:<15.2f} | {mult:<15.2f} | {rate:<15.2f} | {total_q:<15}")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
        
    print("-" * 78)
    print("Simulation Complete.")
    print(f"Total Vehicles Processed: {env.total_vehicles}")
