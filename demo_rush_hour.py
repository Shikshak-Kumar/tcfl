import numpy as np
from agents.mock_traffic_environment import MockTrafficEnvironment
import time

def demo_rush_hour():
    print("============================================================")
    print("DEMO: RUSH HOUR TRAFFIC SIMULATION (MOCK MODE)")
    print("============================================================")
    print("Simulating a day of traffic with morning and evening peaks...")
    
    # Initialize with Rush Hour pattern
    env = MockTrafficEnvironment(
        sumo_config_path="demo_config", 
        max_vehicles=1000, 
        arrival_rate=0.3, 
        traffic_pattern="rush_hour"
    )
    
    state = env.reset()
    done = False
    step = 0
    
    print(f"{'Step':<10} | {'Time of Day':<15} | {'Traffic Level':<15} | {'Queue Length':<15}")
    print("-" * 65)
    
    try:
        while not done and step < 400:
            # Dumb Logic (Fixed-Time) to let queues build up for demo
            if step % 30 < 15:
                action = 0 # NS Green
            else:
                action = 2 # EW Green
                
            next_state, reward, done, info = env.step(action)
            step += 1
            
            # Interpret "Time of Day" based on step (0-400 steps ~ 24 hours scaled)
            # Peak 1 at ~100 steps (Morning Rush)
            # Peak 2 at ~300 steps (Evening Rush)
            
            if step % 20 == 0:
                rate = env.arrival_rate
                level = "LOW"
                if rate > 0.4: level = "MEDIUM"
                if rate > 0.6: level = "HEAVY (Rush)"
                
                total_q = sum(env.lane_queues.values())
                time_label = "Morning" if step < 150 else ("Mid-day" if step < 250 else "Evening")
                
                print(f"{step:<10} | {time_label:<15} | {level:<15} | {total_q:<15}")
                
                if level == "HEAVY (Rush)":
                    print(f"   >>> RUSH HOUR ALERT! High inflow detected ({rate:.2f} cars/sec) <<<")

    except KeyboardInterrupt:
        print("\nSimulation stopped.")
        
    print("-" * 65)
    print("Simulation Complete.")
    print(f"Total Vehicles Processed: {env.total_vehicles}")
    print("Notice how queue lengths spiked during 'RUSH HOUR' periods!")

if __name__ == "__main__":
    demo_rush_hour()
