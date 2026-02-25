import numpy as np
import random
import time
from typing import Tuple, List, Dict, Optional
from utils.tomtom_api import get_real_time_flow, get_incidents
from utils.osm_api import get_osm_pois
from agents.mock_traffic_environment import MockTrafficEnvironment

class TomTomTrafficEnvironment(MockTrafficEnvironment):
    """
    A traffic environment that extends the MockTrafficEnvironment to use
    real-time traffic flow data from the TomTom API to dictate arrival rates
    and congestion.
    """

    def __init__(
        self,
        sumo_config_path: str,
        tomtom_api_key: str,
        lat: float,
        lon: float,
        gui: bool = False,
        tl_id: Optional[str] = None,
        show_phase_console: bool = False,
        show_gst_gui: bool = False,
        max_vehicles: int = 400,
        traffic_pattern: str = "real_time",
        target_pois: Optional[List[str]] = None,
    ):
        self.tomtom_api_key = tomtom_api_key
        self.lat = lat
        self.lon = lon
        self.last_fetch_time = 0.0
        self.fetch_interval = 300  # Fetch new data every 5 minutes
        self.congestion_multiplier = 1.0
        
        # Incident impacts
        self.incident_accident_bonus = 0
        self.incident_weather_penalty = False
        self.incident_jam_multiplier = 1.0
        # Array of user-selected target POI categories (e.g., ['healthcare', 'education'])
        self.target_pois = target_pois or []

        print(f"[{tl_id or 'tomtom_env'}] Initializing TomTomTrafficEnvironment for ({self.lat}, {self.lon})")
        
        # Fetch static OSM POIs once
        print(f"[{tl_id or 'tomtom_env'}] Fetching local OpenStreetMap POIs...")
        self.poi_counts = get_osm_pois(lat, lon, radius_deg=0.05)
        
        # Calculate static POI traffic modifier (attractor/generator logic)
        # Default behavior if no targets specified: standard density scaling
        if not self.target_pois:
            commercial = len(self.poi_counts.get("commercial", []))
            office = len(self.poi_counts.get("office", []))
            leisure = len(self.poi_counts.get("leisure", []))
            
            # Base modifier is 1.0. High density commercial/office areas generate more traffic baseline
            self.poi_arrival_modifier = 1.0 + (commercial + office + leisure) / 1000.0
            self.poi_arrival_modifier = min(1.5, max(1.0, self.poi_arrival_modifier))
            print(f"[{tl_id or 'tomtom_env'}] POI Generation Modifier: {self.poi_arrival_modifier:.2f} based on regional POI count.")
        else:
            # Targeted POI behavior
            print(f"[{tl_id or 'tomtom_env'}] Target POIs applied: {self.target_pois}")
            
            # Count how many of the requested target POIs exist in this specific environment block
            target_count = 0
            for t_poi in self.target_pois:
                target_count += len(self.poi_counts.get(t_poi, []))
                
            if target_count > 0:
                print(f"[{tl_id or 'tomtom_env'}] Discovered {target_count} target POIs! Applying prioritized traffic weight.")
                # If they targeted commercial/leisure, generate an aggressive multiplier
                if any(t in ['commercial', 'leisure', 'office', 'food_dining'] for t in self.target_pois):
                    self.poi_arrival_modifier = min(2.0, 1.2 + (target_count / 100.0))
                else:
                    # Healthcare/Education -> Smooth, less bursty multiplier, but consistently high base
                    self.poi_arrival_modifier = 1.1 + (target_count / 200.0)
            else:
                 # If this intersection DOES NOT have the target POIs, dynamically reduce 
                 # the traffic baseline to simulate traffic organically routing AWAY from this area
                 print(f"[{tl_id or 'tomtom_env'}] No target POIs found in this block. Reducing ambient traffic.")
                 self.poi_arrival_modifier = 0.8
            
            print(f"[{tl_id or 'tomtom_env'}] Targeted POI Modifier established at: {self.poi_arrival_modifier:.2f}")

        # Initialize the base mock environment
        super().__init__(
            sumo_config_path=sumo_config_path,
            gui=gui,
            tl_id=tl_id,
            show_phase_console=show_phase_console,
            show_gst_gui=show_gst_gui,
            max_vehicles=max_vehicles,
            arrival_rate=0.3, # Will be overridden
            traffic_pattern=traffic_pattern,
        )

        # Initial fetch immediately overrides the base_arrival_rate
        self.fetch_tomtom_data()

    def fetch_tomtom_data(self):
        """Fetches traffic data from TomTom and updates congestion parameters."""
        current_time = time.time()
        if current_time - self.last_fetch_time < self.fetch_interval and self.last_fetch_time > 0:
            return  # Throttle API calls

        print(f"[{self.tl_id}] Fetching real-time TomTom data for ({self.lat}, {self.lon})...")
        flow_data = get_real_time_flow(self.tomtom_api_key, self.lat, self.lon)
        self.last_fetch_time = current_time

        if flow_data:
            # congestion_factor = freeFlowSpeed / currentSpeed
            raw_congestion = flow_data['congestion_factor'] 
            self.congestion_multiplier = 0.8 + ((raw_congestion - 1.0) * 1.5)
            self.congestion_multiplier = max(0.5, min(3.0, self.congestion_multiplier))

            print(f"[{self.tl_id}] TomTom Data Received! Speed: {flow_data['currentSpeed']}/{flow_data['freeFlowSpeed']} km/h. "
                  f"Congestion Factor: {raw_congestion:.2f} -> Multiplier: {self.congestion_multiplier:.2f}")
        else:
            print(f"[{self.tl_id}] Failed to fetch TomTom flow data, falling back to previous multiplier: {self.congestion_multiplier:.2f}")
            
        print(f"[{self.tl_id}] Fetching real-time TomTom incidents...")
        incidents = get_incidents(self.tomtom_api_key, self.lat, self.lon)
        
        self.incident_accident_bonus = 0
        self.incident_weather_penalty = False
        self.incident_jam_multiplier = 1.0
        self.incident_lane_closure = False
        
        if incidents:
            print(f"[{self.tl_id}] Found {len(incidents)} incidents near the intersection!")
            for inc in incidents:
                cat = inc.get('properties', {}).get('iconCategory', 0)
                if cat in [1, 13, 14]: # Accident, Cluster, Broken Down
                    self.incident_accident_bonus += 1
                elif cat in [2, 4, 5, 10, 11]: # Fog, Rain, Ice, Wind, Flooding
                    self.incident_weather_penalty = True
                elif cat == 6: # Jam
                    self.incident_jam_multiplier = 1.5
                elif cat in [3, 7, 8, 9, 12]: # Dangerous, Lane Closed, Road Closed, Road Works, Detour
                    self.incident_lane_closure = True
                    
            print(f"[{self.tl_id}] Incident impacts -> Accidents: +{self.incident_accident_bonus}, Weather: {self.incident_weather_penalty}, "
                  f"Jam Mult: {self.incident_jam_multiplier}, Lane Closure: {self.incident_lane_closure}")

    def _update_arrival_rate(self):
        """Overrides the arrival rate logic to incorporate TomTom data and POI modifiers."""
        
        # Check if we need to refresh the TomTom data
        self.fetch_tomtom_data()

        if self.traffic_pattern == "real_time":
            # Just use base_arrival_rate scaled by the live congestion multiplier and static POIs
            self.arrival_rate = self.base_arrival_rate * self.congestion_multiplier * self.incident_jam_multiplier * self.poi_arrival_modifier
            self.arrival_rate = max(0.1, min(0.95, self.arrival_rate))
        else:
            # Fall back to inherited pattern but still scale it by ambient real-world congestion
            super()._update_arrival_rate()
            # Apply the ambient congestion on top of the sine wave / bursty logic
            self.arrival_rate *= (self.congestion_multiplier * self.incident_jam_multiplier * self.poi_arrival_modifier * 0.8) # dampen it a bit to avoid saturation
            self.arrival_rate = max(0.1, min(0.95, self.arrival_rate))
            
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Temporarily back up accident/near_miss logic to override them post-step
        old_accidents = self.total_accidents
        old_near_misses = self.near_misses
        
        # Override random variables right before standard step
        weather_prob_mod = 0.05 if self.incident_weather_penalty else 0.0
        
        state, reward, done, info = super().step(action)
        
        # 1. Apply Accident Bonuses directly
        if self.incident_accident_bonus > 0:
            if random.random() < 0.1: # 10% chance per step per real-world accident to cause a synthetic accident
                self.total_accidents += self.incident_accident_bonus
                self.near_misses += self.incident_accident_bonus * 2
                reward -= 50.0 * self.incident_accident_bonus # Heavy penalty
                
        # 2. Apply weather penalties (sliding/poor visibility)
        if self.incident_weather_penalty:
            if random.random() < weather_prob_mod:
                self.near_misses += 1
                reward -= 5.0
                
        # 3. Apply Lane Closures (restrict flux retrospectively if possible, but easier to just penalize the reward and manually reduce queue drain)
        if self.incident_lane_closure:
            # Revert some of the queue reduction that `super().step()` might have done
            # by randomly adding vehicles back to simulate restricted outflow
            green_edges = []
            if self.current_phase == 0:
                green_edges = ["edge_n", "edge_s"]
            elif self.current_phase == 2:
                green_edges = ["edge_e", "edge_w"]
                
            for edge in green_edges:
                if random.random() < 0.5: # 50% chance a vehicle couldn't actually leave due to lane closed
                    self.lane_queues[edge] += 1
                    
        return state, reward, done, info
