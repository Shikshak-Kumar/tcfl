import numpy as np
import random
import time
from typing import Tuple, List, Dict, Optional

class MockTrafficEnvironment:
    """
    A mock environment that mimics SUMOTrafficEnvironment but generates 
    synthetic data with basic queue dynamics for realism.
    """
    
    def __init__(self, sumo_config_path: str, gui: bool = False, tl_id: Optional[str] = None, show_phase_console: bool = False, show_gst_gui: bool = False, max_vehicles: int = 400, arrival_rate: float = 0.3, traffic_pattern: str = "uniform"):
        self.sumo_config_path = sumo_config_path
        self.gui = gui
        self.show_phase_console = show_phase_console
        self.show_gst_gui = show_gst_gui
        self.state_size = 12
        self.action_size = 4
        
        self.step_count = 0
        self.max_steps = 2000 # Increased to allow max_vehicles to be reached
        self._seed_from_config(sumo_config_path)
        self.traffic_pattern = traffic_pattern
        self.arrival_rate = self.base_arrival_rate
        self.episode_count = 0
        self.neighbors = [] # Coupled environments
        
        # Mock specific state
        self.current_phase = 0
        self.num_phases = 4
        self.tl_id = "mock_tl_0"
        self.incoming_edges = ["edge_n", "edge_e", "edge_s", "edge_w"]
        
        # Internal simulation state
        self.lane_queues = {edge: random.randint(0, 5) for edge in self.incoming_edges}
        self.lane_waiting_times = {edge: 0.0 for edge in self.incoming_edges}
        
        # Synthetic metrics
        self.total_waiting_time = 0.0
        self.total_vehicles = 0
        self.queue_lengths = []
        self.total_accidents = 0
        self.near_misses = 0
        
        print(f"initialized MockTrafficEnvironment (config={sumo_config_path})")

    def _seed_from_config(self, config_path: str):
        """Mock 'data-driven' seeding from SUMO configs."""
        if "netccfg" in config_path: # Central / Dense
            self.base_arrival_rate = 0.4
            self.target_vehicles = 600
        elif "polycfg" in config_path: # Suburban / Sparse
            self.base_arrival_rate = 0.2
            self.target_vehicles = 300
        else:
            self.base_arrival_rate = 0.3
            self.target_vehicles = 400

    def add_neighbor(self, neighbor_env):
        self.neighbors.append(neighbor_env)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.total_waiting_time = 0.0
        self.total_vehicles = 0
        self.queue_lengths = []
        self.total_accidents = 0
        self.near_misses = 0
        self.episode_count += 1
        
        # Reset internal state
        self.lane_queues = {edge: random.randint(0, 5) for edge in self.incoming_edges}
        self.lane_waiting_times = {edge: 0.0 for edge in self.incoming_edges}
        
        return self.get_state()

    def get_state(self) -> np.ndarray:
        # Generate state vector based on internal queues
        # Normalize queue lengths (assuming max queue ~20)
        state = []
        for edge in self.incoming_edges:
            q = self.lane_queues[edge]
            w = self.lane_waiting_times[edge]
            # [Queue density, Waiting density, Vehicle density]
            state.extend([
                min(q / 20.0, 1.0),
                min(w / 300.0, 1.0),
                min((q + random.randint(0, 2)) / 20.0, 1.0)
            ])
        return np.array(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        self.current_phase = action % self.num_phases
        
        # 0. Update Traffic Pattern
        self._update_arrival_rate()

        # 1. Inflow: Add random vehicles to queues
        if self.total_vehicles < self.target_vehicles:
            for edge in self.incoming_edges:
                if random.random() < self.arrival_rate:  # Use dynamic rate
                    self.lane_queues[edge] += 1
                    self.total_vehicles += 1
        
        # 2. Outflow: Reduce queues for edges with GREEN light
        # Simplify phase logic: 
        # Phase 0: NS Green (edge_n, edge_s)
        # Phase 1: NS Yellow
        # Phase 2: EW Green (edge_e, edge_w)
        # Phase 3: EW Yellow
        
        green_edges = []
        if self.current_phase == 0:
            green_edges = ["edge_n", "edge_s"]
        elif self.current_phase == 2:
            green_edges = ["edge_e", "edge_w"]
            
        for edge in green_edges:
            # Vehicles leave faster than they arrive
            flux = random.randint(1, 3)
            actual_flux = min(flux, self.lane_queues[edge])
            self.lane_queues[edge] -= actual_flux
            self.lane_waiting_times[edge] = max(0, self.lane_waiting_times[edge] - 5.0 * actual_flux)
            
            # 2.1 Transition to neighbors (Graph Coupling)
            # This makes the graph attention meaningful
            if actual_flux > 0 and self.neighbors:
                for _ in range(actual_flux):
                    if random.random() < 0.7: # 70% chance to enter a neighbor intersection
                        target_nb = random.choice(self.neighbors)
                        target_edge = random.choice(target_nb.incoming_edges)
                        target_nb.lane_queues[target_edge] += 1
                        # This flow increases 'Pressure' on the neighbor


        # 3. Update Waiting Times & Safety
        step_total_wait = 0.0
        current_step_accidents = 0
        current_step_near_misses = 0
        
        for edge in self.incoming_edges:
            # Add wait time for queued vehicles
            added_wait = self.lane_queues[edge] * 1.0  # 1 sec per vehicle
            self.lane_waiting_times[edge] += added_wait
            step_total_wait += added_wait # Only add the wait INcurred in this step
            
            # Simulate Safety/Accident Risk
            # Higher queue = higher risk of rear-end collision
            queue_len = self.lane_queues[edge]
            if queue_len > 10:
                # Risk increases with queue length
                risk_prob = min(0.05, (queue_len - 10) * 0.002)
                if random.random() < risk_prob:
                    current_step_accidents += 1
                    self.total_accidents += 1
                elif random.random() < (risk_prob * 3):
                    current_step_near_misses += 1
                    self.near_misses += 1
            
        self.total_waiting_time += step_total_wait
        self.queue_lengths.append(sum(self.lane_queues.values()))
        
        # 4. Calculate Reward (Normalized for RL stability)
        total_queue = sum(self.lane_queues.values())
        avg_wait = np.mean(list(self.lane_waiting_times.values()))
        
        # Normalize reward components: 
        # total_queue / 20.0 -> scaled down
        # step_total_wait / 20.0 -> scaled down (instantaneous wait)
        reward = - (total_queue / 20.0 + step_total_wait / 20.0)
        
        # Switching penalty (Practical stability)
        if self.current_phase != action:
            reward -= 0.5
            
        # Safety penalty 
        if current_step_accidents > 0:
            reward -= 2.0
            
        reward = max(-10.0, min(1.0, reward)) # Tight clip for stability
        
        done = (self.step_count >= self.max_steps) or (self.total_vehicles >= self.target_vehicles and sum(self.lane_queues.values()) == 0)
        next_state = self.get_state()
        
        info = {
            'step': self.step_count,
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'queue_lengths': self.queue_lengths,
            'safety': {
                'accidents': self.total_accidents,
                'near_misses': self.near_misses,
                'safety_score': max(0, 100 - (self.total_accidents * 5 + self.near_misses))
            },
            'phase': {
                'phase': self.current_phase,
                'remaining_s': random.uniform(0, 30),
                'duration_s': 30.0,
                'num_phases': self.num_phases
            },
            'gst': self._compute_green_signal_times()
        }
        
        if self.show_phase_console and self.step_count % 10 == 0:
            print(f"[Mock] Step {self.step_count}: Queues {self.lane_queues}, Accidents {self.total_accidents}, Reward {reward:.2f}")

        return next_state, reward, done, info

    def _update_arrival_rate(self):
        """Updates arrival rate based on traffic pattern."""
        if self.traffic_pattern == "uniform":
            self.arrival_rate = self.base_arrival_rate
        elif self.traffic_pattern == "rush_hour":
            # Simulate a sine wave of traffic: Peaking at step 100 and 300
            # Base rate + Amplitude * sin(step_factor)
            cycle_position = (self.step_count % 400) / 400.0 * 2 * np.pi
            traffic_flow = np.sin(cycle_position)
            # Map [-1, 1] to [0.5, 2.5] intensity multiplier for denser traffic
            intensity = 1.5 + (1.0 * traffic_flow) 
            self.arrival_rate = self.base_arrival_rate * intensity
            
            # Clamp to reasonable bounds
            self.arrival_rate = max(0.1, min(0.95, self.arrival_rate))
        elif self.traffic_pattern == "bursty":
             # Randomly switch between Calm (low) and Burst (high)
            if self.step_count % 50 == 0:
                 if random.random() < 0.3:
                     self.arrival_rate = min(0.9, self.base_arrival_rate * 3.0) # Burst
                 else:
                     self.arrival_rate = self.base_arrival_rate * 0.5 # Calm

    def _compute_green_signal_times(self) -> Dict:
        # Generate synthetic GST data based on actual queues
        per_edge = {}
        for edge in self.incoming_edges:
            # More queue = needs more green time
            q = self.lane_queues[edge]
            per_edge[edge] = min(30.0, 5.0 + q * 1.5)
            
        return {
            'per_edge': per_edge,
            'avg_gst': np.mean(list(per_edge.values())),
            'num_lanes': len(self.incoming_edges),
            'min_green_time': 5.0,
            'max_green_time': 30.0
        }

    def get_performance_metrics(self) -> Dict:
        return {
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'average_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': max(self.queue_lengths) if self.queue_lengths else 0,
            'steps': self.step_count,
            'avg_waiting_time_per_vehicle': (self.total_waiting_time / max(1, self.total_vehicles)),
            'green_signal_time': self._compute_green_signal_times(),
            'safety': {
                'total_accidents': self.total_accidents,
                'near_misses': self.near_misses,
                'safety_score': max(0, 100 - (self.total_accidents * 5 + self.near_misses))
            },
            'per_lane_metrics': self._generate_detailed_metrics(), 
            'lane_summary': {}
        }
        
    def _generate_detailed_metrics(self) -> Dict:
        # Generate detailed metrics matching SUMO structure
        detailed = {}
        for edge in self.incoming_edges:
            lane_id = f"{edge}_0"
            detailed[lane_id] = {
                "edge_id": edge,
                "lane_id": lane_id,
                "vehicle_count": self.lane_queues[edge],
                "queue_length": self.lane_queues[edge],
                "waiting_time": self.lane_waiting_times[edge],
                "average_speed": random.uniform(0, 15) if self.lane_queues[edge] == 0 else random.uniform(0, 5),
                "occupancy_percent": min(100, self.lane_queues[edge] * 5),
                "green_signal_time": 0,
                "vehicle_types": {"car": self.lane_queues[edge]},
                "lane_length": 100.0
            }
        return detailed

    def get_pressure(self) -> float:
        """
        Pressure = Σ incoming_queue - Σ outgoing_queue
        In this mock environment, outgoing_queue is simulated as 1/3 of incoming.
        """
        incoming = sum(self.lane_queues.values())
        outgoing = incoming * 0.3 # Mock outgoing flow
        return incoming - outgoing

    def close(self):
        pass
