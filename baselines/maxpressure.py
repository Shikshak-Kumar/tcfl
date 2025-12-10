"""
MaxPressure Traffic Light Controller

A classical pressure-based algorithm that selects phases to maximize
the pressure difference (vehicles upstream minus downstream).

Reference: 
- Varaiya, P. (2013). "Max pressure control of a network of signalized intersections"
- Transportation Research Part C: Emerging Technologies, 36, 177-195.
"""

import traci
import numpy as np
from typing import List, Dict, Optional, Tuple


class MaxPressureController:
    """
    MaxPressure traffic light controller.
    
    Selects phase that maximizes pressure (queue difference).
    """
    
    def __init__(
        self,
        tl_id: str,
        min_phase_duration: float = 5.0,
        yellow_duration: float = 3.0,
        incoming_edges: Optional[List[str]] = None
    ):
        """
        Initialize MaxPressure controller.
        
        Args:
            tl_id: Traffic light ID
            min_phase_duration: Minimum phase duration (seconds)
            yellow_duration: Yellow phase duration (seconds)
            incoming_edges: List of incoming edge IDs
        """
        self.tl_id = tl_id
        self.min_phase_duration = min_phase_duration
        self.yellow_duration = yellow_duration
        self.incoming_edges = incoming_edges or []
        
        self.current_phase = 0
        self.phase_start_time = 0.0
        self.last_phase_switch_time = 0.0
        self._pending_target_phase: Optional[int] = None
        
        # Map phases to controlled edges
        self.phase_to_edges: Dict[int, List[str]] = {}
        self._initialize_phase_mapping()
    
    def _initialize_phase_mapping(self):
        """Initialize mapping from phases to controlled edges."""
        try:
            links = traci.trafficlight.getControlledLinks(self.tl_id)
            edge_sets = []
            
            # Group edges by phase
            for link_group in links:
                edges = set()
                for link in link_group:
                    if link and len(link) >= 1:
                        lane_id = link[0]
                        try:
                            edge_id = traci.lane.getEdgeID(lane_id)
                            if edge_id in self.incoming_edges:
                                edges.add(edge_id)
                        except Exception:
                            pass
                edge_sets.append(list(edges))
            
            # Map phases 0 and 2 (green phases) to edge sets
            if len(edge_sets) >= 2:
                self.phase_to_edges[0] = edge_sets[0] if edge_sets[0] else self.incoming_edges[:len(self.incoming_edges)//2]
                self.phase_to_edges[2] = edge_sets[1] if len(edge_sets) > 1 and edge_sets[1] else self.incoming_edges[len(self.incoming_edges)//2:]
            else:
                # Fallback: split edges evenly
                mid = len(self.incoming_edges) // 2
                self.phase_to_edges[0] = self.incoming_edges[:mid]
                self.phase_to_edges[2] = self.incoming_edges[mid:]
        except Exception:
            # Fallback: simple split
            mid = len(self.incoming_edges) // 2
            self.phase_to_edges[0] = self.incoming_edges[:mid] if self.incoming_edges else []
            self.phase_to_edges[2] = self.incoming_edges[mid:] if self.incoming_edges else []
    
    def _get_queue_length(self, edge_id: str) -> int:
        """Get queue length on an edge."""
        try:
            vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
            queue = 0
            for veh_id in vehicles:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    if speed < 0.1:  # Consider stopped vehicles as queued
                        queue += 1
                except Exception:
                    queue += 1
            return queue
        except Exception:
            return 0
    
    def _get_downstream_queue(self, edge_id: str) -> int:
        """
        Get queue length downstream of an edge.
        Simplified: uses outgoing edges or assumes zero.
        """
        try:
            # Get outgoing edges
            outgoing = traci.edge.getSuccessors(edge_id)
            total_queue = 0
            for out_edge in outgoing:
                total_queue += self._get_queue_length(out_edge)
            return total_queue
        except Exception:
            return 0
    
    def _calculate_pressure(self, phase: int) -> float:
        """
        Calculate pressure for a given phase.
        
        Pressure = sum of (upstream_queue - downstream_queue) for controlled edges
        """
        if phase not in self.phase_to_edges:
            return 0.0
        
        controlled_edges = self.phase_to_edges[phase]
        total_pressure = 0.0
        
        for edge_id in controlled_edges:
            upstream_queue = self._get_queue_length(edge_id)
            downstream_queue = self._get_downstream_queue(edge_id)
            pressure = upstream_queue - downstream_queue
            total_pressure += max(0, pressure)  # Only positive pressure
        
        return total_pressure
    
    def _select_best_phase(self) -> int:
        """Select phase with maximum pressure."""
        pressures = {}
        
        # Calculate pressure for green phases
        for phase in [0, 2]:
            pressures[phase] = self._calculate_pressure(phase)
        
        # Select phase with maximum pressure
        best_phase = max(pressures.keys(), key=lambda p: pressures[p])
        return best_phase
    
    def step(self, current_time: float) -> Optional[int]:
        """
        Execute one step of MaxPressure control.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Current phase index if changed, None otherwise
        """
        elapsed = current_time - self.last_phase_switch_time
        
        # Handle pending phase switch (yellow -> green)
        if self._pending_target_phase is not None:
            if elapsed >= self.yellow_duration:
                self.current_phase = self._pending_target_phase
                self._pending_target_phase = None
                self.last_phase_switch_time = current_time
                try:
                    traci.trafficlight.setPhase(self.tl_id, self.current_phase)
                    traci.trafficlight.setPhaseDuration(self.tl_id, self.min_phase_duration)
                except Exception:
                    pass
                return self.current_phase
            return None
        
        # Check if minimum phase duration has passed
        if elapsed >= self.min_phase_duration:
            # Select best phase based on pressure
            best_phase = self._select_best_phase()
            
            # If different from current, switch
            if best_phase != self.current_phase and self.current_phase in [0, 2]:
                # Go to yellow first
                yellow_phase = 1 if self.current_phase == 0 else 3
                self._pending_target_phase = best_phase
                self.current_phase = yellow_phase
                self.last_phase_switch_time = current_time
                
                try:
                    traci.trafficlight.setPhase(self.tl_id, yellow_phase)
                    traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_duration)
                except Exception as e:
                    print(f"Error setting yellow phase: {e}")
                
                return yellow_phase
        
        return None
    
    def reset(self, initial_time: float = 0.0):
        """Reset controller to initial state."""
        self.current_phase = 0
        self.phase_start_time = initial_time
        self.last_phase_switch_time = initial_time
        self._pending_target_phase = None
        self._initialize_phase_mapping()
        try:
            traci.trafficlight.setPhase(self.tl_id, 0)
            traci.trafficlight.setPhaseDuration(self.tl_id, self.min_phase_duration)
        except Exception:
            pass
    
    def get_current_phase(self) -> int:
        """Get current phase index."""
        return self.current_phase

