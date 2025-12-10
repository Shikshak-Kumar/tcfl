"""
Fixed-Time Traffic Light Controller

This is the traditional timer-based approach used in many cities.
Phases cycle with predetermined durations regardless of traffic conditions.

Reference: Traditional traffic engineering practice (pre-1980s)
"""

import traci
from typing import Dict, Optional


class FixedTimeController:
    """
    Fixed-time traffic light controller.
    
    Cycles through phases with fixed durations, independent of traffic.
    """
    
    def __init__(self, tl_id: str, phase_durations: Dict[int, float] = None):
        """
        Initialize fixed-time controller.
        
        Args:
            tl_id: Traffic light ID
            phase_durations: Dict mapping phase index to duration (seconds).
                           Default: 30s green, 3s yellow
        """
        self.tl_id = tl_id
        self.current_phase = 0
        self.phase_start_time = 0.0
        
        # Default: 30s green phases, 3s yellow phases
        if phase_durations is None:
            self.phase_durations = {
                0: 30.0,  # Green phase 1 (e.g., North-South)
                1: 3.0,   # Yellow phase 1
                2: 30.0,  # Green phase 2 (e.g., East-West)
                3: 3.0,   # Yellow phase 2
            }
        else:
            self.phase_durations = phase_durations
    
    def step(self, current_time: float) -> Optional[int]:
        """
        Execute one step of fixed-time control.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Current phase index if changed, None otherwise
        """
        elapsed = current_time - self.phase_start_time
        current_duration = self.phase_durations.get(self.current_phase, 30.0)
        
        if elapsed >= current_duration:
            # Move to next phase
            num_phases = len(self.phase_durations)
            self.current_phase = (self.current_phase + 1) % num_phases
            self.phase_start_time = current_time
            
            try:
                traci.trafficlight.setPhase(self.tl_id, self.current_phase)
                traci.trafficlight.setPhaseDuration(
                    self.tl_id, 
                    self.phase_durations[self.current_phase]
                )
            except Exception as e:
                print(f"Error setting phase: {e}")
            
            return self.current_phase
        
        return None
    
    def reset(self, initial_time: float = 0.0):
        """Reset controller to initial state."""
        self.current_phase = 0
        self.phase_start_time = initial_time
        try:
            traci.trafficlight.setPhase(self.tl_id, 0)
            traci.trafficlight.setPhaseDuration(self.tl_id, self.phase_durations[0])
        except Exception:
            pass
    
    def get_current_phase(self) -> int:
        """Get current phase index."""
        return self.current_phase

