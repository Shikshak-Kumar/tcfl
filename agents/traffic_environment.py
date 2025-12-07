import traci
import numpy as np
import random
import os
import sys
from typing import Tuple, List, Dict, Optional
import time

class SUMOTrafficEnvironment:
    
    def __init__(self, sumo_config_path: str, gui: bool = False, tl_id: Optional[str] = None, show_phase_console: bool = False, show_gst_gui: bool = False):
        self.sumo_config_path = sumo_config_path
        self.gui = gui
        self.show_phase_console = show_phase_console
        self.show_gst_gui = show_gst_gui
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = 400
        self.min_phase_duration_s: float = 5.0
        self.yellow_duration_s: float = 3.0
        self.last_phase_switch_time: float = 0.0
        self._pending_target_phase: Optional[int] = None
        
        self.phases = {
            "GGrrGGrr",
            "yyrryyrr",
            "rrGGrrGG",
            "rryyrryy"
        }
        
        self.state_size = 12
        self.action_size = 4
        
        self.tl_id: Optional[str] = tl_id
        self.incoming_edges: List[str] = []
        self.num_phases: int = 1
        
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.queue_lengths = []
        self.gst_history: List[Dict] = []
        self._gst_poi_ids: Dict[str, str] = {}
        self._per_vehicle_wait: Dict[str, float] = {}
        self.class_avg_time_s: Dict[str, float] = {
            'car': 2.0,
            'truck': 3.5,
            'bus': 3.5,
            'motorcycle': 1.5,
            'bicycle': 2.0
        }
        self.class_startup_time_s: Dict[str, float] = {
            'car': 0.7,
            'truck': 1.2,
            'bus': 1.2,
            'motorcycle': 0.5,
            'bicycle': 0.6
        }
        
    def start_simulation(self):
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_config_path,
            "--no-step-log", "true",
            "--collision.action", "teleport",
            "--time-to-teleport", "60",
            "--step-length", "1.0"
        ]
        try:
            cfg_dir = os.path.dirname(self.sumo_config_path)
            view_file = os.path.join(cfg_dir, "osm.view.xml")
            if self.gui and os.path.isfile(view_file):
                sumo_cmd += ["--gui-settings-file", view_file]
        except Exception:
            pass
        
        traci.start(sumo_cmd)

        try:
            if self.tl_id is None:
                tls_ids = traci.trafficlight.getIDList()
                if tls_ids:
                    self.tl_id = tls_ids[0]
                    print(f"Using traffic light: {self.tl_id}")
            if self.tl_id is not None:
                links = traci.trafficlight.getControlledLinks(self.tl_id)
                lane_ids = []
                for conn_group in links:
                    for conn in conn_group:
                        if conn and len(conn) >= 1:
                            lane_ids.append(conn[0])
                edge_ids = []
                for lane_id in lane_ids:
                    try:
                        edge_ids.append(traci.lane.getEdgeID(lane_id))
                    except Exception:
                        pass
                seen = set()
                ordered_unique = []
                for e in edge_ids:
                    if e not in seen and e != '':
                        seen.add(e)
                        ordered_unique.append(e)

                self.incoming_edges = ordered_unique[:5]
                print(f"Incoming edges: {self.incoming_edges}")
                self._edge_to_lane: Dict[str, str] = {}
                for conn_group in links:
                    for conn in conn_group:
                        if conn and len(conn) >= 1:
                            lane_id = conn[0]
                            try:
                                edge_id = traci.lane.getEdgeID(lane_id)
                                if edge_id in self.incoming_edges and edge_id not in self._edge_to_lane:
                                    self._edge_to_lane[edge_id] = lane_id
                            except Exception:
                                pass

                try:
                    logics = traci.trafficlight.getAllProgramLogics(self.tl_id)
                    if logics:
                        logic = logics[0]
                        phases = getattr(logic, 'phases', None)
                        if phases is None:
                            phases = getattr(logic, 'getPhases', lambda: [])()
                        self.num_phases = max(1, len(phases))
                    else:
                        self.num_phases = 1
                except Exception:
                    self.num_phases = 1
        except Exception as e:
            print(f"Topology discovery failed: {e}")
        self.episode_count += 1
        self.step_count = 0
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.queue_lengths = []
        self._per_vehicle_wait = {}
        self._setup_gst_gui_pois()
        
    def stop_simulation(self):
        try:
            traci.close()
        except Exception:
            pass
        
    def reset(self) -> np.ndarray:
        if self.episode_count > 0:
            self.stop_simulation()
            
        self.start_simulation()
        return self.get_state()
        
    def get_state(self) -> np.ndarray:
        try:
            metrics: List[float] = []
            edges = self.incoming_edges[:4]
            while len(edges) < 4:
                edges.append("")
            for edge_id in edges:
                veh_count = self._get_vehicle_count(edge_id) if edge_id else 0
                queue_len = self._get_queue_length(edge_id) if edge_id else 0
                wait_time = self._get_waiting_time(edge_id) if edge_id else 0.0
                metrics.extend([
                    min(veh_count / 20.0, 1.0),
                    min(queue_len / 20.0, 1.0),
                    min(wait_time / 200.0, 1.0)
                ])
            return np.array(metrics, dtype=float)
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size)
    
    def _get_vehicle_count(self, edge_id: str) -> int:
        try:
            return int(traci.edge.getLastStepVehicleNumber(edge_id))
        except:
            return 0

    def _get_queue_length(self, edge_id: str) -> int:
        try:
            vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
            queue_length = 0
            for veh_id in vehicles:
                    queue_length += 1
            return queue_length
        except:
            return 0
    
    def _get_waiting_time(self, edge_id: str) -> float:
        try:
            return traci.edge.getWaitingTime(edge_id)
        except:
            return 0.0

    def _map_vehicle_to_class(self, veh_id: str) -> str:
        try:
            vclass = traci.vehicle.getVehicleClass(veh_id)
        except Exception:
            try:
                type_id = traci.vehicle.getTypeID(veh_id)
                vclass = traci.vehicletype.getVehicleClass(type_id)
            except Exception:
                vclass = 'passenger'
            return 'car'
            return 'truck'
            return 'bus'
        if vclass in ('motorcycle',):
            return 'motorcycle'
        if vclass in ('bicycle',):
            return 'bicycle'
        return 'car'

    def _compute_green_signal_times(self) -> Dict:
        try:
            edges = self.incoming_edges[:5]
            num_lanes = max(1, len(edges))
            gst_per_edge: Dict[str, float] = {}
            
            min_green_time = 5.0  # Minimum 5 seconds for safety
            max_green_time = 30.0
            
            for edge_id in edges:
                if not edge_id:
                    continue
                    
                vehicle_ids = []
                try:
                    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                except Exception:
                    vehicle_ids = []
                
                if not vehicle_ids:
                    lane_id = getattr(self, '_edge_to_lane', {}).get(edge_id, None)
                    if lane_id:
                        try:
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                        except Exception:
                            vehicle_ids = []
                
                class_counts: Dict[str, int] = {}
                for vid in vehicle_ids:
                    cls = self._map_vehicle_to_class(vid)
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                total_vehicles = sum(class_counts.values())
                base_time = 0.0
                
                for cls, cnt in class_counts.items():
                    avg_t = self.class_avg_time_s.get(cls, 2.0)
                    start_t = self.class_startup_time_s.get(cls, 0.7)
                    base_time += float(cnt) * (avg_t + start_t)
                
                queue_length = self._get_queue_length(edge_id)
                
                if total_vehicles == 0:
                    gst = 0.0
                else:
                    time_per_vehicle = 2.5
                    
                    vehicle_time = total_vehicles * time_per_vehicle
                    
                    queue_penalty = queue_length * 1.5
                    
                    required_time = vehicle_time + queue_penalty
                    
                    density_factor = min(2.0, max(1.0, total_vehicles / 5.0))
                    
                    gst = required_time * density_factor
                    
                    gst = max(0.0, min(max_green_time, gst))
                
                gst_per_edge[edge_id] = float(gst)

            avg_gst = float(np.mean(list(gst_per_edge.values()))) if gst_per_edge else 0.0
            snapshot = {
                'per_edge': gst_per_edge,
                'avg_gst': avg_gst,
                'num_lanes': num_lanes,
                'min_green_time': min_green_time,
                'max_green_time': max_green_time
            }
            
            try:
                snapshot_with_time = dict(snapshot)
                snapshot_with_time['time'] = float(traci.simulation.getTime())
                self.gst_history.append(snapshot_with_time)
                if len(self.gst_history) > 2000:
                    self.gst_history = self.gst_history[-2000:]
            except Exception:
                pass
            return snapshot
        except Exception as e:
            print(f"Error computing GST: {e}")
            return {'per_edge': {}, 'avg_gst': 5.0, 'num_lanes': 0}

    def _setup_gst_gui_pois(self):
        if not (self.gui and self.show_gst_gui and self.tl_id and self.incoming_edges):
            return
        try:
            for edge_id in self.incoming_edges:
                lane_id = getattr(self, '_edge_to_lane', {}).get(edge_id, None)
                if not lane_id:
                    continue
                try:
                    shape = traci.lane.getShape(lane_id)
                    x, y = shape[0]
                except Exception:
                    pos = traci.junction.getPosition(self.tl_id)
                    x, y = pos[0], pos[1]
                poi_id = f"gst_{edge_id}"
                try:
                    traci.poi.remove(poi_id)
                except Exception:
                    pass
                try:
                    traci.poi.add(poi_id, x, y, (0, 255, 0, 255), name=f"GST {edge_id}: --s")
                except Exception:
                    traci.poi.add(poi_id, x, y)
                self._gst_poi_ids[edge_id] = poi_id
        except Exception:
            pass

    def _update_gst_gui_labels(self, gst_snapshot: Dict):
        if not (self.gui and self.show_gst_gui):
            return
        try:
            per_edge = gst_snapshot.get('per_edge', {})
            for edge_id, val in per_edge.items():
                poi_id = self._gst_poi_ids.get(edge_id)
                if not poi_id:
                    continue
                label = f"GST {edge_id[-6:]}: {float(val):.1f}s"
                try:
                    traci.poi.setParameter(poi_id, "name", label)
                except Exception:
                    v = float(val)
                    r = int(max(0, min(255, 40 + 10 * v)))
                    g = int(max(0, 255 - 5 * v))
                    traci.poi.setColor(poi_id, (r, g, 0, 255))
        except Exception:
            pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        try:
            self._set_traffic_light_phase(action)
            
            traci.simulationStep()
            self.step_count += 1

            try:
                veh_ids = traci.vehicle.getIDList()
                for vid in veh_ids:
                    try:
                        w = float(traci.vehicle.getAccumulatedWaitingTime(vid))
                    except Exception:
                        w = 0.0
                    self._per_vehicle_wait[vid] = w
            except Exception:
                pass
            
            next_state = self.get_state()
            
            reward = self._calculate_reward()
            
            done = self.step_count >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
            
            phase_info = self.get_current_phase_info()
            gst_snapshot = self._compute_green_signal_times()
            if self.show_phase_console:
                try:
                    per_edge = gst_snapshot.get('per_edge', {})
                    if per_edge:
                        print(f"\n[TARGET INTERSECTION - REAL-TIME ROAD METRICS]")
                        print("-" * 50)
                        road_num = 1
                        for edge_id, gst_val in per_edge.items():
                            vehicles = self._get_vehicle_count(edge_id)
                            queue = self._get_queue_length(edge_id)
                            waiting = self._get_waiting_time(edge_id)
                            speed = self._get_average_speed(edge_id)
                            
                            light_status = self._get_traffic_light_status(edge_id)
                            
                            print(f"Road #{road_num} ({edge_id}): "
                                  f"Vehicles={vehicles}, Queue={queue}, "
                                  f"Waiting={waiting:.1f}s, GST={float(gst_val):.1f}s, "
                                  f"Speed={speed:.1f}m/s, Light={light_status}")
                            road_num += 1
                        print("-" * 40)
                except Exception:
                    pass
            self._update_gst_gui_labels(gst_snapshot)
            if self.show_phase_console and phase_info:
                print(f"[TLS {self.tl_id}] phase={phase_info.get('phase')} remaining={phase_info.get('remaining_s'):.1f}s/{phase_info.get('duration_s'):.1f}s")

            info = {
                'step': self.step_count,
                'total_waiting_time': self.total_waiting_time,
                'total_vehicles': self.total_vehicles,
                'queue_lengths': self.queue_lengths.copy()
            }
            if phase_info:
                info['phase'] = phase_info
            info['gst'] = gst_snapshot

            try:
                if phase_info:
                    phase_idx = int(phase_info.get('phase', -1))
                    remaining = float(phase_info.get('remaining_s', 0.0))
                    if phase_idx in (0, 2) and remaining <= 0.5:
                        queue_sum = 0
                        for e in self.incoming_edges:
                            queue_sum += self._get_queue_length(e)
                        base = float(gst_snapshot.get('avg_gst', 0.0))
                        extension = base + 0.5 * queue_sum
                        extension = max(0.0, min(20.0, extension))
                        if extension > 0.5:
                            try:
                                traci.trafficlight.setPhaseDuration(self.tl_id, remaining + extension)
                            except Exception:
                                pass
            except Exception:
                pass
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self.get_state(), -10.0, True, {}
    
    def _set_traffic_light_phase(self, phase: int):
        try:
            if self.tl_id is None:
                return
            desired = int(phase) % max(1, self.num_phases)
            sim_time = traci.simulation.getTime()
            try:
                current = traci.trafficlight.getPhase(self.tl_id)
            except Exception:
                current = -1
            def is_green(p: int) -> bool:
                return p in (0, 2)

            def yellow_for_transition(src: int, dst: int) -> Optional[int]:
                if src == 0 and dst == 2:
                    return 1  # EW -> yellow
                if src == 2 and dst == 0:
                    return 3  # NS -> yellow
                return None

            if self._pending_target_phase is not None:
                remaining = 0.0
                try:
                    next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
                    remaining = max(0.0, next_switch - sim_time)
                except Exception:
                    pass
                if remaining <= 0.1:
                    traci.trafficlight.setPhase(self.tl_id, self._pending_target_phase)
                    try:
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.min_phase_duration_s)
                    except Exception:
                        pass
                    self.last_phase_switch_time = sim_time
                    self._pending_target_phase = None
                return

            if desired != current and (sim_time - self.last_phase_switch_time) >= self.min_phase_duration_s:
                if is_green(current) and is_green(desired) and current != desired:
                    y = yellow_for_transition(current, desired)
                    if y is not None:
                        traci.trafficlight.setPhase(self.tl_id, y)
                        try:
                            traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_duration_s)
                        except Exception:
                            pass
                        self.last_phase_switch_time = sim_time
                        self._pending_target_phase = desired
                        return
                traci.trafficlight.setPhase(self.tl_id, desired)
                try:
                    traci.trafficlight.setPhaseDuration(self.tl_id, self.min_phase_duration_s)
                except Exception:
                    pass
                self.last_phase_switch_time = sim_time
        except Exception as e:
            print(f"Error setting traffic light phase: {e}")
    
    def _calculate_reward(self) -> float:
        try:
            waits = 0.0
            queues = 0
            vehs = 0
            speeds = []
            
            for e in self.incoming_edges:
                waits += self._get_waiting_time(e)
                q = self._get_queue_length(e)
                queues += q
                vehs += self._get_vehicle_count(e)
                speeds.append(self._get_average_speed(e))

            self.total_waiting_time += waits
            self.total_vehicles += vehs
            self.queue_lengths.append(queues)
            
            avg_speed = sum(speeds) / max(1, len(speeds))
            wait_baseline = 50.0
            wait_component = (wait_baseline - min(waits, 100.0)) / 100.0 * 0.6
            queue_baseline = 5
            queue_component = (queue_baseline - min(queues, 10)) / 10.0 * 0.6
            speed_baseline = 7.0
            speed_component = (avg_speed - speed_baseline) / 14.0 * 0.4
            throughput_component = min(vehs, 5) * 0.02
            total_reward = wait_component + queue_component + speed_component + throughput_component
            total_reward = max(-1.0, min(1.0, total_reward))
            
            return total_reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0

    def get_current_phase_info(self) -> Dict:
        try:
            if self.tl_id is None:
                return {}
            phase_index = traci.trafficlight.getPhase(self.tl_id)
            next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
            sim_time = traci.simulation.getTime()
            remaining = max(0.0, next_switch - sim_time)
            duration = 0.0
            try:
                logics = traci.trafficlight.getAllProgramLogics(self.tl_id)
                if logics:
                    phases = getattr(logics[0], 'phases', None)
                    if phases is None:
                        phases = getattr(logics[0], 'getPhases', lambda: [])()
                    if phases and 0 <= phase_index < len(phases):
                        duration = float(getattr(phases[phase_index], 'duration', 0.0))
            except Exception:
                pass
            return {
                'phase': int(phase_index),
                'remaining_s': float(remaining),
                'duration_s': float(duration),
                'num_phases': int(self.num_phases)
            }
        except Exception:
            return {}
    
    def _get_detailed_lane_metrics(self) -> Dict:
        try:
            print("hgjh")
            detailed = {
                "per_lane_metrics": {},
                "lane_summary": {}
            }
            
            # Loop through each incoming road (edge)
            for edge_id in self.incoming_edges:
                if not edge_id:
                    continue

                # Here edge_id is already an edge, use it directly
                real_edge = edge_id

                # Collect all lane IDs that belong to this edge
                try:
                    all_lane_ids = traci.lane.getIDList()
                    lane_ids = [lid for lid in all_lane_ids if traci.lane.getEdgeID(lid) == real_edge]
                except Exception:
                    lane_ids = []

                # Loop each lane individually
                for lane_id in lane_ids:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    veh_count = len(veh_ids)

                    queue_length = 0
                    waiting_time = 0.0
                    vehicle_types = {}

                    for vid in veh_ids:
                        # Vehicle class
                        cls = self._map_vehicle_to_class(vid)
                        vehicle_types[cls] = vehicle_types.get(cls, 0) + 1

                        # Queue: speed < 0.1 e waiting
                        try:
                            speed = traci.vehicle.getSpeed(vid)
                            if speed < 0.1:
                                queue_length += 1
                        except Exception:
                            pass

                        # SUMO waiting time
                        try:
                            wt = traci.vehicle.getAccumulatedWaitingTime(vid)
                            waiting_time += wt
                        except Exception:
                            pass

                    # Occupancy
                    try:
                        occupancy = traci.lane.getLastStepOccupancy(lane_id)
                    except Exception:
                        occupancy = 0.0

                    # Speed
                    try:
                        avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                    except Exception:
                        avg_speed = 0.0

                    # Lane length
                    try:
                        length = traci.lane.getLength(lane_id)
                    except Exception:
                        length = 0.0

                    # GST from edge
                    try:
                        gst_snapshot = self._compute_green_signal_times()
                        gst_value = gst_snapshot.get("per_edge", {}).get(real_edge, 0.0)
                    except Exception:
                        gst_value = 0.0

                    # Store lane metrics
                    detailed["per_lane_metrics"][lane_id] = {
                        "edge_id": real_edge,
                        "lane_id": lane_id,
                        "vehicle_count": veh_count,
                        "queue_length": queue_length,
                        "waiting_time": waiting_time,
                        "average_speed": avg_speed,
                        "occupancy_percent": occupancy,
                        "green_signal_time": gst_value,
                        "vehicle_types": vehicle_types,
                        "lane_length": length
                    }

            # Summary over all lanes
            lanes = list(detailed["per_lane_metrics"].values())

            if lanes:
                detailed["lane_summary"] = {
                    "total_lanes": len(lanes),
                    "total_vehicles": sum(l["vehicle_count"] for l in lanes),
                    "total_queue_length": sum(l["queue_length"] for l in lanes),
                    "total_waiting_time": sum(l["waiting_time"] for l in lanes),
                    "average_speed": float(np.mean([l["average_speed"] for l in lanes])),
                    "average_occupancy": float(np.mean([l["occupancy_percent"] for l in lanes])),
                    "num_congested_lanes": len([l for l in lanes if l["queue_length"] > 3])
                }

            return detailed

        except Exception as e:
            print(f"Error in lane metrics: {e}")
            return {"per_lane_metrics": {}, "lane_summary": {}}

    def _get_lane_length(self, edge_id: str) -> float:
        try:
            return traci.edge.getLength(edge_id)
        except:
            return 0.0
    
    def _get_traffic_light_status(self, edge_id: str) -> str:
        try:
            if self.tl_id is None:
                return "Unknown"
            
            current_phase = traci.trafficlight.getPhase(self.tl_id)
            
            links = traci.trafficlight.getControlledLinks(self.tl_id)
            
            for link_group in links:
                for link in link_group:
                    if link and len(link) >= 3:
                        lane_id = link[0]
                        try:
                            if traci.lane.getEdgeID(lane_id) == edge_id:
                                signal_state = traci.trafficlight.getRedYellowGreenState(self.tl_id)
                                link_index = links.index(link_group)
                                
                                if link_index < len(signal_state):
                                    signal = signal_state[link_index]
                                    if signal == 'G':
                                        return "Green"
                                    elif signal == 'r':
                                        return "Red"
                                    elif signal == 'y':
                                        return "Yellow"
                                    else:
                                        return f"Unknown {signal}"
                        except:
                            continue
            
            return "Unknown"
        except:
            return "Error"
    
    def _get_average_speed(self, edge_id: str) -> float:
        try:
            return traci.edge.getLastStepMeanSpeed(edge_id)
        except:
            return 0.0
    
    def get_performance_metrics(self) -> Dict:
        metrics = {
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'average_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': max(self.queue_lengths) if self.queue_lengths else 0,
            'steps': self.step_count
        }
        try:
            if self._per_vehicle_wait:
                total_per_vehicle_wait = float(sum(self._per_vehicle_wait.values()))
                num_vehicles = len(self._per_vehicle_wait)
                metrics['avg_waiting_time_per_vehicle'] = total_per_vehicle_wait / max(1, num_vehicles)
            else:
                metrics['avg_waiting_time_per_vehicle'] = 0.0
        except Exception:
            metrics['avg_waiting_time_per_vehicle'] = 0.0
        
        detailed_lane_metrics = self._get_detailed_lane_metrics()
        metrics.update(detailed_lane_metrics)
        
        last_gst = self._compute_green_signal_times()
        metrics['green_signal_time'] = last_gst
        try:
            per_edge_sums: Dict[str, float] = {}
            per_edge_counts: Dict[str, int] = {}
            for snap in self.gst_history:
                for e, val in snap.get('per_edge', {}).items():
                    per_edge_sums[e] = per_edge_sums.get(e, 0.0) + float(val)
                    per_edge_counts[e] = per_edge_counts.get(e, 0) + 1
            avg_per_edge = {e: (per_edge_sums[e] / max(1, per_edge_counts[e])) for e in per_edge_sums}
            metrics['green_signal_time_avg_per_edge'] = avg_per_edge
            metrics['green_signal_time_history'] = self.gst_history[-200:]
        except Exception:
            pass
        return metrics
    
    def close(self):
        try:
            self.stop_simulation()
        except:
            pass
