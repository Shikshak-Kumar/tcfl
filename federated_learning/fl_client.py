import numpy as np
import shutil
from typing import Dict, List, Tuple, Optional
import flwr as fl
from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment
import os
import json
import time


class TrafficFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        sumo_config_path: str,
        state_size: int = 12,
        action_size: int = 4,
        gui: bool = False,
        show_phase_console: bool = False,
        show_gst_gui: bool = False,
        use_tomtom: bool = False,
        tomtom_city: Optional[str] = None,
        target_pois: Optional[List[str]] = None,
    ):

        self.client_id = client_id
        self.config_path = sumo_config_path
        self.gui = gui
        self.show_phase_console = show_phase_console
        self.show_gst_gui = show_gst_gui
        self.use_tomtom = use_tomtom
        self.tomtom_city = tomtom_city
        self.target_pois = target_pois

        self.agent = DQNAgent(state_size, action_size)

        # Environment selection: CLI=Mock, GUI=SUMO
        if not gui:
            self.use_mock = True
            print(
                f"[{self.client_id}] CLI mode: Using high-fidelity MockTrafficEnvironment."
            )
        else:
            self.use_mock = False
            sumo_binary = "sumo-gui"
            if not shutil.which(sumo_binary):
                raise RuntimeError(
                    f"CRITICAL ERROR: 'sumo-gui' not found in PATH.\n"
                    f"GUI mode requires an installed SUMO simulator.\n"
                    f"To run without SUMO, omit the --gui flag to use Mock mode."
                )
            print(f"[{self.client_id}] GUI mode: Initializing real SUMO simulation.")

        if self.use_mock:
            if self.use_tomtom and self.tomtom_city:
                from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
                from utils.tomtom_api import CITY_COORDINATES
                lat, lon = CITY_COORDINATES[self.tomtom_city]
                self.env = TomTomTrafficEnvironment(
                    self.config_path,
                    tomtom_api_key=os.environ.get("TOMTOM_API_KEY", "oK2pgm45ieRxyEPgv876db2lGarwDFm2"),
                    lat=lat,
                    lon=lon,
                    gui=False,
                    show_phase_console=show_phase_console,
                    show_gst_gui=show_gst_gui,
                    max_vehicles=1000,
                    traffic_pattern="real_time",
                    target_pois=self.target_pois
                )
            else:
                from agents.mock_traffic_environment import MockTrafficEnvironment
                self.env = MockTrafficEnvironment(
                    sumo_config_path,
                    gui=False,
                    show_phase_console=show_phase_console,
                    show_gst_gui=show_gst_gui,
                    max_vehicles=1000,
                    traffic_pattern="rush_hour",
                )
        else:
            self.env = SUMOTrafficEnvironment(
                sumo_config_path,
                gui=True,
                show_phase_console=show_phase_console,
                show_gst_gui=show_gst_gui,
            )

        self.episodes_per_round = 2
        self.max_steps_per_episode = 1000

        self.training_history = []
        self.performance_metrics = []

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return self.agent.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]):
        self.agent.set_weights(parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        if parameters is not None:
            self.set_parameters(parameters)

        episodes = config.get("episodes", self.episodes_per_round)
        learning_rate = config.get("learning_rate", 0.001)

        self.agent.learning_rate = learning_rate
        for param_group in self.agent.optimizer.param_groups:
            param_group["lr"] = learning_rate

        training_metrics = self._train_agent(episodes)

        self.training_history.append(
            {
                "round": config.get("round", 0),
                "episodes": episodes,
                "metrics": training_metrics,
            }
        )

        # Filter out complex types like dicts and lists for Flower compatibility
        fl_metrics = {
            k: v
            for k, v in training_metrics.items()
            if isinstance(v, (int, float, str, bool, bytes))
        }

        print(f"[{self.client_id}] Sending training metrics to server: {fl_metrics}")

        return (
            self.get_parameters(config),
            episodes * self.max_steps_per_episode,
            fl_metrics,
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        evaluation_metrics = self._evaluate_agent()

        # ---- save full metrics locally (for analysis) ----
        try:
            os.makedirs("results", exist_ok=True)
            with open(f"results/{self.client_id}_eval.json", "w") as f:
                json.dump(evaluation_metrics, f, indent=2)
        except Exception:
            pass

        # ---- ONLY SEND SCALARS TO FLOWER ----
        safe_metrics = {
            "average_reward": float(evaluation_metrics.get("average_reward", 0.0)),
            "waiting_time": float(evaluation_metrics.get("waiting_time", 0.0)),
            "queue_length": float(evaluation_metrics.get("queue_length", 0.0)),
            "throughput": float(evaluation_metrics.get("throughput", 0)),
            "throughput_ratio": float(evaluation_metrics.get("throughput_ratio", 0.0)),
            "max_queue_length": float(evaluation_metrics.get("max_queue_length", 0.0)),
            "total_steps": int(evaluation_metrics.get("total_steps", 1)),
        }

        print(f"[{self.client_id}] Sending eval metrics:", safe_metrics)

        return (
            safe_metrics["average_reward"],
            safe_metrics["total_steps"],
            safe_metrics,
        )

    def _train_agent(self, episodes: int) -> Dict:
        total_reward = 0
        total_steps = 0
        losses = []

        total_waiting_time = 0
        total_queue_length = 0
        total_congested_lanes = 0
        total_lanes = 0

        for episode in range(episodes):
            if self.env:
                self.env.close()

            if self.use_mock:
                if self.use_tomtom and self.tomtom_city:
                    from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
                    from utils.tomtom_api import CITY_COORDINATES, get_api_key
                    lat, lon = CITY_COORDINATES[self.tomtom_city]
                    self.env = TomTomTrafficEnvironment(
                        self.config_path,
                        tomtom_api_key=get_api_key(),
                        lat=lat,
                        lon=lon,
                        gui=False,
                        show_phase_console=self.show_phase_console,
                        show_gst_gui=self.show_gst_gui,
                        max_vehicles=1000,
                        traffic_pattern="real_time",
                    )
                else:
                    from agents.mock_traffic_environment import MockTrafficEnvironment
                    self.env = MockTrafficEnvironment(
                        self.config_path,
                        gui=False,
                        show_phase_console=self.show_phase_console,
                        show_gst_gui=self.show_gst_gui,
                        max_vehicles=1000,
                        traffic_pattern="rush_hour",
                    )
            else:
                self.env = SUMOTrafficEnvironment(
                    self.config_path,
                    gui=True,
                    show_phase_console=self.show_phase_console,
                    show_gst_gui=self.show_gst_gui,
                )

            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(self.max_steps_per_episode):
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, done)

                loss = None
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    if loss is not None:
                        losses.append(loss)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            total_reward += episode_reward
            total_steps += episode_steps

            # collect metrics per episode
            perf = self.env.get_performance_metrics()
            lane_summary = perf.get("lane_summary", {})

            total_waiting_time += float(lane_summary.get("total_waiting_time", 0.0))
            total_queue_length += float(lane_summary.get("total_queue_length", 0.0))
            total_congested_lanes += float(lane_summary.get("num_congested_lanes", 0.0))
            total_lanes += float(lane_summary.get("total_lanes", 1.0))

        return {
            "average_reward": float(total_reward / max(episodes, 1)),
            "total_steps": int(total_steps),
            "average_loss": float(np.mean(losses) if losses else 0.0),
            "waiting_time": total_waiting_time / episodes,
            "queue_length": total_queue_length / episodes,
            "num_congested_lanes": total_congested_lanes / episodes,
            "total_lanes": total_lanes / episodes,
        }

    def _evaluate_agent(self) -> Dict:
        """Evaluation loop in simulation."""
        if self.env:
            self.env.close()

        if self.use_mock:
            if self.use_tomtom and self.tomtom_city:
                from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
                from utils.tomtom_api import CITY_COORDINATES
                lat, lon = CITY_COORDINATES[self.tomtom_city]
                self.env = TomTomTrafficEnvironment(
                    self.config_path,
                    tomtom_api_key="oK2pgm45ieRxyEPgv876db2lGarwDFm2",
                    lat=lat,
                    lon=lon,
                    gui=False,
                    show_phase_console=self.show_phase_console,
                    show_gst_gui=self.show_gst_gui,
                    target_pois=self.target_pois,
                    max_vehicles=1000,
                    traffic_pattern="real_time",
                )
            else:
                from agents.mock_traffic_environment import MockTrafficEnvironment
                self.env = MockTrafficEnvironment(
                    self.config_path,
                    gui=False,
                    show_phase_console=self.show_phase_console,
                    show_gst_gui=self.show_gst_gui,
                    max_vehicles=1000,
                    traffic_pattern="rush_hour",
                )
        else:
            self.env = SUMOTrafficEnvironment(
                self.config_path,
                gui=True,
                show_phase_console=self.show_phase_console,
                show_gst_gui=self.show_gst_gui,
            )
        state = self.env.reset()
        total_reward = 0
        total_steps = 0

        for step in range(self.max_steps_per_episode):
            action = self.agent.act(state, training=False)
            next_state, reward, done, info = self.env.step(action)

            state = next_state
            total_reward += reward
            total_steps += 1

            if done:
                break

        performance = self.env.get_performance_metrics()

        return {
            "total_reward": total_reward,
            "average_reward": total_reward / max(total_steps, 1),
            "total_steps": total_steps,
            "waiting_time": performance["total_waiting_time"],
            "queue_length": performance["average_queue_length"],
            "max_queue_length": performance["max_queue_length"],
            "throughput": performance.get("throughput", 0),
            "total_departed": performance.get("total_departed", 0),
            "throughput_ratio": performance.get("throughput_ratio", 0.0),
            "avg_waiting_time_per_vehicle": performance.get(
                "avg_waiting_time_per_vehicle", 0.0
            ),
            "green_signal_time": performance.get("green_signal_time", {}),
            "per_lane_metrics": performance.get("per_lane_metrics", {}),
            "lane_summary": performance.get("lane_summary", {}),
        }

    def save_training_history(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def save_performance_metrics(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.performance_metrics, f, indent=2)

    def get_client_info(self) -> Dict:
        return {
            "client_id": self.client_id,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "episodes_per_round": self.episodes_per_round,
            "max_steps_per_episode": self.max_steps_per_episode,
        }
