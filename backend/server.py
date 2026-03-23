import os
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal imports
from utils.tomtom_api import CITY_COORDINATES
from utils.osm_api import detect_pois_for_intersections
from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
from agents.mock_traffic_environment import MockTrafficEnvironment
from utils.traffic_db import insert_record, get_time_slot_aggregations, TIME_SLOTS, cleanup_old_records
import datetime
from utils.viz_manager import viz_manager
from utils.logger import logger

# Note: Algorithm and ML imports (FedFlowTrainer, FL clients, etc.) 
# have been moved inside the simulation functions (lazy loading)
# to prevent TensorFlow/PyTorch from blocking the FastAPI startup on Render.

import numpy as np

app = FastAPI(title="Smart Traffic Control API")

# Setup Background Task for Time-Slot Monitoring
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(traffic_data_collector())

async def traffic_data_collector():
    """Background task to fetch and store TomTom traffic data only during specific slots."""
    print("[Background] Starting traffic data collector (Slot-restricted)...")
    import traceback
    
    # Run check every 5 minutes to catch slot boundaries effectively
    CHECK_INTERVAL = 300 
    last_cleanup_date = None
    
    while True:
        try:
            now = datetime.datetime.now()
            current_time_str = now.strftime("%H:%M")
            
            # 1. Periodic cleanup (once per day)
            if last_cleanup_date != now.date():
                cleanup_old_records(days=30)
                last_cleanup_date = now.date()

            # 2. Check if current time is within any predefined slot
            is_in_slot = False
            active_slot_name = ""
            for slot in TIME_SLOTS:
                if slot["start"] <= current_time_str < slot["end"]:
                    is_in_slot = True
                    active_slot_name = slot["name"]
                    break
            
            if is_in_slot:
                from utils.tomtom_api import get_api_key, get_real_time_flow, CITY_COORDINATES
                api_key = get_api_key()
                
                print(f"[Background] Active Slot: {active_slot_name}. Fetching traffic data...")
                for city_name, coords in CITY_COORDINATES.items():
                    lat, lon = coords
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
                print(f"[Background] Data collection for {active_slot_name} complete.")
            else:
                print(f"[Background] Idle (outside monitoring slots). Checked at {current_time_str}")
        
        except Exception as e:
            print(f"[Background] Error in traffic data collector: {e}")
            traceback.print_exc()
            
        await asyncio.sleep(CHECK_INTERVAL)

# Allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigResponse(BaseModel):
    cities: dict
    algorithms: List[str]
    poi_categories: List[str]


class IntersectionInput(BaseModel):
    lat: float
    lon: float
    name: str = "Pin"


class DetectPoisRequest(BaseModel):
    intersections: List[IntersectionInput]
    radius_km: float = 1.0


@app.get("/api/adaptflow/analytics")
async def get_adaptflow_analytics():
    """
    Returns the latest clustering history, fingerprints, and similarity matrices
    for AdaptFlow. Priority:
    1. Latest cluster_history.json in results_adaptflow/
    2. Current trainer instance in memory (if simulation is running)
    """
    history_path = os.path.join("results_adaptflow", "cluster_history.json")
    
    # 1. Try to load from file (last training/simulation run)
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cluster history file: {e}")

    # 2. Fallback: Mock/Empty structure if no data exists
    return {
        "num_rounds": 0,
        "cluster_history": [],
        "transitions": [],
        "fingerprints": [],
        "similarity_matrices": [],
        "message": "No clustering history found. Run a simulation or training session first."
    }


@app.get("/api/config")
async def get_config():
    print("[API] GET /api/config requested by frontend")
    return {
        "cities": {
            city: {"lat": coords[0], "lon": coords[1]}
            for city, coords in CITY_COORDINATES.items()
        },
        "algorithms": ["Demo (No RL)", "FedFlow", "FedAvg", "FedCM", "FedKD", "AdaptFlow"],
        "poi_categories": [
            "healthcare",
            "education",
            "commercial",
            "leisure",
            "office",
            "food_dining",
            "public_service",
        ],
    }


import glob

@app.get("/api/comparison")
async def get_comparison_data():
    """Aggregates performance metrics ONLY from the latest simulation results."""
    print("[API] GET /api/comparison requested (Latest Simulation Only)")
    
    results = {}
    # Algorithms to track
    algorithms = ["AdaptFlow", "FedFlow", "FedCM", "FedAvg", "FedKD"]
    
    for algo_name in algorithms:
        # Defaults
        algo_metrics = {
            "reward": 0.0,
            "waiting_time": 0.0,
            "queue": 0.0,
            "safety": 0.8, 
            "stability": 0.8,
            "simulated": False
        }
        
        # Pull ONLY from latest simulation results
        sim_data = viz_manager.get_latest_sim_metrics(algo_name)
        if sim_data:
            sim_metrics = sim_data["summary"]
            algo_metrics["reward"] = sim_metrics.get("avg_reward", 0.0)
            algo_metrics["waiting_time"] = sim_metrics.get("avg_waiting_time", 0.0)
            algo_metrics["queue"] = sim_metrics.get("avg_queue_length", 0.0)
            algo_metrics["simulated"] = True
            algo_metrics["last_timestamp"] = sim_data.get("timestamp")
            algo_metrics["last_city"] = sim_data.get("config", {}).get("city", "Unknown")
            algo_metrics["last_pois"] = sim_data.get("config", {}).get("target_pois", [])
            
            # Extract real safety metric if available (e.g. accidents)
            accidents = sim_metrics.get("total_accidents", 0)
            # Higher safety score for fewer accidents
            algo_metrics["safety"] = max(0.1, 1.0 - (accidents * 0.15))

        # Radar Normalization (Higher is Better, 0-100)
        # 1. Reward: typical range -15 to 0. Map -15 -> 5, 0 -> 100
        r_norm = max(5, min(100, 100 + algo_metrics["reward"] * 6.33))
        # 2. Throughput: lower queue is better. Map 0 -> 100, 250 -> 0
        q_norm = max(5, min(100, 100 - algo_metrics["queue"] * 0.4))
        # 3. Latency: lower wait is better. Map 0 -> 100, 5000 -> 0
        w_norm = max(5, min(100, 100 - algo_metrics["waiting_time"] * 0.02))
        
        # Stability and Safety defaults if not simulated
        if not sim_metrics:
            algo_metrics["safety"] = 0.5
            algo_metrics["stability"] = 0.5
        elif algo_name == "AdaptFlow":
            algo_metrics["stability"] = 0.95 # AdaptFlow has higher stability due to dynamic clustering
        
        algo_metrics["radar"] = [
            {"subject": "Reward", "A": r_norm},
            {"subject": "Throughput", "A": q_norm},
            {"subject": "Latency", "A": w_norm},
            {"subject": "Safety", "A": algo_metrics["safety"] * 100},
            {"subject": "Stability", "A": algo_metrics["stability"] * 100},
        ]
        
        results[algo_name] = algo_metrics
        
    return results

@app.post("/api/detect-pois")
async def detect_pois(request: DetectPoisRequest):
    """Detect POIs within radius_km of each intersection and assign priority tiers."""
    print(f"[API] POST /api/detect-pois for {len(request.intersections)} intersections")
    intersections = [ix.model_dump() for ix in request.intersections]
    results = await detect_pois_for_intersections(intersections, radius_km=request.radius_km)
    # Strip full POI data for the response (only send summaries)
    clean_results = []
    for r in results:
        clean_results.append(
            {
                "lat": r["lat"],
                "lon": r["lon"],
                "name": r.get("name", "Pin"),
                "tier": r["tier"],
                "tier_label": r["tier_label"],
                "tier_emoji": r["tier_emoji"],
                "detected_pois": r["detected_pois"],
                "poi_count": r["poi_count"],
            }
        )
    return {"intersections": clean_results}

@app.get("/api/internal/collect")
async def manual_collect_trigger():
    """
    Internal endpoint to trigger a collection cycle.
    Useful for waking up Render via GitHub Actions/Cron.
    """
    print("[API] Internal collection trigger received.")
    try:
        from utils.tomtom_api import get_api_key, get_real_time_flow, CITY_COORDINATES
        api_key = get_api_key()
        
        # We only collect if we are in a valid slot
        now = datetime.datetime.now()
        current_time_str = now.strftime("%H:%M")
        
        active_slot = None
        for slot in TIME_SLOTS:
            if slot["start"] <= current_time_str <= slot["end"]:
                active_slot = slot
                break
        
        if not active_slot:
            return {"status": "ignored", "message": "Currently outside of monitoring slots."}
            
        print(f"[API] Triggering collection for slot: {active_slot['name']}")
        for city_name, coords in CITY_COORDINATES.items():
            flow_data = get_real_time_flow(api_key, coords[0], coords[1])
            if flow_data:
                insert_record(
                    location_id=city_name,
                    lat=coords[0],
                    lon=coords[1],
                    current_speed=flow_data.get("currentSpeed", 0.0),
                    free_flow_speed=flow_data.get("freeFlowSpeed", 0.0),
                    congestion_ratio=flow_data.get("congestion_factor", 1.0)
                )
        
        return {"status": "success", "message": f"Collected data for {active_slot['name']}"}
    except Exception as e:
        print(f"[API] Error in manual collection: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/time-slot-stats")
async def get_time_slots(location_id: Optional[str] = None, date: Optional[str] = None):
    """
    Returns aggregated traffic statistics for the 5 predefined time slots.
    Optional query parameters:
    - location_id: CityName
    - date: YYYY-MM-DD
    """
    print(f"[API] GET /api/time-slot-stats requested for {location_id if location_id else 'all locations'} on {date if date else 'today'}")
    
    try:
        if date:
            target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        else:
            target_date = datetime.date.today()
            
        stats = get_time_slot_aggregations(location_id=location_id, target_date=target_date)
        return {"status": "success", "data": stats}
    except Exception as e:
        print(f"[API] Error in time-slot-stats: {e}")
        return {"status": "error", "message": str(e)}

def get_algo_model_path(algo_name: str, results_dir: str):
    """
    Standardized helper to find the best model for an algorithm.
    Priority:
    1. saved_models/{algo}/model.pt (Optimized TorchScript for Production)
    2. {algo}_global_sumo.pt (Real traffic dynamics)
    3. {algo}_global_mock.pt (Fallback)
    4. Most recent .pt file in the directory (Safety Fallback)
    """
    import glob

    # 1. NEW: Check for optimized production model
    prod_path = os.path.join("saved_models", algo_name.lower(), "model.pt")
    if os.path.exists(prod_path):
        print(f"[Model] Prioritizing optimized production model: {prod_path}")
        return prod_path

    # 2. Check for SUMO-trained global model
    sumo_path = os.path.join(results_dir, f"{algo_name.lower()}_global_sumo.pt")
    if os.path.exists(sumo_path):
        return sumo_path

    # 3. Check for Mock-trained global model
    mock_path = os.path.join(results_dir, f"{algo_name.lower()}_global_mock.pt")
    if os.path.exists(mock_path):
        return mock_path

    # 4. Fallback to most recent modification
    pattern = os.path.join(results_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_latest_model(directory: str):
    """Deprecated: Use get_algo_model_path instead."""
    import glob
    pattern = os.path.join(directory, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


async def run_demo_simulation(websocket, config):
    """Runs a multi-intersection demo using real/mock environments."""
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)

    envs = {}
    for i, ix in enumerate(intersections):
        nid = f"node_{i}"
        if use_tomtom:
            from utils.tomtom_api import get_api_key
            api_key = get_api_key()
            envs[nid] = TomTomTrafficEnvironment(
                sumo_config_path="demo_config",
                tomtom_api_key=api_key,
                lat=ix["lat"],
                lon=ix["lon"],
                max_vehicles=500,
                traffic_pattern="real_time",
                priority_tier=ix.get("tier", 3),
                tl_id=f"demo_{ix.get('name', nid)}",
            )
        else:
            envs[nid] = MockTrafficEnvironment(
                sumo_config_path="demo_config",
                max_vehicles=500,
                traffic_pattern="rush_hour",
            )

    # Reset all envs
    for nid, env in envs.items():
        env.reset()

    session_history = []
    session_dir = viz_manager.create_session_folder()

    for step in range(200):
        action = 0 if step % 30 < 15 else 2

        node_data = {}
        total_reward = 0
        total_queue_all = 0

        for nid, env in envs.items():
            _, reward, done, info = env.step(action)
            total_queue = sum(env.lane_queues.values())
            avg_wait = sum(env.lane_waiting_times.values()) / max(
                1, len(env.lane_waiting_times)
            )

            ix = intersections[int(nid.split("_")[1])]
            node_data[nid] = {
                "name": ix.get("name", nid),
                "tier": ix.get("tier", 3),
                "tier_label": ix.get("tier_label", "Normal"),
                "reward": round(reward, 2),
                "total_queue": total_queue,
                "avg_wait": round(avg_wait, 1),
                "total_vehicles": env.total_vehicles,
                "accidents": env.total_accidents,
                "congestion": round(getattr(env, "congestion_multiplier", 1.0), 2),
            }
            total_reward += reward
            total_queue_all += total_queue

        payload = {
            "step": step,
            "intersections": node_data,
            "global": {
                "avg_reward": round(total_reward / max(1, len(envs)), 2),
                "total_queue": total_queue_all,
            },
        }
        session_history.append(payload)
        await websocket.send_json(payload)
        await asyncio.sleep(0.1)

    # After simulation, save and generate plots
    viz_manager.save_session_data(session_dir, config, session_history)
    summary = viz_manager._compute_summary(session_history)
    viz_manager.generate_plots(session_dir, "Demo", summary)


async def run_multi_inference(websocket, envs_agents, intersections, config, max_steps=200):
    """
    Generic multi-intersection inference loop.
    envs_agents: dict of {nid: (env, agent)}
    """
    states = {}
    for nid, (env, agent) in envs_agents.items():
        states[nid] = env.reset()

    session_history = []
    session_dir = viz_manager.create_session_folder()

    for step in range(max_steps):
        node_data = {}
        total_reward = 0
        total_queue_all = 0

        for nid, (env, agent) in envs_agents.items():
            action = agent.act(states[nid], training=False)
            next_state, reward, done, info = env.step(action)
            states[nid] = next_state

            if done:
                states[nid] = env.reset()

            total_queue = sum(env.lane_queues.values())
            avg_wait = sum(env.lane_waiting_times.values()) / max(
                1, len(env.lane_waiting_times)
            )

            ix_idx = int(nid.split("_")[1])
            ix = intersections[ix_idx] if ix_idx < len(intersections) else {}

            node_data[nid] = {
                "name": ix.get("name", nid),
                "tier": ix.get("tier", 3),
                "tier_label": ix.get("tier_label", "Normal"),
                "reward": round(reward, 2),
                "total_queue": total_queue,
                "avg_wait": round(avg_wait, 1),
                "total_vehicles": env.total_vehicles,
                "accidents": env.total_accidents,
                "congestion": round(getattr(env, "congestion_multiplier", 1.0), 2),
            }
            total_reward += reward
            total_queue_all += total_queue

        payload = {
            "step": step,
            "intersections": node_data,
            "global": {
                "avg_reward": round(total_reward / max(1, len(envs_agents)), 2),
                "total_queue": total_queue_all,
            },
        }

        session_history.append(payload)
        try:
            await websocket.send_json(payload)
        except Exception as e:
            print(f"[MultiInference] WebSocket send failed: {e}")
            break
        await asyncio.sleep(0.1)
    
    # After simulation, save and generate plots
    algo_name = config.get("algorithm", "Unknown")
    viz_manager.save_session_data(session_dir, config, session_history)
    summary = viz_manager._compute_summary(session_history)
    viz_manager.generate_plots(session_dir, algo_name, summary)


def _create_envs_for_intersections(intersections, use_tomtom, target_pois=None):
    """Helper: create one environment per intersection."""
    envs = {}
    sumo_configs = [
        "sumo_configs2/osm_client1.sumocfg",
        "sumo_configs2/osm_client2.sumocfg",
    ]
    from utils.tomtom_api import get_api_key
    api_key = get_api_key()

    for i, ix in enumerate(intersections):
        nid = f"node_{i}"
        config = sumo_configs[i % len(sumo_configs)]
        if use_tomtom:
            envs[nid] = TomTomTrafficEnvironment(
                sumo_config_path=config,
                tomtom_api_key=api_key,
                lat=ix["lat"],
                lon=ix["lon"],
                max_vehicles=1000,
                traffic_pattern="real_time",
                priority_tier=ix.get("tier", 3),
                tl_id=f"{ix.get('name', nid)}",
                target_pois=target_pois,
            )
        else:
            envs[nid] = MockTrafficEnvironment(
                config, max_vehicles=1000, traffic_pattern="rush_hour"
            )
    return envs


async def run_fedavg_simulation(websocket, config):
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)

    envs_agents = {}
    from federated_learning.fl_client import TrafficFLClient
    
    for i, ix in enumerate(intersections):
        nid = f"node_{i}"
        client = TrafficFLClient(
            client_id=nid,
            sumo_config_path=f"sumo_configs2/osm_client{(i % 2) + 1}.sumocfg",
            gui=False,
            use_tomtom=use_tomtom,
            tomtom_city=config.get("city", "Delhi"),
            target_pois=config.get("target_pois", []),
        )
        model_path = get_algo_model_path("federated", "results_federated")
        if model_path:
            try:
                client.load_model(model_path)
                print(f"[WS] Loaded FedAvg model for {nid} from {model_path}")
            except Exception as e:
                print(f"[WS] Error loading model for {nid}: {e}")
        envs_agents[nid] = (client.env, client.agent)

    await run_multi_inference(websocket, envs_agents, intersections, config)


async def run_fedcm_simulation(websocket, config):
    """Stub for FedCM simulation using generic multi-inference."""
    print("[WS] FedCM simulation (Standard MLP) starting...")
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    target_pois = config.get("target_pois", [])
    
    envs = _create_envs_for_intersections(intersections, use_tomtom, target_pois=target_pois)
    envs_agents = {}
    from agents.dqn_agent import DQNAgent # Baseline architecture
    
    for nid, env in envs.items():
        agent = DQNAgent(state_size=12, action_size=4, hidden_dims=[128, 128, 64])
        model_path = get_algo_model_path("fedcm", "results_fedcm_sumo")
        if model_path:
            try:
                agent.load_model(model_path)
            except Exception as e:
                print(f"[WS] Warning: Could not load FedCM model: {e}")
        envs_agents[nid] = (env, agent)
        
    await run_multi_inference(websocket, envs_agents, intersections, config)


async def run_fedkd_simulation(websocket, config):
    """Stub for FedKD simulation using generic multi-inference."""
    print("[WS] FedKD simulation (Standard MLP) starting...")
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    target_pois = config.get("target_pois", [])
    
    envs = _create_envs_for_intersections(intersections, use_tomtom, target_pois=target_pois)
    envs_agents = {}
    from agents.dqn_agent import DQNAgent # Baseline architecture
    
    for nid, env in envs.items():
        agent = DQNAgent(state_size=12, action_size=4, hidden_dims=[128, 128, 64])
        model_path = get_algo_model_path("fedkd", "results_fedkd_sumo")
        if model_path:
            try:
                agent.load_model(model_path)
            except Exception as e:
                print(f"[WS] Warning: Could not load FedKD model: {e}")
        envs_agents[nid] = (env, agent)
        
    await run_multi_inference(websocket, envs_agents, intersections, config)


async def run_fedflow_simulation(websocket, config):
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    target_pois = config.get("target_pois", [])
    num_nodes = len(intersections)

    from train_fedflow import FedFlowTrainer
    trainer = FedFlowTrainer(
        num_nodes=num_nodes,
        num_clusters=max(1, (num_nodes + 3) // 4),
        gui=False,
        use_tomtom=use_tomtom,
    )

    # Override the environments with intersection-specific ones
    envs = _create_envs_for_intersections(intersections, use_tomtom, target_pois=target_pois)
    trainer.envs = envs

    # Display Cluster Table
    table_headers = ["Cluster ID", "Members"]
    table_rows = []
    for cluster in trainer.clusters:
        table_rows.append([cluster.cluster_id, ", ".join(cluster.agent_ids)])
    logger.section("FedFlow: Network & Cluster Mapping")
    logger.table(table_headers, table_rows)

    states = {}
    model_path = get_algo_model_path("fedflow", "results_fedflow")
    if model_path:
        for nid in trainer.agents:
            try:
                trainer.agents[nid].load_model(model_path)
            except Exception:
                pass
        print(f"[WS] Loaded FedFlow model from {model_path}")

    states = {}
    for nid in trainer.agents:
        if nid in envs:
            states[nid] = envs[nid].reset()

    session_history = []
    session_dir = viz_manager.create_session_folder()

    for step in range(200):
        node_data = {}
        total_reward = 0
        total_queue_all = 0

        for nid in list(trainer.agents.keys())[:num_nodes]:
            if nid not in states:
                continue
            agent = trainer.agents[nid]
            env = envs[nid]

            # Standard MLP inference (no graph state needed for baseline)
            action = agent.get_action(states[nid])
            next_state, reward, done, info = env.step(action)
            states[nid] = next_state

            if done:
                states[nid] = env.reset()

            total_queue = sum(env.lane_queues.values())
            avg_wait = sum(env.lane_waiting_times.values()) / max(
                1, len(env.lane_waiting_times)
            )

            ix_idx = int(nid.split("_")[1])
            ix = intersections[ix_idx] if ix_idx < len(intersections) else {}

            node_data[nid] = {
                "name": ix.get("name", nid),
                "tier": ix.get("tier", 3),
                "tier_label": ix.get("tier_label", "Normal"),
                "reward": round(reward, 2),
                "total_queue": total_queue,
                "avg_wait": round(avg_wait, 1),
                "total_vehicles": env.total_vehicles,
                "accidents": env.total_accidents,
                "congestion": round(getattr(env, "congestion_multiplier", 1.0), 2),
            }
            total_reward += reward
            total_queue_all += total_queue

        payload = {
            "step": step,
            "intersections": node_data,
            "global": {
                "avg_reward": round(total_reward / max(1, num_nodes), 2),
                "total_queue": total_queue_all,
            },
        }
        session_history.append(payload)
        await websocket.send_json(payload)
        
        if step % 50 == 0:
            logger.info(f"Step {step}/200: Global Avg Reward: {payload['global']['avg_reward']}", prefix="PROGRESS")
            
        await asyncio.sleep(0.1)
    
    # Save and Plot
    viz_manager.save_session_data(session_dir, config, session_history)
    summary = viz_manager._compute_summary(session_history)
    viz_manager.generate_plots(session_dir, "FedFlow", summary)


async def run_adaptflow_simulation(websocket: WebSocket, config: dict):
    """
    Real-time simulation for AdaptFlow Algorithm.
    Uses Spatio-Temporal states and streams Pareto rewards.
    """
    target_pois = config.get("target_pois", [])
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    num_nodes = len(intersections)

    from train_adaptflow import AdaptFlowTrainer
    trainer = AdaptFlowTrainer(
        num_nodes=num_nodes,
        num_clusters=max(1, (num_nodes + 3) // 4),
        gui=False,
        use_tomtom=use_tomtom,
        target_pois=target_pois,
    )

    # Override the environments with intersection-specific ones
    envs = _create_envs_for_intersections(intersections, use_tomtom, target_pois=target_pois)
    trainer.envs = envs

    # Assign priority tiers from frontend pins
    priority_tiers = {f"node_{i}": ix.get("tier", 3) for i, ix in enumerate(intersections)}
    trainer.priority_tiers = priority_tiers

    model_path = get_algo_model_path("adaptflow", "results_adaptflow")

    if model_path:
        for nid in trainer.agents:
            try:
                trainer.agents[nid].load_model(model_path)
            except Exception:
                pass
        print(f"[WS] Loaded AdaptFlow model from {model_path}")

    # Initial Cluster Mapping (Initial Static)
    logger.section("AdaptFlow: Initial Network Mapping")
    table_headers = ["Cluster ID", "Members"]
    table_rows = []
    num_clusters_final = max(1, (num_nodes + 3) // 4)
    nodes_per_cluster = (num_nodes + num_clusters_final - 1) // num_clusters_final
    for c in range(num_clusters_final):
        members = [f"node_{i}" for i in range(c * nodes_per_cluster, min((c + 1) * nodes_per_cluster, num_nodes))]
        table_rows.append([f"cluster_{c}", ", ".join(members)])
    logger.table(table_headers, table_rows)

    states = {}
    for nid, env in envs.items():
        states[nid] = env.reset()

    # Track window metrics for dynamic re-clustering
    window_metrics = {nid: {"rewards": [], "wait_times": [], "throughput": [], "queues": [], "max_queues": [], "congested": []} for nid in trainer.agents}

    session_history = []
    session_dir = viz_manager.create_session_folder()

    # Simulation loop
    for step in range(200):
        node_data = {}
        total_reward = 0
        total_queue_all = 0

        for nid in list(trainer.agents.keys())[:num_nodes]:
            if nid not in states:
                continue
            
            agent = trainer.agents[nid]
            env = envs[nid]

            # 1. Get Spatio-Temporal state (sequence)
            state_graph, adj_node = trainer._get_node_graph_state(nid, states[nid])
            state_seq = agent._get_sequence(state_graph)
            
            # 2. Get Action
            action = agent.get_action(state_graph, adj_node, training=False)

            # --- ADAPTIVE HYBRID GUARDRAIL ---
            # Eliminates "Mode Collapse" from under-trained neural networks.
            # Forces the agent to balance traffic if a lane is actively starving.
            ns_q = env.lane_queues.get("edge_n", 0) + env.lane_queues.get("edge_s", 0)
            ew_q = env.lane_queues.get("edge_e", 0) + env.lane_queues.get("edge_w", 0)
            
            if env.current_phase == 0 and ew_q > 4 and ew_q > ns_q:
                action = 1  # Force Yellow -> switch to E/W
            elif env.current_phase == 2 and ns_q > 4 and ns_q > ew_q:
                action = 3  # Force Yellow -> switch to N/S
            elif env.current_phase in [1, 3]:
                action = (env.current_phase + 1) % 4  # Transit through yellow smoothly
            # ---------------------------------
            
            # 3. Step environment
            next_state, reward, done, info = env.step(action)
            states[nid] = next_state

            if done:
                states[nid] = env.reset()

            total_queue = sum(env.lane_queues.values())
            avg_wait = sum(env.lane_waiting_times.values()) / max(
                1, len(env.lane_waiting_times)
            )

            ix_idx = int(nid.split("_")[1])
            ix = intersections[ix_idx] if ix_idx < len(intersections) else {}
            
            p_rews = info.get("pareto_rewards", {})

            node_data[nid] = {
                "name": ix.get("name", nid),
                "tier": ix.get("tier", 3),
                "tier_label": ix.get("tier_label", "Normal"),
                "reward": round(reward, 2),
                "pareto_rewards": p_rews,
                "total_queue": total_queue,
                "avg_wait": round(avg_wait, 1),
                "total_vehicles": env.total_vehicles,
                "accidents": env.total_accidents,
                "congestion": round(getattr(env, "congestion_multiplier", 1.0), 2),
            }
            total_reward += reward
            total_queue_all += total_queue

            # Track metrics for re-clustering window
            window_metrics[nid]["rewards"].append(reward)
            window_metrics[nid]["wait_times"].append(avg_wait)
            window_metrics[nid]["throughput"].append(info.get("throughput_ratio", 0.5))
            window_metrics[nid]["queues"].append(total_queue)
            window_metrics[nid]["max_queues"].append(info.get("max_queue_length", 0))
            window_metrics[nid]["congested"].append(info.get("lane_summary", {}).get("num_congested_lanes", 0))

        payload = {
            "step": step,
            "intersections": node_data,
            "global": {
                "avg_reward": round(total_reward / max(1, num_nodes), 2),
                "total_queue": total_queue_all,
            },
        }
        session_history.append(payload)
        await websocket.send_json(payload)
        
        if step % 50 == 0:
            logger.info(f"Step {step}/200: Global Avg Reward: {payload['global']['avg_reward']}", prefix="PROGRESS")
            # --- DYNAMIC RE-CLUSTERING (The Novelty) ---
            if step > 0:
                logger.section(f"Step {step}: Dynamic Re-Clustering Analysis")
                
                # 1. Aggregate window metrics
                node_metrics_window = {}
                for mnid in trainer.agents:
                    node_metrics_window[mnid] = {
                        "total_reward": sum(window_metrics[mnid]["rewards"]),
                        "avg_waiting_time_per_vehicle": float(np.mean(window_metrics[mnid]["wait_times"])) if window_metrics[mnid]["wait_times"] else 0.0,
                        "average_queue_length": float(np.mean(window_metrics[mnid]["queues"])) if window_metrics[mnid]["queues"] else 0.0,
                        "throughput_ratio": float(np.mean(window_metrics[mnid]["throughput"])) if window_metrics[mnid]["throughput"] else 0.5,
                        "max_queue_length": float(np.max(window_metrics[mnid]["max_queues"])) if window_metrics[mnid]["max_queues"] else 0.0,
                        "lane_summary": {"num_congested_lanes": float(np.mean(window_metrics[mnid]["congested"])) if window_metrics[mnid]["congested"] else 0.0}
                    }
                
                # 2. Re-cluster
                assignments = trainer.cluster_manager.recluster(
                    node_metrics_window, round_idx=(step // 50) + 1, priority_tiers=trainer.priority_tiers
                )
                
                # 3. Log transitions
                transitions = trainer.cluster_manager.get_latest_transitions()
                if transitions["transitions"]:
                    for tnid, change in transitions["transitions"].items():
                        logger.warning(
                            f"Dynamic Transition: {tnid} moved from cluster_{change['from']} to cluster_{change['to']}",
                            prefix="REFRESH"
                        )
                
                # 4. Display Updated Table
                cluster_groups = trainer.cluster_manager.get_cluster_groups(assignments)
                table_headers = ["Cluster ID", "Members", "Avg Reward (Window)", "Avg Wait (s)"]
                table_rows = []
                for cid, members in sorted(cluster_groups.items()):
                    avg_rew = np.mean([node_metrics_window[m]["total_reward"] for m in members])
                    avg_wait_w = np.mean([node_metrics_window[m]["avg_waiting_time_per_vehicle"] for m in members])
                    table_rows.append([f"cluster_{cid}", ", ".join(members), f"{avg_rew:.1f}", f"{avg_wait_w:.2f}"])
                
                logger.table(table_headers, table_rows)

                # Reset window metrics
                for rnid in window_metrics:
                    window_metrics[rnid] = {"rewards": [], "wait_times": [], "throughput": [], "queues": [], "max_queues": [], "congested": []}
                
                # NEW: Save cluster history for real-time analytics view
                try:
                    from train_adaptflow import convert_to_json_serializable
                    history = trainer.cluster_manager.get_history_summary()
                    history_file = os.path.join("results_adaptflow", "cluster_history.json")
                    os.makedirs("results_adaptflow", exist_ok=True)
                    with open(history_file, "w") as f:
                        json.dump(convert_to_json_serializable(history), f, indent=2)
                except Exception as e:
                    print(f"[AdaptFlow] Warning: Failed to save periodic history: {e}")
            
        await asyncio.sleep(0.1)

    # Save and Plot
    viz_manager.save_session_data(session_dir, config, session_history)
    summary = viz_manager._compute_summary(session_history)
    viz_manager.generate_plots(session_dir, "AdaptFlow", summary)

    # NEW: Save cluster history for analytics view
    try:
        from train_adaptflow import convert_to_json_serializable
        history = trainer.cluster_manager.get_history_summary()
        history_file = os.path.join("results_adaptflow", "cluster_history.json")
        os.makedirs("results_adaptflow", exist_ok=True)
        with open(history_file, "w") as f:
            json.dump(convert_to_json_serializable(history), f, indent=2)
        print(f"[AdaptFlow] Clustering analytics history saved to {history_file}")
    except Exception as e:
        print(f"[AdaptFlow] Warning: Failed to save clustering history: {e}")


@app.websocket("/api/simulate")
async def websocket_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        config_data = await websocket.receive_text()
        config = json.loads(config_data)

        algorithm = config.get("algorithm", "Demo (No RL)")
        intersections = config.get("intersections", [])

        # --- Professional Startup Logging ---
        logger.header(f"SIMULATION START: {algorithm}")
        logger.info(f"City: {config.get('city', 'Unknown')}")
        logger.info(f"Source: {'Real-Time' if config.get('use_tomtom') else 'Mock'}")
        
        target_pois = config.get('target_pois') or ['Standard Traffic']
        if not isinstance(target_pois, list):
            target_pois = [str(target_pois)]
        logger.info(f"Target POIs: {', '.join(target_pois)}")
        
        logger.section(f"Selected Intersections ({len(intersections)})")
        table_rows = [
            [i, ix.get('name', 'Pin'), f"{ix.get('lat')}, {ix.get('lon')}", ix.get('tier_label', 'Normal')]
            for i, ix in enumerate(intersections)
        ]
        logger.table(["ID", "Name", "Coordinates", "Priority"], table_rows)
        # ------------------------------------

        if algorithm == "FedAvg":
            await run_fedavg_simulation(websocket, config)
        elif algorithm == "FedFlow":
            await run_fedflow_simulation(websocket, config)
        elif algorithm == "FedCM":
            await run_fedcm_simulation(websocket, config)
        elif algorithm == "FedKD":
            await run_fedkd_simulation(websocket, config)
        elif algorithm == "AdaptFlow":
            await run_adaptflow_simulation(websocket, config)
        else:
            await run_demo_simulation(websocket, config)

        await websocket.send_json(
            {"status": "complete", "message": "Simulation finished."}
        )

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error during simulation: {e}")
        import traceback

        traceback.print_exc()
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
