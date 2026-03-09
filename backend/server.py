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

# Algorithm imports
from train_fedflow import FedFlowTrainer
from federated_learning.fl_client import TrafficFLClient
from federated_learning.fedcm_client import FedCMClient
from federated_learning.fedcm_server import FedCMServer
from federated_learning.fedkd_client import TrafficFedKDClient
from federated_learning.fedkd_server import TrafficFedKDServer
import numpy as np

app = FastAPI(title="Smart Traffic Control API")

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


@app.get("/api/config")
async def get_config():
    print("[API] GET /api/config requested by frontend")
    return {
        "cities": {
            city: {"lat": coords[0], "lon": coords[1]}
            for city, coords in CITY_COORDINATES.items()
        },
        "algorithms": ["Demo (No RL)", "FedFlow", "FedAvg", "FedCM", "FedKD"],
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


@app.post("/api/detect-pois")
async def detect_pois(request: DetectPoisRequest):
    """Detect POIs within radius_km of each intersection and assign priority tiers."""
    print(f"[API] POST /api/detect-pois for {len(request.intersections)} intersections")
    intersections = [ix.model_dump() for ix in request.intersections]
    results = detect_pois_for_intersections(intersections, radius_km=request.radius_km)
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


def get_latest_model(directory: str):
    """Find the most recently modified .pt model file in the given directory."""
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
        await websocket.send_json(payload)
        await asyncio.sleep(0.1)


async def run_multi_inference(websocket, envs_agents, intersections, max_steps=200):
    """
    Generic multi-intersection inference loop.
    envs_agents: dict of {nid: (env, agent)}
    """
    states = {}
    for nid, (env, agent) in envs_agents.items():
        states[nid] = env.reset()

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

        try:
            await websocket.send_json(payload)
        except Exception as e:
            print(f"[MultiInference] WebSocket send failed: {e}")
            break
        await asyncio.sleep(0.1)


def _create_envs_for_intersections(intersections, use_tomtom):
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
    for i, ix in enumerate(intersections):
        nid = f"node_{i}"
        client = TrafficFLClient(
            client_id=nid,
            sumo_config_path=f"sumo_configs2/osm_client{(i % 2) + 1}.sumocfg",
            gui=False,
            use_tomtom=use_tomtom,
            tomtom_city=config.get("city", "Delhi"),
            target_pois=[],
        )
        model_path = get_latest_model("results_federated")
        if model_path:
            try:
                client.load_model(model_path)
                print(f"[WS] Loaded FedAvg model for {nid} from {model_path}")
            except Exception as e:
                print(f"[WS] Error loading model for {nid}: {e}")
        envs_agents[nid] = (client.env, client.agent)

    await run_multi_inference(websocket, envs_agents, intersections)


async def run_fedflow_simulation(websocket, config):
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    num_nodes = len(intersections)

    trainer = FedFlowTrainer(
        num_nodes=num_nodes,
        num_clusters=max(1, num_nodes // 2),
        gui=False,
        use_tomtom=use_tomtom,
    )

    # Override the environments with intersection-specific ones
    envs = _create_envs_for_intersections(intersections, use_tomtom)
    trainer.envs = envs

    model_path = get_latest_model("results_fedflow")
    if model_path:
        for nid in trainer.agents:
            try:
                trainer.agents[nid].load_model(model_path)
            except Exception:
                pass
        print(f"[WS] Loaded FedFlow model from {model_path}")

    # Reset all
    states = {}
    for nid in trainer.agents:
        if nid in envs:
            states[nid] = envs[nid].reset()

    for step in range(200):
        node_data = {}
        total_reward = 0
        total_queue_all = 0

        for nid in list(trainer.agents.keys())[:num_nodes]:
            if nid not in states:
                continue
            agent = trainer.agents[nid]
            env = envs[nid]

            state_graph, adj_node = trainer._get_node_graph_state(nid, states[nid])
            action = agent.get_action(state_graph, adj_node)
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
        await websocket.send_json(payload)
        await asyncio.sleep(0.1)


async def run_fedcm_simulation(websocket, config):
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    city = config.get("city", "Delhi")

    envs_agents = {}
    for i, ix in enumerate(intersections):
        nid = f"node_{i}"
        client = FedCMClient(
            client_id=nid,
            sumo_config_path=f"sumo_configs2/osm_client{(i % 2) + 1}.sumocfg",
            agent_type="DQN",
            hidden_dims=[256, 128],
            use_tomtom=use_tomtom,
            tomtom_city=city,
            target_pois=[],
            gui=False,
        )
        model_path = get_latest_model("results_fedcm")
        if model_path:
            try:
                client.load_model(model_path)
                print(f"[WS] Loaded FedCM model for {nid} from {model_path}")
            except Exception as e:
                print(f"[WS] Error loading model for {nid}: {e}")
        envs_agents[nid] = (client.env, client.agent)

    await run_multi_inference(websocket, envs_agents, intersections)


async def run_fedkd_simulation(websocket, config):
    intersections = config.get("intersections", [])
    use_tomtom = config.get("use_tomtom", True)
    city = config.get("city", "Delhi")

    envs_agents = {}
    for i, ix in enumerate(intersections):
        nid = f"node_{i}"
        client = TrafficFedKDClient(
            client_id=nid,
            sumo_config_path=f"sumo_configs2/osm_client{(i % 2) + 1}.sumocfg",
            hidden_dims=[256, 128],
            use_tomtom=use_tomtom,
            tomtom_city=city,
            target_pois=[],
            gui=False,
        )
        model_path = get_latest_model("results_fedkd_sumo")
        if model_path:
            try:
                client.load_model(model_path)
                print(f"[WS] Loaded FedKD model for {nid} from {model_path}")
            except Exception as e:
                print(f"[WS] Error loading model for {nid}: {e}")
        envs_agents[nid] = (client.env, client.agent)

    await run_multi_inference(websocket, envs_agents, intersections)


@app.websocket("/api/simulate")
async def websocket_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        config_data = await websocket.receive_text()
        config = json.loads(config_data)

        algorithm = config.get("algorithm", "Demo (No RL)")
        intersections = config.get("intersections", [])

        print(
            f"[WS] Starting {algorithm} simulation with {len(intersections)} intersections"
        )

        if algorithm == "FedAvg":
            await run_fedavg_simulation(websocket, config)
        elif algorithm == "FedFlow":
            await run_fedflow_simulation(websocket, config)
        elif algorithm == "FedCM":
            await run_fedcm_simulation(websocket, config)
        elif algorithm == "FedKD":
            await run_fedkd_simulation(websocket, config)
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
