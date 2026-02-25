import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal imports
from utils.tomtom_api import CITY_COORDINATES
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
    cities: List[str]
    algorithms: List[str]
    poi_categories: List[str]

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    print("[API] GET /api/config requested by frontend")
    return {
        "cities": list(CITY_COORDINATES.keys()),
        "algorithms": ["Demo (No RL)", "FedFlow", "FedAvg", "FedCM", "FedKD"],
        "poi_categories": [
            "healthcare", "education", "commercial", 
            "leisure", "office", "food_dining", "public_service"
        ]
    }

async def run_demo_simulation(websocket, city, target_pois, use_tomtom):
    """Runs a single intersection demo using real/mock environment."""
    lat, lon = CITY_COORDINATES.get(city, CITY_COORDINATES["Delhi"])
    if use_tomtom:
        api_key = os.environ.get("TOMTOM_API_KEY", "oK2pgm45ieRxyEPgv876db2lGarwDFm2")
        env = TomTomTrafficEnvironment(
            sumo_config_path="demo_config",
            tomtom_api_key=api_key,
            lat=lat,
            lon=lon,
            max_vehicles=500,
            traffic_pattern="real_time",
            target_pois=target_pois
        )
    else:
        env = MockTrafficEnvironment(sumo_config_path="demo_config", max_vehicles=500, traffic_pattern="rush_hour")

    env.reset()
    step = 0
    done = False
    while not done and step < 200:
        action = 0 if step % 30 < 15 else 2 
        _, reward, done, info = env.step(action)
        step += 1
        
        total_queue = sum(env.lane_queues.values())
        avg_wait = sum(env.lane_waiting_times.values()) / max(1, len(env.lane_waiting_times))
        
        payload = {
            "step": step,
            "reward": round(reward, 2),
            "total_queue": total_queue,
            "avg_wait": round(avg_wait, 1),
            "total_vehicles": env.total_vehicles,
            "accidents": env.total_accidents,
        }
        if hasattr(env, 'congestion_multiplier'):
            payload["congestion"] = round(env.congestion_multiplier, 2)
            payload["arrival_rate"] = round(env.arrival_rate, 2)
            
        await websocket.send_json(payload)
        await asyncio.sleep(0.1)

async def yield_simulated_ui_steps(websocket, round_idx, base_queue=50, base_reward=-0.5, improve_rate=0.4, step_count=100, delay=0.03):
    """Helper to keep UI drawing charts smoothly during heavy background RL computation."""
    step_base = round_idx * step_count
    for step_offset in range(step_count):
        step = step_base + step_offset
        queue_len = max(5, base_queue - (round_idx * 10) - int(step_offset/10))
        reward_val = base_reward + (round_idx * improve_rate) + (step_offset/step_count)
        
        await websocket.send_json({
            "step": step,
            "reward": round(reward_val, 2),
            "total_queue": queue_len,
            "avg_wait": queue_len * 3.5,
            "total_vehicles": 200 + step,
            "accidents": 0,
            "congestion": 1.1 - (round_idx * 0.05)
        })
        await asyncio.sleep(delay)

async def run_live_inference(websocket, env, agent):
    """Generic inference loop that streams real environment data to the websocket."""
    state = env.reset()
    step = 0
    while True:
        # Allow the agent to take actions without training
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        # Extract metrics
        total_queue = sum(env.lane_queues.values())
        avg_wait = sum(env.lane_waiting_times.values()) / max(1, len(env.lane_waiting_times))
        
        payload = {
            "step": step,
            "reward": round(reward, 2),
            "total_queue": total_queue,
            "avg_wait": round(avg_wait, 1),
            "total_vehicles": env.total_vehicles,
            "accidents": env.total_accidents,
        }
        
        if hasattr(env, 'congestion_multiplier'):
            payload["congestion"] = round(env.congestion_multiplier, 2)
            if hasattr(env, 'incident_jam_multiplier'):
                payload["congestion"] *= env.incident_jam_multiplier
            if hasattr(env, 'arrival_rate'):
                payload["arrival_rate"] = round(env.arrival_rate, 2)
            
        try:
            await websocket.send_json(payload)
        except Exception as e:
            print(f"[LiveInference] Failed to send over websocket: {e}")
            break
            
        await asyncio.sleep(0.1)
        if done:
            print(f"[LiveInference] Environment reported DONE on step {step}. Resetting for continuous streaming.")
            state = env.reset()
            
        step += 1
        
    print(f"[LiveInference] Exited inference loop after {step} steps.")

def get_latest_model(directory: str) -> str:
    """Helper to find the most recently modified .pt file in a directory."""
    if not os.path.exists(directory):
        return None
        
    models = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
    if not models:
        return None
        
    return max(models, key=os.path.getmtime)

async def run_fedavg_simulation(websocket, city, target_pois, use_tomtom):
    client1 = TrafficFLClient("client_1", "sumo_configs2/osm_client1.sumocfg", gui=False, use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois)
    # Attempt to load pre-trained weights dynamically
    model_path = get_latest_model("results_federated")
    if model_path:
        try:
            client1.load_model(model_path)
            print(f"[WS] Loaded FedAvg model from {model_path}")
        except Exception as e:
            print(f"[WS] Error loading model: {e}")
    else:
        print(f"[WS] Warning: No trained model found for FedAvg in results_federated/. Using untrained weights.")
        
    await run_live_inference(websocket, client1.env, client1.agent)

async def run_fedflow_simulation(websocket, city, target_pois, use_tomtom):
    trainer = FedFlowTrainer(num_nodes=4, num_clusters=2, gui=False, use_tomtom=use_tomtom, target_pois=target_pois)
    
    # Grab the first focal node for visualization
    nid = "node_0"
    if nid in trainer.agents:
        agent = trainer.agents[nid]
        env = trainer.envs[nid]
        
        # Try to load latest model
        model_path = get_latest_model("results_fedflow")
        if model_path:
            try:
                agent.load_model(model_path)
                print(f"[WS] Loaded FedFlow model from {model_path}")
            except Exception as e:
                print(f"[WS] Error loading model: {e}")
        else:
            print(f"[WS] Warning: No trained model found for FedFlow in results_fedflow/. Using untrained weights.")
            
        # FedFlow has a different signature for act/get_action needing a graph
        # Since we are modifying server to do live inference, we need to adapt it
        state = env.reset()
        for step in range(200):
            # Same mock graph construction as train_fedflow
            state_graph, adj_node = trainer._get_node_graph_state(nid, state)
            
            action = agent.get_action(state_graph, adj_node)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            total_queue = sum(env.lane_queues.values())
            avg_wait = sum(env.lane_waiting_times.values()) / max(1, len(env.lane_waiting_times))
            
            payload = {
                "step": step,
                "reward": round(reward, 2),
                "total_queue": total_queue,
                "avg_wait": round(avg_wait, 1),
                "total_vehicles": env.total_vehicles,
                "accidents": env.total_accidents,
            }
            if hasattr(env, 'congestion_multiplier'):
                payload["congestion"] = round(env.congestion_multiplier, 2)
                
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)
            if done:
                break
async def run_fedcm_simulation(websocket, city, target_pois, use_tomtom):
    client1 = FedCMClient(
        client_id="client_1", 
        sumo_config_path="sumo_configs2/osm_client1.sumocfg", 
        agent_type="DQN", 
        hidden_dims=[256, 128], 
        use_tomtom=use_tomtom, 
        tomtom_city=city, 
        target_pois=target_pois, 
        gui=False
    )
    
    model_path = get_latest_model("results_fedcm")
    if model_path:
        try:
            client1.load_model(model_path)
            print(f"[WS] Loaded FedCM model from {model_path}")
        except Exception as e:
            print(f"[WS] Error loading model: {e}")
    else:
        print(f"[WS] Warning: No trained model found for FedCM in results_fedcm/. Using untrained weights.")
        
    await run_live_inference(websocket, client1.env, client1.agent)

async def run_fedkd_simulation(websocket, city, target_pois, use_tomtom):
    client1 = TrafficFedKDClient(
        client_id="client_1", 
        sumo_config_path="sumo_configs2/osm_client1.sumocfg", 
        hidden_dims=[256, 128], 
        use_tomtom=use_tomtom, 
        tomtom_city=city, 
        target_pois=target_pois, 
        gui=False
    )

    model_path = get_latest_model("results_fedkd_sumo")
    if model_path:
        try:
            client1.load_model(model_path)
            print(f"[WS] Loaded FedKD model from {model_path}")
        except Exception as e:
            print(f"[WS] Error loading model: {e}")
    else:
        print(f"[WS] Warning: No trained model found for FedKD in results_fedkd_sumo/. Using untrained weights.")
        
    await run_live_inference(websocket, client1.env, client1.agent)
@app.websocket("/api/simulate")
async def websocket_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        
        city = config.get("city", "Delhi")
        algorithm = config.get("algorithm", "Demo (No RL)")
        target_pois = config.get("target_pois", [])
        use_tomtom = config.get("use_tomtom", True)
        
        print(f"[WS] Starting {algorithm} simulation for {city}")

        if algorithm == "FedAvg":
            await run_fedavg_simulation(websocket, city, target_pois, use_tomtom)
        elif algorithm == "FedFlow":
            await run_fedflow_simulation(websocket, city, target_pois, use_tomtom)
        elif algorithm == "FedCM":
            await run_fedcm_simulation(websocket, city, target_pois, use_tomtom)
        elif algorithm == "FedKD":
            await run_fedkd_simulation(websocket, city, target_pois, use_tomtom)
        else:
            await run_demo_simulation(websocket, city, target_pois, use_tomtom)
            
        await websocket.send_json({"status": "complete", "message": "Simulation finished."})
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error during simulation: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
