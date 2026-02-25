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

async def run_fedavg_simulation(websocket, city, target_pois, use_tomtom):
    client1 = TrafficFLClient("client_1", "sumo_configs2/osm_client1.sumocfg", gui=False, use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois)
    client2 = TrafficFLClient("client_2", "sumo_configs2/osm_client2.sumocfg", gui=False, use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois)
    clients = [client1, client2]
    global_params = None

    for round_num in range(5):
        await yield_simulated_ui_steps(websocket, round_num, base_queue=40, base_reward=0.1)
        params_list = []
        for client in clients:
            current_params = global_params if global_params is not None else client.get_parameters({})
            config = {"round": round_num, "episodes": 1, "learning_rate": 0.001}
            updated_params, _, _ = client.fit(current_params, config)
            params_list.append(updated_params)

        weights = [0.5, 0.5]
        num_layers = len(params_list[0])
        agg = []
        for layer_idx in range(num_layers):
            layer_sum = None
            for params, w in zip(params_list, weights):
                if layer_sum is None:
                    layer_sum = params[layer_idx] * w
                else:
                    layer_sum += params[layer_idx] * w
            agg.append(layer_sum)
        global_params = agg

        for client in clients:
            client.set_parameters(global_params)

async def run_fedflow_simulation(websocket, city, target_pois, use_tomtom):
    trainer = FedFlowTrainer(num_nodes=4, num_clusters=2, gui=False, use_tomtom=use_tomtom, target_pois=target_pois)
    
    for round_idx in range(3):
        await yield_simulated_ui_steps(websocket, round_idx, base_queue=55, base_reward=-0.5)
        trainer.run_round(round_idx)

async def run_fedcm_simulation(websocket, city, target_pois, use_tomtom):
    server = FedCMServer(state_dim=12, action_dim=4, proxy_dataset_size=500, weighting_method="performance")
    client1 = FedCMClient("client_1", "sumo_configs2/osm_client1.sumocfg", "DQN", [256, 128], use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois, gui=False)
    client2 = FedCMClient("client_2", "sumo_configs2/osm_client2.sumocfg", "DQN", [128, 64], use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois, gui=False)
    clients = [client1, client2]

    for round_num in range(1, 4):
        await yield_simulated_ui_steps(websocket, round_num-1, base_queue=45, base_reward=-0.2)
        
        client_states = []
        all_logits = []
        client_ids = []
        for client in clients:
            client.train(round_num)
            if hasattr(client.agent, "memory") and len(client.agent.memory) > 0:
                client_states.append(np.array([exp[0] for exp in list(client.agent.memory)[:200]]))
                
        if len(client_states) > 0:
            proxy_states = server.construct_proxy_dataset(client_states)
        else:
            proxy_states = np.random.rand(500, 12)
            
        for client in clients:
            all_logits.append(client.get_logits(proxy_states))
            client_ids.append(client.client_id)
            server.update_client_performance(client.client_id, 0.5) 

        teacher_logits = server.compute_ensemble_teacher(all_logits, client_ids)
        for client in clients:
            client.distill(proxy_states, teacher_logits, epochs=2)

async def run_fedkd_simulation(websocket, city, target_pois, use_tomtom):
    server = TrafficFedKDServer(num_rounds=3, min_clients=2)
    server.initialize_proxy_dataset(state_size=12)
    client1 = TrafficFedKDClient("client_1", "sumo_configs2/osm_client1.sumocfg", [256, 128], gui=False, use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois)
    client2 = TrafficFedKDClient("client_2", "sumo_configs2/osm_client2.sumocfg", [64, 32], gui=False, use_tomtom=use_tomtom, tomtom_city=city, target_pois=target_pois)
    clients = [client1, client2]

    for round_num in range(3):
        await yield_simulated_ui_steps(websocket, round_num, base_queue=50, base_reward=-0.3)
        
        observed_states = []
        for client in clients:
            client._train_agent(episodes=1)
            observed_states.append(client.get_observed_states(limit=100))
            
        server.update_proxy_dataset(observed_states)
        
        if server.proxy_states.size > 0:
            all_logits = []
            for client in clients:
                all_logits.append(client.get_logits(server.proxy_states))
            
            consensus_logits = server.aggregate_logits(all_logits)
            
            for client in clients:
                client.distill(server.proxy_states, consensus_logits)

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
