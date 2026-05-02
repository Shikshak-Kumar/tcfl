"""
AdaptFlow Deployed Evaluation Script
=====================================
Loads all 6 trained AdaptFlow agents (from round-10 models) and evaluates them
in a SINGLE multi-TLS SUMO simulation — exactly the same setup FedDQN / MA2C
train and evaluate in.

WHY this matters
-----------------
During training, each AdaptFlow node trains in its OWN single-TLS simulation
(isolation mode). This creates an unfair comparison because:
  - Uncontrolled neighbouring TLS cause spillback → inflated queue/wait metrics
  - Different nodes see different time windows (begin=0..1500) → unequal traffic load

In DEPLOYMENT (the correct evaluation), all 6 trained agents are placed in one
simulation simultaneously → network benefits from coordinated control → metrics
are directly comparable with FedDQN and MA2C.

Output
------
  <results_dir>/deployed_eval.json
  Keys: avg_wait, avg_queue, tp_ratio, episodes, n_tls, n_lanes
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

# ── path bootstrap ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # adaptflow_ac/
_REPO = os.path.dirname(_ROOT)   # tcfl/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agents.adaptflow import AdaptFlowAgent
from utils.sumo_scenario import get_sumo_config_paths

_MAX_QUEUE = 40
_N_PHASES  = 4


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─── per-TLS state in AdaptFlow format: (1, 6) ─────────────────────────────────

def _get_tls_state(traci, tls_id: str, tls_lanes: List[str], num_phases: int) -> np.ndarray:
    """Extract (1, 6) state [q_NS, q_EW, phase_oh × 4] for a single TLS."""
    n_lanes = len(tls_lanes)
    half = max(1, n_lanes // 2)
    ns_q = ew_q = 0.0
    for i, lane in enumerate(tls_lanes):
        try:
            q = traci.lane.getLastStepHaltingNumber(lane)
        except Exception:
            q = 0
        if i < half:
            ns_q += q
        else:
            ew_q += q
    q_ns = min(ns_q / (_MAX_QUEUE * half), 1.0)
    q_ew = min(ew_q / (_MAX_QUEUE * max(1, n_lanes - half)), 1.0)

    try:
        phase = traci.trafficlight.getPhase(tls_id)
        phase = min(phase, num_phases - 1)
    except Exception:
        phase = 0

    oh = np.zeros(4, dtype=np.float32)
    oh[min(phase, 3)] = 1.0
    return np.array([q_ns, q_ew, *oh], dtype=np.float32).reshape(1, 6)


def _get_tls_queue_wait(traci, tls_lanes: List[str]):
    """Return (total_queue_halting, total_lane_wait) for this TLS."""
    total_q = 0
    total_w = 0.0
    for lane in tls_lanes:
        try:
            total_q += traci.lane.getLastStepHaltingNumber(lane)
            total_w += traci.lane.getWaitingTime(lane)
        except Exception:
            pass
    return total_q, total_w


# ─── single deployment episode ─────────────────────────────────────────────────

def run_episode(
    config_path: str,
    agents: Dict[str, AdaptFlowAgent],
    max_steps: int,
    gui: bool,
) -> Dict:
    """
    Run ONE episode in the multi-TLS SUMO sim with all agents deployed.
    Returns episode-level aggregate metrics.
    """
    port = _find_free_port()
    binary = "sumo-gui" if gui else "sumo"
    cmd = [
        binary,
        "-c", config_path,
        "--remote-port", str(port),
        "--start",
        "--quit-on-end",
        "--no-warnings",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.5)

    try:
        import traci
        traci.init(port)

        # Discover all TLS in the simulation (up to 6)
        all_tls = list(traci.trafficlight.getIDList())
        n_tls = min(len(all_tls), 6)
        tls_ids = all_tls[:n_tls]

        # Per-TLS lane lists and phase counts
        tls_lanes: Dict[str, List[str]] = {}
        tls_nphases: Dict[str, int] = {}
        total_lanes = 0
        for tid in tls_ids:
            lanes = list(traci.trafficlight.getControlledLanes(tid))
            tls_lanes[tid] = lanes
            total_lanes += len(lanes)
            try:
                defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tid)
                nphases = len(defs[0].phases) if defs and defs[0].phases else _N_PHASES
            except Exception:
                nphases = _N_PHASES
            tls_nphases[tid] = nphases

        # Map each TLS to its agent (node i → agent "node_i")
        # If we have fewer agents than TLS, wrap around; if fewer TLS, use first N agents
        agent_keys = sorted(agents.keys())  # ["node_0", "node_1", ...]
        tls_agent_map: Dict[str, AdaptFlowAgent] = {}
        for i, tid in enumerate(tls_ids):
            key = agent_keys[i % len(agent_keys)]
            tls_agent_map[tid] = agents[key]

        # Reset per-agent state histories so LSTM starts fresh
        for ag in agents.values():
            ag.state_history.clear()

        # Metric accumulators
        # We use lane-level metrics (consistent with FedDQN/MA2C stored format):
        #   avg_queue = total halting vehicles / total lanes / steps
        #   avg_wait  = total lane waiting time / total lanes / steps
        queue_total = 0.0   # sum over all steps of (halting vehicles across all TLS lanes)
        wait_total  = 0.0   # sum over all steps of (lane waiting time across all TLS lanes)
        n_lanes_total = max(1, total_lanes)
        n_steps_done  = 0
        arrived_total = 0
        done = False
        step = 0

        while not done and step < max_steps:
            for tid in tls_ids:
                # Build single-node state: (1, 6)
                state = _get_tls_state(traci, tid, tls_lanes[tid], tls_nphases[tid])
                adj = np.array([[1.0]], dtype=np.float32)  # self-loop only
                actions = tls_agent_map[tid].get_action(state, adj)
                action = int(actions[0]) % tls_nphases[tid]
                try:
                    traci.trafficlight.setPhase(tid, action)
                except Exception:
                    pass

            traci.simulationStep()
            step += 1
            n_steps_done += 1

            # Collect per-step global metrics across all controlled TLS
            step_queue = 0
            step_wait  = 0.0
            for tid in tls_ids:
                tq, tw = _get_tls_queue_wait(traci, tls_lanes[tid])
                step_queue += tq
                step_wait  += tw
            queue_total += step_queue
            wait_total  += step_wait

            # Count arrived vehicles
            try:
                arrived_ids = traci.simulation.getArrivedIDList()
                arrived_total += len(arrived_ids)
            except Exception:
                pass

            # Episode ends when simulation has no more vehicles
            try:
                remaining = traci.simulation.getMinExpectedNumber()
                done = remaining == 0
            except Exception:
                done = False

        # Per-lane per-step metrics (directly comparable with FedDQN/MA2C raw data / n_lanes)
        denom = max(1, n_lanes_total * n_steps_done)
        avg_queue = queue_total / denom
        avg_wait  = wait_total  / denom
        tp_ratio  = arrived_total / max(1, arrived_total + queue_total / max(1, n_steps_done))

        return {
            "avg_wait": float(avg_wait),
            "avg_queue": float(avg_queue),
            "tp_ratio": float(tp_ratio),
            "arrived": int(arrived_total),
            "steps": int(step),
            "n_tls": int(n_tls),
            "n_lanes": int(n_lanes_total),
        }

    except Exception as e:
        print(f"  [ERROR] Episode failed: {e}")
        return {"avg_wait": 0.0, "avg_queue": 0.0, "tp_ratio": 0.0,
                "arrived": 0, "steps": 0, "n_tls": 0, "n_lanes": 0}
    finally:
        try:
            traci.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate AdaptFlow in deployed multi-TLS mode")
    ap.add_argument("--results-dir", required=True,
                    help="Path to AdaptFlow results dir (contains node_*_round_*_model.pt)")
    ap.add_argument("--sumo-scenario", default="dwarka_mor",
                    help="SUMO scenario key (dwarka_mor, china_osm, …)")
    ap.add_argument("--round", type=int, default=0,
                    help="Round number to load models from (0 = use highest available)")
    ap.add_argument("--episodes", type=int, default=3,
                    help="Number of evaluation episodes")
    ap.add_argument("--steps", type=int, default=500,
                    help="Max steps per episode")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    # ── locate model files ────────────────────────────────────────────────────
    import glob as _glob
    if args.round == 0:
        # Auto-detect highest round
        pts = _glob.glob(os.path.join(results_dir, "node_0_round_*_model.pt"))
        if not pts:
            print(f"[ERROR] No model files found in {results_dir}")
            sys.exit(1)
        rounds_available = sorted(
            [int(p.split("_round_")[1].split("_")[0]) for p in pts]
        )
        chosen_round = rounds_available[-1]
    else:
        chosen_round = args.round
    print(f"\n{'='*65}")
    print(f"  AdaptFlow Deployed Evaluation")
    print(f"  Loading round-{chosen_round} models from: {results_dir}")
    print(f"  Scenario: {args.sumo_scenario}  |  Episodes: {args.episodes}  |  Steps: {args.steps}")
    print(f"{'='*65}\n")

    # ── load agents ───────────────────────────────────────────────────────────
    agents: Dict[str, AdaptFlowAgent] = {}
    for node_idx in range(6):
        model_path = os.path.join(
            results_dir, f"node_{node_idx}_round_{chosen_round}_model.pt"
        )
        if not os.path.exists(model_path):
            print(f"  [WARN] Missing model: {model_path} — skipping node {node_idx}")
            continue
        ag = AdaptFlowAgent(node_id=node_idx, state_dim=6, action_dim=4)
        # Support both old and new PyTorch (weights_only added in 1.13+)
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location="cpu")
        ag.model.load_state_dict(state_dict)
        ag.model.eval()
        agents[f"node_{node_idx}"] = ag
        print(f"  Loaded model: node_{node_idx}_round_{chosen_round}_model.pt")

    if not agents:
        print("[ERROR] No agents loaded. Aborting.")
        sys.exit(1)

    # ── resolve SUMO config ───────────────────────────────────────────────────
    paths = get_sumo_config_paths(args.sumo_scenario)
    if not paths:
        print(f"[ERROR] No SUMO configs found for scenario '{args.sumo_scenario}'")
        sys.exit(1)
    config_path = paths[0]   # node0 config — same as FedDQN/MA2C use
    print(f"\n  SUMO config: {os.path.basename(config_path)}")
    print(f"  (Using node0 config = same traffic window as FedDQN/MA2C baseline)\n")

    # ── run evaluation episodes ───────────────────────────────────────────────
    ep_results = []
    for ep in range(args.episodes):
        print(f"  Episode {ep + 1}/{args.episodes} ...", end=" ", flush=True)
        result = run_episode(config_path, agents, args.steps, args.gui)
        ep_results.append(result)
        print(
            f"Wait={result['avg_wait']:.2f}s  Queue={result['avg_queue']:.4f}  "
            f"TP={result['tp_ratio']:.4f}  Arrived={result['arrived']}"
        )

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg = {
        "avg_wait":  float(np.mean([r["avg_wait"]  for r in ep_results])),
        "avg_queue": float(np.mean([r["avg_queue"] for r in ep_results])),
        "tp_ratio":  float(np.mean([r["tp_ratio"]  for r in ep_results])),
        "episodes":  len(ep_results),
        "round":     chosen_round,
        "n_tls":     ep_results[0]["n_tls"] if ep_results else 0,
        "n_lanes":   ep_results[0]["n_lanes"] if ep_results else 0,
        "per_episode": ep_results,
    }

    out_path = os.path.join(results_dir, "deployed_eval.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DEPLOYED EVALUATION COMPLETE")
    print(f"  Avg Waiting Time : {agg['avg_wait']:.2f} s/vehicle")
    print(f"  Avg Queue Length : {agg['avg_queue']:.4f} veh/lane (episode mean)")
    print(f"  Throughput Ratio : {agg['tp_ratio']:.4f}")
    print(f"  Saved to: {out_path}")
    print(f"{'='*65}\n")
    print("  >> Now re-run: python compare_dwarka_mor.py  (or compare_china_osm.py)")
    print("     The comparison will auto-detect and use deployed_eval.json\n")


if __name__ == "__main__":
    main()
