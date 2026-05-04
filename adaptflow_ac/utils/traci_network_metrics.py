"""
Network-level traffic metrics via TraCI (shared by Fed-DQN, MA2C, logging).

Definitions match research logs under results/dwarka_mor:
  - Controlled lanes exclude SUMO **internal** junction lanes (ids starting with ':'), which are
    usually empty; including them dilutes queue/wait toward zero (unrealistic summaries).
  - avg_queue (average_queue_length): mean halting vehicles per approach lane (often 0.0–0.5).
  - queue_total_halting: sum of halting vehicles over all approach lanes (interpretable “cars stopped”).
  - queue_max_halting: max halting count on any one approach lane (peak queue).
  - avg_wait: mean over approach lanes of (lane waiting time / max(1, vehicles on lane)) [s],
    so moving traffic still contributes a readable delay proxy vs summing empty internal lanes.
  - tp_ratio (throughput_ratio): cumulative_arrived / max(1, cumulative_departed) over the episode.
    Often ~0.5–0.55 when inflow/outflow are loosely balanced; weak signal for comparing policies.
  - arrival_rate: cumulative_arrived / episode_sim_steps — sensitive to how quickly trips finish.
  - Joint multi-TLS logs (Fed DQN, MA2C) use **network_approach_lane_means**: for each TLS compute
    approach_lane_snapshot, then average those stats over intersections (same spirit as averaging
    AdaptFlow nodes). Do not pool all lanes from all TLS into one mean.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import traci
except ImportError:  # pragma: no cover
    traci = None  # type: ignore


def controlled_lanes_unique(tl_ids: Sequence[str]) -> List[str]:
    if traci is None or not traci.isLoaded():
        return []
    seen = set()
    out: List[str] = []
    for tl in tl_ids:
        for ln in traci.trafficlight.getControlledLanes(tl):
            if ln not in seen:
                seen.add(ln)
                out.append(ln)
    # Internal junction lanes (':...') are typically empty; they skew means toward 0.
    ext = [ln for ln in out if not ln.startswith(":")]
    return ext if ext else out


def approach_lane_snapshot(tl_ids: Sequence[str]) -> Tuple[float, float, float, float]:
    """(mean_halting_per_lane, total_halting_veh, max_halting_on_lane, avg_wait_proxy_s).

    tl_ids: a **single** TLS id in a one-element list, e.g. ``[tl_id]``.
    """
    lanes = controlled_lanes_unique(tl_ids)
    if not lanes:
        return 0.0, 0.0, 0.0, 0.0
    qs = [float(traci.lane.getLastStepHaltingNumber(ln)) for ln in lanes]
    delays: List[float] = []
    for ln in lanes:
        n = int(traci.lane.getLastStepVehicleNumber(ln))
        w = float(traci.lane.getWaitingTime(ln))
        delays.append(w / float(max(1, n)))
    return (
        float(np.mean(qs)),
        float(np.sum(qs)),
        float(np.max(qs)),
        float(np.mean(delays)),
    )


def network_approach_lane_means(
    tl_ids: Sequence[str],
) -> Tuple[float, float, float, float]:
    """Mean over intersections of each TLS's approach_lane_snapshot (fair multi-agent metric).

    Pooling all lanes from all TLS into one average **underestimates** congestion compared to
    reporting per-node means (AdaptFlow-style). Use this for joint-sumocfg baselines.
    """
    ids = [str(t) for t in tl_ids]
    if not ids:
        return 0.0, 0.0, 0.0, 0.0
    rows = [approach_lane_snapshot([tl]) for tl in ids]
    return (
        float(np.mean([r[0] for r in rows])),
        float(np.mean([r[1] for r in rows])),
        float(np.mean([r[2] for r in rows])),
        float(np.mean([r[3] for r in rows])),
    )


def snapshot_lane_averages(tl_ids: Sequence[str]) -> Tuple[float, float]:
    """(mean halting per lane, wait proxy) aggregated consistently over all ``tl_ids``."""
    m, _t, _x, w = network_approach_lane_means(tl_ids)
    return m, w


def fed_dqn_step_reward(tl_id: str, sigma: float = 0.1) -> float:
    """Per-intervention reward aligned with results README: -(wait/10 + sigma*queue)."""
    lanes = controlled_lanes_unique([tl_id])
    if not lanes:
        return 0.0
    qs = [float(traci.lane.getLastStepHaltingNumber(ln)) for ln in lanes]
    delays = [
        float(traci.lane.getWaitingTime(ln))
        / float(max(1, int(traci.lane.getLastStepVehicleNumber(ln))))
        for ln in lanes
    ]
    aq = float(np.mean(qs))
    aw = float(np.mean(delays))
    return -(aw / 10.0 + sigma * aq)


class EpisodeThroughputTally:
    """Cumulative arrived/departed counts over an episode (call step() each sim step)."""

    def __init__(self) -> None:
        self.arrived = 0
        self.departed = 0

    def step(self) -> None:
        if traci is None or not traci.isLoaded():
            return
        self.arrived += int(traci.simulation.getArrivedNumber())
        self.departed += int(traci.simulation.getDepartedNumber())

    def ratio(self) -> float:
        return float(self.arrived) / max(1.0, float(self.departed))


def episode_arrival_rate(tally: EpisodeThroughputTally, n_sim_steps: int) -> float:
    """Mean vehicles that reached their destination per simulation second (1 step ≈ 1 s)."""
    return float(tally.arrived) / max(1, int(n_sim_steps))


def num_congested_lanes(tl_ids: Sequence[str], halting_threshold: int = 1) -> int:
    lanes = controlled_lanes_unique(tl_ids)
    if not lanes:
        return 0
    return int(
        sum(1 for ln in lanes if traci.lane.getLastStepHaltingNumber(ln) >= halting_threshold)
    )


def max_queue_over_lanes(tl_ids: Sequence[str]) -> float:
    lanes = controlled_lanes_unique(tl_ids)
    if not lanes:
        return 0.0
    return float(max(traci.lane.getLastStepHaltingNumber(ln) for ln in lanes))
