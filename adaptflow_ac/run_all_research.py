#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ONE-SHOT RESEARCH RUN — sumo_configs2 (Dwarka Mor) + joint osm.sumocfg
================================================================================

Run this file only. It will:

  1. Preflight: SUMO on PATH, backend configs present (joint + 6 node cfgs).
  2. Train all four methods with identical rounds / nodes / steps / seed.
  3. Save everything under one timestamped folder (+ optional fixed --results-dir).
  4. **Automatically** generate training plots (same run — no extra command).
     Output: <results>/plots_training/  (unless you pass --no-plot).
  5. Write RESEARCH_COMPLETE.md and research_summary.json (final-round metrics).

Usage (from any directory):

  python F:\\ttraffic\\tcfl\\adaptflow_ac\\run_all_research.py

  python run_all_research.py --rounds 5 --steps 300

  python run_all_research.py --results-dir F:\\out\\my_run --no-plot   # train only

  python run_all_research.py --plot-only --results-dir F:\\out\\my_run

Requires: SUMO (sumo on PATH), Python deps from requirements.txt, tcfl/backend/.
================================================================================
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Bootstrap: script directory = adaptflow_ac root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.research_paths import sumo_configs2_joint_path
from utils.sumo_scenario import get_sumo_config_paths


def _run(cmd: List[str], cwd: str) -> None:
    print("\n" + "=" * 72)
    print(">>> " + " ".join(cmd))
    print("=" * 72)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(
            f"\n[ERROR] Command failed with exit code {r.returncode}. Stopping.\n"
        )


def _which_sumo() -> Optional[str]:
    return shutil.which("sumo")


def preflight(joint_sumocfg: str, nodes_expected: int) -> None:
    exe = _which_sumo()
    if not exe:
        print(
            "\n[ERROR] SUMO executable 'sumo' not found on PATH.\n"
            "Install SUMO and add its bin folder to PATH, then retry.\n"
        )
        raise SystemExit(2)
    print(f"[OK] SUMO: {exe}")

    if not os.path.isfile(joint_sumocfg):
        print(f"\n[ERROR] Missing joint config:\n  {joint_sumocfg}\n")
        raise SystemExit(2)
    print(f"[OK] Joint sumocfg: {joint_sumocfg}")

    sc_dir = os.path.dirname(os.path.abspath(joint_sumocfg))
    net_xml = os.path.join(sc_dir, "osm.net.xml")
    net_gz = os.path.join(sc_dir, "osm.net.xml.gz")
    if not os.path.isfile(net_xml) and not os.path.isfile(net_gz):
        print(
            f"\n[ERROR] Network file missing: need osm.net.xml or osm.net.xml.gz in:\n"
            f"  {sc_dir}\n"
        )
        raise SystemExit(2)
    print(f"[OK] Network: {net_xml if os.path.isfile(net_xml) else net_gz}")

    trips = os.path.join(sc_dir, "osm.passenger.trips.xml")
    if not os.path.isfile(trips):
        print(
            f"\n[ERROR] Route file missing (referenced by sumocfg):\n  {trips}\n"
        )
        raise SystemExit(2)
    print(f"[OK] Trips: {trips}")

    node_cfgs = get_sumo_config_paths("dwarka_mor")
    missing = [p for p in node_cfgs if not os.path.isfile(p)]
    if missing:
        print("\n[ERROR] Missing AdaptFlow node config(s):")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(2)
    if len(node_cfgs) < nodes_expected:
        print(
            f"\n[WARN] dwarka_mor lists {len(node_cfgs)} node configs; "
            f"you asked for --nodes {nodes_expected}. Training will cap to available TLS."
        )
    print(f"[OK] AdaptFlow node configs: {len(node_cfgs)} file(s) under dwarka_mor preset.")


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def aggregate_summary(base: str) -> Dict[str, Any]:
    """Load last-round / last-episode metrics from each algorithm folder."""
    summary: Dict[str, Any] = {"results_dir": os.path.abspath(base)}

    af = os.path.join(base, "adaptflow", "adaptflow_all_rounds.json")
    if os.path.isfile(af):
        try:
            with open(af, encoding="utf-8") as f:
                rounds = json.load(f)
            if isinstance(rounds, list) and rounds:
                r = rounds[-1]
                nodes = r.get("nodes") or {}
                summary["adaptflow"] = {
                    "round": r.get("round"),
                    "mean_total_reward": _mean(
                        [float(v.get("total_reward", 0)) for v in nodes.values()]
                    ),
                    "mean_queue": _mean(
                        [
                            float(
                                v.get("metrics", {}).get(
                                    "queue_total_halting",
                                    v.get("metrics", {}).get("average_queue_length", 0),
                                )
                            )
                            for v in nodes.values()
                        ]
                    ),
                    "mean_wait_s": _mean(
                        [float(v.get("avg_waiting_time", 0)) for v in nodes.values()]
                    ),
                    "mean_tp_ratio": _mean(
                        [
                            float(v.get("metrics", {}).get("throughput_ratio", 0))
                            for v in nodes.values()
                        ]
                    ),
                    "mean_arrival_rate": _mean(
                        [
                            float(
                                v.get("metrics", {}).get(
                                    "arrival_rate",
                                    v.get("metrics", {}).get("throughput_ratio", 0),
                                )
                            )
                            for v in nodes.values()
                        ]
                    ),
                    "mean_loss": _mean([float(v.get("loss", 0)) for v in nodes.values()]),
                }
        except Exception as e:
            summary["adaptflow"] = {"error": str(e)}

    fd_path = os.path.join(base, "fed_dqn_tsc", "fed_dqn_tsc_all_rounds.json")
    if os.path.isfile(fd_path):
        try:
            with open(fd_path, encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list) and rows:
                row = rows[-1]
                summary["fed_dqn_tsc"] = {
                    "round": row.get("round"),
                    "avg_reward": float(row.get("avg_reward", 0)),
                    "avg_loss": float(row.get("avg_loss", 0)),
                    "avg_queue": float(
                        row.get("avg_queue_total", row.get("avg_queue", 0))
                    ),
                    "avg_wait_s": float(row.get("avg_wait", 0)),
                    "avg_tp_ratio": float(row.get("avg_tp_ratio", 0)),
                    "avg_arrival_rate": float(
                        row.get("avg_arrival_rate", row.get("avg_tp_ratio", 0))
                    ),
                }
        except Exception as e:
            summary["fed_dqn_tsc"] = {"error": str(e)}

    ma_path = os.path.join(base, "multi_agent_ac", "multi_agent_ac_by_round.json")
    if os.path.isfile(ma_path):
        try:
            with open(ma_path, encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list) and rows:
                row = rows[-1]
                summary["multi_agent_ac"] = {
                    "episode": row.get("round"),
                    "avg_reward": float(row.get("avg_reward", 0)),
                    "avg_loss": float(row.get("avg_loss", 0)),
                    "avg_queue": float(
                        row.get("avg_queue_total", row.get("avg_queue", 0))
                    ),
                    "avg_wait_s": float(row.get("avg_wait", 0)),
                    "tp_ratio": float(row.get("tp_ratio", 0)),
                    "arrival_rate": float(
                        row.get("arrival_rate", row.get("tp_ratio", 0))
                    ),
                }
        except Exception as e:
            summary["multi_agent_ac"] = {"error": str(e)}

    dq_path = os.path.join(base, "dqtsca", "dqtsca_training.json")
    if os.path.isfile(dq_path):
        try:
            with open(dq_path, encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list) and rows:
                row = rows[-1]
                summary["dqtsca"] = {
                    "episode": row.get("episode"),
                    "total_reward": float(row.get("total_reward", 0)),
                    "avg_loss": float(row.get("avg_loss", 0)),
                    "avg_queue": float(
                        row.get("avg_queue_total", row.get("avg_queue", 0))
                    ),
                    "avg_wait_s": float(row.get("avg_wait", 0)),
                    "tp_ratio": float(row.get("tp_ratio", 0)),
                    "arrival_rate": float(
                        row.get("arrival_rate", row.get("tp_ratio", 0))
                    ),
                }
        except Exception as e:
            summary["dqtsca"] = {"error": str(e)}

    return summary


def _glob_one(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def write_completion_report(
    base: str, manifest: Dict[str, Any], summary: Dict[str, Any]
) -> None:
    """Markdown report for the paper / lab book."""
    lines = [
        "# Research run complete",
        "",
        f"- **Output directory:** `{os.path.abspath(base)}`",
        f"- **UTC timestamp (manifest):** {manifest.get('created_utc', '')}",
        "",
        "## Hyperparameters",
        "",
        "```json",
        json.dumps(manifest.get("hyperparameters", {}), indent=2),
        "```",
        "",
        "## Files generated",
        "",
        "| Artifact | Path |",
        "|----------|------|",
        f"| Manifest | `{os.path.join(base, 'training_manifest.json')}` |",
        f"| Training curves (combined) | `{os.path.join(base, 'plots_training', 'training_curves_combined.png')}` |",
        f"| Per-metric lines | `{os.path.join(base, 'plots_training', 'line_*.png')}` |",
        f"| Aggregated metrics | `{os.path.join(base, 'research_summary.json')}` |",
        f"| AdaptFlow rounds JSON | `{os.path.join(base, 'adaptflow', 'adaptflow_all_rounds.json')}` |",
        f"| FedDQN rounds JSON | `{os.path.join(base, 'fed_dqn_tsc', 'fed_dqn_tsc_all_rounds.json')}` |",
        f"| MA2C by-round JSON | `{os.path.join(base, 'multi_agent_ac', 'multi_agent_ac_by_round.json')}` |",
        f"| DQTSCA training JSON | `{os.path.join(base, 'dqtsca', 'dqtsca_training.json')}` |",
        "",
        "## Final-round snapshot (training logs)",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Notes (methods / demand — conference write-up)",
        "",
        "- **Demand parity:** All Dwarka Mor configs use **begin=0, end=3600** and the same "
        "`osm.passenger.trips.xml`, matching **`osm.sumocfg`**, so **vehicle insertion is aligned** "
        "across AdaptFlow federated nodes and FedDQN / MA2C / DQTSCA.",
        "- **AdaptFlow** still trains **six separate SUMO processes** (one TLS focal per config); "
        "heterogeneity is **TLS/processing** (e.g. jam-threshold), not staggered demand windows.",
        "- **Baselines** use **joint** `osm.sumocfg` (one process, all TLS interact).",
        "- **DQTSCA** acts on the **first** TLS ID in the joint net; cite as single-intersection control.",
        "- **Throughput plots** use **arrival rate** (completed trips per simulation second) when logged; "
        "the legacy ratio arrived÷departed is often ~0.5–0.55 and is a weak control signal.",
        "- **Cross-method throughput:** AdaptFlow uses **per-node** SUMO scenarios; baselines use the **joint** net. "
        "Absolute arrival rates are not always comparable—use **queue / delay** for primary cross-method ranking, "
        "and arrival curves mainly for **within-method** learning trends.",
        "- **TP ratio (trip completion):** `arrived ÷ departed` stays low if **episodes are short** (many vehicles still "
        "driving). Default **`--steps 2500`** is the same for all four algorithms (fair horizon). Use `--steps 3600` for a "
        "full-hour sumocfg. AdaptFlow may use `--adaptflow-tp-reward` and `--adaptflow-completion-bonus` (see manifest); "
        "set both to **0** for identical reward definition vs baselines.",
        "",
    ]
    path = os.path.join(base, "RESEARCH_COMPLETE.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Wrote {path}")

    # Quick pointer for checkpoint paths
    ckpt_lines = ["# Model checkpoints (for deployment / eval)", ""]
    g = _glob_one(os.path.join(base, "adaptflow", "adaptflow_global_*.pt"))
    if g:
        ckpt_lines.append(f"- AdaptFlow global: `{g}`")
    fd = os.path.join(base, "fed_dqn_tsc", "agent_0.pt")
    if os.path.isfile(fd):
        ckpt_lines.append(f"- FedDQN agent_0: `{fd}`")
    ma = os.path.join(base, "multi_agent_ac", "agent_0.pt")
    if os.path.isfile(ma):
        ckpt_lines.append(f"- MA2C agent_0: `{ma}`")
    dq = os.path.join(base, "dqtsca", "model.pt")
    if os.path.isfile(dq):
        ckpt_lines.append(f"- DQTSCA: `{dq}`")
    ckpt_path = os.path.join(base, "CHECKPOINTS.md")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ckpt_lines) + "\n")
    print(f"[OK] Wrote {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot: train 4 algorithms + plots + summary (sumo_configs2 research)."
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument(
        "--steps",
        type=int,
        default=2500,
        help="Steps per episode (sim seconds), identical for AdaptFlow + all baselines. 2500 balances runtime vs trip completion; "
        "use 3600 for a full-hour horizon.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output folder (default: results_research_sumo_configs2_<UTC>)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib (default: plots ARE generated after training in this same script)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only aggregate + plot; requires existing --results-dir with training JSONs",
    )
    parser.add_argument("--skip-adaptflow", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument(
        "--adaptflow-tp-reward",
        type=float,
        default=0.08,
        help="AdaptFlow only: per-step bonus = scale × newly arrived vehicles (0 disables).",
    )
    parser.add_argument(
        "--adaptflow-completion-bonus",
        type=float,
        default=15.0,
        help="AdaptFlow only: on last sim step, add scale × trip completion ratio arrived/max(1,departed). "
        "0 disables. Baselines do not use this (disclose if comparing fairly).",
    )
    args = parser.parse_args()

    py = sys.executable
    joint = os.path.abspath(sumo_configs2_joint_path())

    if args.plot_only:
        if not args.results_dir or not os.path.isdir(args.results_dir):
            print("[ERROR] --plot-only requires an existing --results-dir")
            raise SystemExit(2)
        base = os.path.abspath(args.results_dir)
        man_path = os.path.join(base, "training_manifest.json")
        manifest: Dict[str, Any] = {}
        if os.path.isfile(man_path):
            with open(man_path, encoding="utf-8") as f:
                manifest = json.load(f)
        if not args.no_plot:
            _run(
                [
                    py,
                    os.path.join(_HERE, "plot_research_training_curves.py"),
                    "--results-dir",
                    base,
                ],
                cwd=_HERE,
            )
        summary = aggregate_summary(base)
        with open(os.path.join(base, "research_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_completion_report(base, manifest, summary)
        print(f"\n[DONE] Plot-only / summary refresh: {base}\n")
        return

    preflight(joint, args.nodes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = args.results_dir or os.path.join(
        _HERE, f"results_research_sumo_configs2_{stamp}"
    )
    os.makedirs(base, exist_ok=True)

    manifest = {
        "created_utc": stamp,
        "map": "sumo_configs2 — AdaptFlow: dwarka_mor node configs; baselines: osm.sumocfg",
        "joint_sumocfg": joint,
        "adaptflow_sumo_scenario": "dwarka_mor",
        "demand_parity_paper_standard": {
            "trips_file": "osm.passenger.trips.xml (same for all node cfgs and osm.sumocfg)",
            "simulation_horizon_s": [0, 3600],
            "note": "All osm_node*.sumocfg use begin=0, end=3600 like osm.sumocfg so vehicle "
            "insertion over time matches across AdaptFlow nodes and baseline algorithms. "
            "Cross-node variation is infrastructure/TLS (jam-threshold, teleport, rerouting), not time-shifted demand.",
        },
        "hyperparameters": {
            "rounds": args.rounds,
            "nodes": args.nodes,
            "steps_per_episode": args.steps,
            "seed": args.seed,
            "fed_dqn_episodes_per_round": 1,
            "fed_dqn_fine_tune": False,
            "ma2c_batch_size": 100,
            "dqtsca_total_train_steps": 100000,
            "adaptflow_tp_reward_scale": args.adaptflow_tp_reward,
            "adaptflow_completion_bonus_scale": args.adaptflow_completion_bonus,
        },
        "subdirs": {
            "adaptflow": os.path.join(base, "adaptflow"),
            "fed_dqn_tsc": os.path.join(base, "fed_dqn_tsc"),
            "multi_agent_ac": os.path.join(base, "multi_agent_ac"),
            "dqtsca": os.path.join(base, "dqtsca"),
        },
    }
    with open(os.path.join(base, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not args.skip_adaptflow:
        _run(
            [
                py,
                os.path.join(_HERE, "train", "train_adaptflow.py"),
                "--rounds",
                str(args.rounds),
                "--nodes",
                str(args.nodes),
                "--steps",
                str(args.steps),
                "--sumo-scenario",
                "dwarka_mor",
                "--sumo-headless",
                "--results-dir",
                manifest["subdirs"]["adaptflow"],
                "--tp-reward-scale",
                str(args.adaptflow_tp_reward),
                "--completion-bonus-scale",
                str(args.adaptflow_completion_bonus),
            ],
            cwd=_HERE,
        )

    if not args.skip_baselines:
        fd = manifest["subdirs"]["fed_dqn_tsc"]
        _run(
            [
                py,
                os.path.join(_HERE, "train", "train_fed_dqn_tsc.py"),
                "--sumocfg",
                joint,
                "--rounds",
                str(args.rounds),
                "--nodes",
                str(args.nodes),
                "--steps",
                str(args.steps),
                "--episodes-per-round",
                "1",
                "--results-dir",
                fd,
                "--seed",
                str(args.seed),
            ],
            cwd=_HERE,
        )

        ma = manifest["subdirs"]["multi_agent_ac"]
        _run(
            [
                py,
                os.path.join(_HERE, "train", "train_multi_agent_ac.py"),
                "--sumocfg",
                joint,
                "--rounds",
                str(args.rounds),
                "--nodes",
                str(args.nodes),
                "--steps",
                str(args.steps),
                "--batch-size",
                "100",
                "--results-dir",
                ma,
                "--seed",
                str(args.seed),
            ],
            cwd=_HERE,
        )

        dq = manifest["subdirs"]["dqtsca"]
        _run(
            [
                py,
                os.path.join(_HERE, "train", "train_dqtsca.py"),
                "--sumocfg",
                joint,
                "--rounds",
                str(args.rounds),
                "--steps",
                str(args.steps),
                "--results-dir",
                dq,
                "--seed",
                str(args.seed),
            ],
            cwd=_HERE,
        )

    if not args.no_plot:
        print(
            "\n"
            + "=" * 72
            + "\n  PHASE: Training plots (included in this run — plot_research_training_curves.py)\n"
            + f"  → {os.path.join(base, 'plots_training')}\n"
            + "=" * 72
        )
        _run(
            [
                py,
                os.path.join(_HERE, "plot_research_training_curves.py"),
                "--results-dir",
                base,
            ],
            cwd=_HERE,
        )
    else:
        print("\n  [SKIP] Plots (--no-plot). Re-run with: python plot_research_training_curves.py --results-dir <this folder>\n")

    summary = aggregate_summary(base)
    with open(os.path.join(base, "research_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_completion_report(base, manifest, summary)

    print("\n" + "#" * 72)
    print("# SUCCESS — training" + (" + plots" if not args.no_plot else "") + " + reports finished")
    print(f"# Output: {os.path.abspath(base)}")
    if not args.no_plot:
        print(f"# Plots: {os.path.join(os.path.abspath(base), 'plots_training')}")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
