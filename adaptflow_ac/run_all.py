"""
Train / compare entry. For a full paper-style run (train 4 methods + plots + summary),
prefer **`run_all_research.py`** (single command, preflight + reports).
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.research_paths import sumo_configs2_joint_path


def run_command(cmd: list, cwd: str) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"Error: exit code {r.returncode}")
        raise SystemExit(r.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="AdaptFlow AC suite: train baselines on joint sumocfg; "
        "AdaptFlow on --sumo-scenario (default dwarka_mor for sumo_configs2)."
    )
    parser.add_argument(
        "--sumocfg",
        type=str,
        default=None,
        help="Joint .sumocfg for FedDQN / MA2C / DQTSCA (default: backend/sumo_configs2/osm.sumocfg)",
    )
    parser.add_argument(
        "--sumo-scenario",
        type=str,
        default="dwarka_mor",
        help="AdaptFlow map preset (dwarka_mor = sumo_configs2 nodes)",
    )
    parser.add_argument(
        "--action", type=str, choices=["train", "compare", "all"], default="all"
    )
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sumo-headless",
        action="store_true",
        help="Use sumo (no GUI) for AdaptFlow when scenario defaults to GUI",
    )
    args = parser.parse_args()

    joint = args.sumocfg or sumo_configs2_joint_path()
    py = sys.executable
    train_dir = os.path.join(_HERE, "train")

    if args.action in ("train", "all"):
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING ALL 4 ALGORITHMS (unified rounds/steps/nodes)")
        print("=" * 60)

        af_flags = [
            py,
            os.path.join(train_dir, "train_adaptflow.py"),
            "--rounds",
            str(args.rounds),
            "--nodes",
            str(args.nodes),
            "--steps",
            str(args.steps),
            "--sumo-scenario",
            args.sumo_scenario,
        ]
        if args.sumo_headless:
            af_flags.append("--sumo-headless")
        run_command(af_flags, cwd=_HERE)

        run_command(
            [
                py,
                os.path.join(train_dir, "train_fed_dqn_tsc.py"),
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
                "--seed",
                str(args.seed),
            ],
            cwd=_HERE,
        )

        run_command(
            [
                py,
                os.path.join(train_dir, "train_multi_agent_ac.py"),
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
                "--seed",
                str(args.seed),
            ],
            cwd=_HERE,
        )

        run_command(
            [
                py,
                os.path.join(train_dir, "train_dqtsca.py"),
                "--sumocfg",
                joint,
                "--rounds",
                str(args.rounds),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
            ],
            cwd=_HERE,
        )

    if args.action in ("compare", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: COMPARISON (evaluation)")
        print("=" * 60)
        run_command(
            [
                py,
                os.path.join(_HERE, "compare.py"),
                "--mode",
                "sumo",
                "--sumocfg",
                joint,
                "--episodes",
                "5",
            ],
            cwd=_HERE,
        )


if __name__ == "__main__":
    main()
