"""
SUMO map presets. Default is unchanged: sumo_configs2 OSM client pair.

Use explicit `sumo_scenario` in WebSocket config, CLI `--sumo-scenario`, or env
`SUMO_SCENARIO=china` (only when the request/script does not override).
"""
from __future__ import annotations

import os
from typing import List, Optional

DEFAULT_SUMO_CONFIGS: List[str] = [
    "sumo_configs2/osm_client1.sumocfg",
    "sumo_configs2/osm_client2.sumocfg",
]

CHINA_SUMO_CONFIGS: List[str] = [
    "sumo_configs_china/china_client1.sumocfg",
    "sumo_configs_china/china_client2.sumocfg",
]


def normalize_scenario(name: Optional[str]) -> str:
    if not name:
        return "default"
    n = str(name).strip().lower()
    if n in ("china", "sumo_configs_china", "cn"):
        return "china"
    return "default"


def effective_sumo_scenario(config_explicit: Optional[str] = None) -> str:
    """Prefer non-empty WebSocket/CLI value; else SUMO_SCENARIO env; else default."""
    if config_explicit is not None and str(config_explicit).strip() != "":
        return normalize_scenario(config_explicit)
    return normalize_scenario(os.environ.get("SUMO_SCENARIO"))


def get_sumo_config_paths(scenario: Optional[str] = None) -> List[str]:
    s = normalize_scenario(scenario) if scenario is not None else effective_sumo_scenario()
    if s == "china":
        return list(CHINA_SUMO_CONFIGS)
    return list(DEFAULT_SUMO_CONFIGS)


def distinct_results_dir(
    default_dir: str,
    chosen_dir: str,
    sumo_scenario: Optional[str] = None,
) -> str:
    """
    If the user kept the script's default --results-dir and training uses the China
    map, write under ``{default_dir}_china`` so runs stay separate from OSM baselines.
    """
    if chosen_dir != default_dir:
        return chosen_dir
    if effective_sumo_scenario(sumo_scenario) == "china" and not chosen_dir.endswith(
        "_china"
    ):
        return f"{default_dir}_china"
    return chosen_dir


def deployment_model_subdir(algo_stem: str, sumo_scenario: Optional[str] = None) -> str:
    """saved_models/<name> or saved_models/<name>_china for API loading."""
    stem = algo_stem.strip().lower().replace(" ", "_")
    if effective_sumo_scenario(sumo_scenario) == "china":
        return f"{stem}_china"
    return stem
