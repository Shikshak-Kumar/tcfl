"""
SUMO map presets. Default is unchanged: sumo_configs2 OSM client pair.

Use explicit `sumo_scenario` in WebSocket config, CLI `--sumo-scenario`, or env
`SUMO_SCENARIO` (e.g. china, china_osm).
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

CHINA_OSM_SUMO_CONFIGS: List[str] = [
    "sumo_configs_china_osm/osm_client1.sumocfg",
    "sumo_configs_china_osm/osm_client2.sumocfg",
]


def normalize_scenario(name: Optional[str]) -> str:
    if not name:
        return "default"
    n = str(name).strip().lower().replace("-", "_")
    if n in ("china", "sumo_configs_china", "cn"):
        return "china"
    if n in (
        "china_osm",
        "chinaosm",
        "sumo_configs_china_osm",
        "osm_china",
    ):
        return "china_osm"
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
    if s == "china_osm":
        return list(CHINA_OSM_SUMO_CONFIGS)
    return list(DEFAULT_SUMO_CONFIGS)


def scenario_results_suffix(resolved: str) -> str:
    if resolved == "china":
        return "_china"
    if resolved == "china_osm":
        return "_china_osm"
    return ""


def distinct_results_dir(
    default_dir: str,
    chosen_dir: str,
    sumo_scenario: Optional[str] = None,
) -> str:
    """
    If the user kept the script's default --results-dir and uses a China map preset,
    write under ``{default_dir}_china`` or ``{default_dir}_china_osm``.
    """
    if chosen_dir != default_dir:
        return chosen_dir
    resolved = effective_sumo_scenario(sumo_scenario)
    suf = scenario_results_suffix(resolved)
    if suf and not chosen_dir.endswith(suf):
        return f"{default_dir}{suf}"
    return chosen_dir


def deployment_model_subdir(algo_stem: str, sumo_scenario: Optional[str] = None) -> str:
    """saved_models/<name> or <name>_china / <name>_china_osm for API loading."""
    stem = algo_stem.strip().lower().replace(" ", "_")
    r = effective_sumo_scenario(sumo_scenario)
    if r == "china":
        return f"{stem}_china"
    if r == "china_osm":
        return f"{stem}_china_osm"
    return stem


def scenario_label_for_log(sumo_scenario: Optional[str] = None) -> str:
    r = effective_sumo_scenario(sumo_scenario)
    if r == "china_osm":
        return "China OSM"
    if r == "china":
        return "China (synthetic)"
    return "default"


def effective_sumo_headless(cli_flag: bool) -> bool:
    """
    True if --sumo-headless / --real-sumo was passed, or env SUMO_HEADLESS is 1/true/yes/on.
    Use this for real TraCI via `sumo` with no sumo-gui windows.
    """
    if cli_flag:
        return True
    v = os.environ.get("SUMO_HEADLESS", "").strip().lower()
    return v in ("1", "true", "yes", "on")
