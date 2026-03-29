"""
SUMO map presets. Default is unchanged: sumo_configs2 OSM client pair.

Use explicit `sumo_scenario` in WebSocket config, CLI `--sumo-scenario`, or env
`SUMO_SCENARIO` (e.g. china, china_osm, china_rural_osm, india_rural_osm).
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
    # One dedicated config per node: distinct begin-time offsets + TLS parameters
    # so each node experiences genuinely different congestion fingerprints.
    "sumo_configs_china_osm/osm_node0.sumocfg",  # Hospital  – begin=0,    jam=20
    "sumo_configs_china_osm/osm_node1.sumocfg",  # School    – begin=400,  jam=25
    "sumo_configs_china_osm/osm_node2.sumocfg",  # Commerce  – begin=800,  jam=30
    "sumo_configs_china_osm/osm_node3.sumocfg",  # Resident  – begin=1200, jam=35
    "sumo_configs_china_osm/osm_node4.sumocfg",  # Industry  – begin=1600, jam=40
    "sumo_configs_china_osm/osm_node5.sumocfg",  # Transit   – begin=2000, jam=28
]

CHINA_RURAL_OSM_SUMO_CONFIGS: List[str] = [
    # Rural OSM map — 9270 vehicles, ~2.56 veh/s. Six nodes with distinct begin
    # offsets (600 s apart) and varied jam-thresholds for rural road characteristics.
    "sumo_configs_china_rural_osm/osm_node0.sumocfg",  # Village market  – begin=0,    jam=15
    "sumo_configs_china_rural_osm/osm_node1.sumocfg",  # Rural school    – begin=600,  jam=20
    "sumo_configs_china_rural_osm/osm_node2.sumocfg",  # Farming district– begin=1200, jam=30
    "sumo_configs_china_rural_osm/osm_node3.sumocfg",  # Residential out – begin=1800, jam=35
    "sumo_configs_china_rural_osm/osm_node4.sumocfg",  # Industrial zone – begin=2400, jam=45
    "sumo_configs_china_rural_osm/osm_node5.sumocfg",  # Highway junction– begin=3000, jam=22
]

INDIA_RURAL_OSM_SUMO_CONFIGS: List[str] = [
    # India rural OSM map — 3355 vehicles over 3600 s (~0.93 veh/s).
    # Six nodes with 400 s begin-time offsets and India-specific TLS parameters
    # (higher teleport patience, varied jam-thresholds for mixed rural traffic).
    "sumo_configs_india_rural_osm/osm_node0.sumocfg",  # Town market      – begin=0,    jam=12
    "sumo_configs_india_rural_osm/osm_node1.sumocfg",  # School/temple    – begin=400,  jam=18
    "sumo_configs_india_rural_osm/osm_node2.sumocfg",  # Farming district – begin=800,  jam=25
    "sumo_configs_india_rural_osm/osm_node3.sumocfg",  # Village resident – begin=1200, jam=30
    "sumo_configs_india_rural_osm/osm_node4.sumocfg",  # Industrial/hwy   – begin=1600, jam=40
    "sumo_configs_india_rural_osm/osm_node5.sumocfg",  # State hwy junc   – begin=2000, jam=22
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
    if n in (
        "china_rural_osm",
        "chinaruralosm",
        "rural_osm",
        "sumo_configs_china_rural_osm",
        "china_rural",
    ):
        return "china_rural_osm"
    if n in (
        "india_rural_osm",
        "indiaruralosm",
        "india_rural",
        "sumo_configs_india_rural_osm",
        "india",
        "in_rural",
    ):
        return "india_rural_osm"
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
    if s == "china_rural_osm":
        return list(CHINA_RURAL_OSM_SUMO_CONFIGS)
    if s == "india_rural_osm":
        return list(INDIA_RURAL_OSM_SUMO_CONFIGS)
    return list(DEFAULT_SUMO_CONFIGS)


def scenario_results_suffix(resolved: str) -> str:
    if resolved == "china":
        return "_china"
    if resolved == "china_osm":
        return "_china_osm"
    if resolved == "china_rural_osm":
        return "_china_rural_osm"
    if resolved == "india_rural_osm":
        return "_india_rural_osm"
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
    """saved_models/<name> or <name>_china / <name>_china_osm / <name>_china_rural_osm."""
    stem = algo_stem.strip().lower().replace(" ", "_")
    r = effective_sumo_scenario(sumo_scenario)
    if r == "china":
        return f"{stem}_china"
    if r == "china_osm":
        return f"{stem}_china_osm"
    if r == "china_rural_osm":
        return f"{stem}_china_rural_osm"
    if r == "india_rural_osm":
        return f"{stem}_india_rural_osm"
    return stem


def scenario_label_for_log(sumo_scenario: Optional[str] = None) -> str:
    r = effective_sumo_scenario(sumo_scenario)
    if r == "china_osm":
        return "China OSM"
    if r == "china_rural_osm":
        return "China Rural OSM"
    if r == "india_rural_osm":
        return "India Rural OSM"
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


def effective_training_gui(
    sumo_scenario: Optional[str],
    use_tomtom: bool,
    gui_cli: bool,
    sumo_headless: bool,
) -> bool:
    """
    ``china`` / ``china_osm`` / ``china_rural_osm`` / ``india_rural_osm`` run real SUMO
    with sumo-gui by default (no mock), unless headless is enabled or TomTom drives the
    backend.  Default and other scenarios keep ``gui_cli`` (mock when False and not headless).
    """
    if sumo_headless:
        return False
    if use_tomtom:
        return gui_cli
    if effective_sumo_scenario(sumo_scenario) in (
        "china", "china_osm", "china_rural_osm", "india_rural_osm"
    ):
        return True
    return gui_cli
