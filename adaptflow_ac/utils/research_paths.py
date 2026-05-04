"""Resolve SUMO config paths for reproducible research runs."""
from __future__ import annotations

import os

_ADAPT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
_TCFL_ROOT = os.path.normpath(os.path.join(_ADAPT_ROOT, ".."))
_BACKEND = os.path.join(_TCFL_ROOT, "backend")


def sumo_configs2_joint_path() -> str:
    """Full-map config with all TLS in one simulation (baselines)."""
    return os.path.join(_BACKEND, "sumo_configs2", "osm.sumocfg")


def adaptflow_dwarka_mor_results_default() -> str:
    """Default AdaptFlow results dir when using --sumo-scenario dwarka_mor."""
    return os.path.join(_ADAPT_ROOT, "results_adaptflow_dwarka_mor")
