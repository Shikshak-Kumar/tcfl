"""
Map discrete RL actions (default 4 movement groups) to SUMO traffic-light phase indices.

Compatible with SUMO 1.26+ where TraCI may omit trafficlight.getPhaseNumber — we derive
the phase count from the active program definition.
"""
from __future__ import annotations

from typing import Any

try:
    import traci
except ImportError:  # pragma: no cover
    traci = None  # type: ignore


def tls_phase_count(tl_id: str) -> int:
    """
    Number of phases in the TLS current program (SUMO 1.18–1.26+).

    Older SUMO: trafficlight.getPhaseNumber(tl_id).
    Newer SUMO: use getCompleteRedYellowGreenDefinition / getAllProgramLogics.
    """
    if traci is None or not traci.isLoaded():
        return 4

    dom = traci.trafficlight

    getter = getattr(dom, "getPhaseNumber", None)
    if callable(getter):
        try:
            n = int(getter(tl_id))
            if n > 0:
                return n
        except Exception:
            pass

    logics: Any = None
    for name in ("getCompleteRedYellowGreenDefinition", "getAllProgramLogics"):
        fn = getattr(dom, name, None)
        if not callable(fn):
            continue
        try:
            logics = fn(tl_id)
            if logics:
                break
        except Exception:
            logics = None

    if not logics:
        return 4

    logic = logics[0]
    phases = getattr(logic, "phases", None)
    if phases is None and isinstance(logic, dict):
        phases = logic.get("phases")
    if phases is not None:
        try:
            n = len(phases)
            return max(1, n)
        except TypeError:
            pass

    return 4


def phase_index_for_discrete_action(
    n_phase: int,
    action: int,
    n_actions: int = 4,
) -> int:
    """Return a valid phase index in [0, n_phase-1] for discrete action ∈ [0, n_actions-1]."""
    if n_phase <= 0:
        return 0
    a = max(0, min(n_actions - 1, int(action)))
    if n_phase <= n_actions:
        return min(a, n_phase - 1)
    if n_phase >= 2 * n_actions:
        return min(a * 2, n_phase - 1)
    return min(
        int(round(a * (n_phase - 1) / max(1, n_actions - 1))),
        n_phase - 1,
    )


def set_phase_from_discrete_action(
    tl_id: str,
    action: int,
    n_actions: int = 4,
) -> None:
    """Apply discrete RL action to TLS via TraCI."""
    if traci is None or not traci.isLoaded():
        return
    n = tls_phase_count(tl_id)
    idx = phase_index_for_discrete_action(n, action, n_actions)
    traci.trafficlight.setPhase(tl_id, idx)
