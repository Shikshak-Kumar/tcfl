# Research run complete

- **Output directory:** `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846`
- **UTC timestamp (manifest):** 20260503_192846

## Hyperparameters

```json
{
  "rounds": 10,
  "nodes": 6,
  "steps_per_episode": 2500,
  "seed": 42,
  "fed_dqn_episodes_per_round": 1,
  "fed_dqn_fine_tune": false,
  "ma2c_batch_size": 100,
  "dqtsca_total_train_steps": 100000,
  "adaptflow_tp_reward_scale": 0.08,
  "adaptflow_completion_bonus_scale": 15.0
}
```

## Files generated

| Artifact | Path |
|----------|------|
| Manifest | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\training_manifest.json` |
| Training curves (combined) | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\plots_training\training_curves_combined.png` |
| Per-metric lines | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\plots_training\line_*.png` |
| Aggregated metrics | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\research_summary.json` |
| AdaptFlow rounds JSON | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\adaptflow\adaptflow_all_rounds.json` |
| FedDQN rounds JSON | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\fed_dqn_tsc\fed_dqn_tsc_all_rounds.json` |
| MA2C by-round JSON | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\multi_agent_ac\multi_agent_ac_by_round.json` |
| DQTSCA training JSON | `F:\ttraffic\tcfl\adaptflow_ac\results_research_sumo_configs2_20260503_192846\dqtsca\dqtsca_training.json` |

## Final-round snapshot (training logs)

```json
{
  "results_dir": "F:\\ttraffic\\tcfl\\adaptflow_ac\\results_research_sumo_configs2_20260503_192846",
  "adaptflow": {
    "round": 10,
    "mean_total_reward": 61.65560941405919,
    "mean_queue": 0.07106666666666668,
    "mean_wait_s": 0.2530666666666667,
    "mean_tp_ratio": 0.9106914212548015,
    "mean_arrival_rate": 0.42688668220849385,
    "mean_loss": -3.103645042826732
  },
  "fed_dqn_tsc": {
    "round": 10,
    "avg_reward": -29.94121031746036,
    "avg_loss": 0.0010050427622523546,
    "avg_queue": 0.0,
    "avg_wait_s": 0.0,
    "avg_tp_ratio": 0.912291933418694,
    "avg_arrival_rate": 0.57
  },
  "multi_agent_ac": {
    "episode": 10,
    "avg_reward": -0.014659845238095295,
    "avg_loss": 0.9315526686391483,
    "avg_queue": 0.0,
    "avg_wait_s": 0.0,
    "tp_ratio": 0.912291933418694,
    "arrival_rate": 0.57
  },
  "dqtsca": {
    "episode": 10,
    "total_reward": 0.0,
    "avg_loss": 7.56461260934826,
    "avg_queue": 0.0,
    "avg_wait_s": 0.0,
    "tp_ratio": 1.0,
    "arrival_rate": 0.6110044486068836
  }
}
```

## Notes (methods / demand — conference write-up)

- **Demand parity:** All Dwarka Mor configs use **begin=0, end=3600** and the same `osm.passenger.trips.xml`, matching **`osm.sumocfg`**, so **vehicle insertion is aligned** across AdaptFlow federated nodes and FedDQN / MA2C / DQTSCA.
- **AdaptFlow** still trains **six separate SUMO processes** (one TLS focal per config); heterogeneity is **TLS/processing** (e.g. jam-threshold), not staggered demand windows.
- **Baselines** use **joint** `osm.sumocfg` (one process, all TLS interact).
- **DQTSCA** acts on the **first** TLS ID in the joint net; cite as single-intersection control.
- **Throughput plots** use **arrival rate** (completed trips per simulation second) when logged; the legacy ratio arrived÷departed is often ~0.5–0.55 and is a weak control signal.
- **Cross-method throughput:** AdaptFlow uses **per-node** SUMO scenarios; baselines use the **joint** net. Absolute arrival rates are not always comparable—use **queue / delay** for primary cross-method ranking, and arrival curves mainly for **within-method** learning trends.
- **TP ratio (trip completion):** `arrived ÷ departed` stays low if **episodes are short** (many vehicles still driving). Default **`--steps 2500`** is the same for all four algorithms (fair horizon). Use `--steps 3600` for a full-hour sumocfg. AdaptFlow may use `--adaptflow-tp-reward` and `--adaptflow-completion-bonus` (see manifest); set both to **0** for identical reward definition vs baselines.
