import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class CongestionStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        congestion_scores = []
        client_params = []

        # ---- extract metrics from each client ----
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            print(f"[Server] Received metrics from client: {metrics}")

            wait = float(metrics.get("waiting_time", 0))
            queue = float(metrics.get("queue_length", 0))
            congested = float(metrics.get("num_congested_lanes", 0))
            lanes = float(metrics.get("total_lanes", 1))

            # same formula as your simulator
            norm_wait = min(wait / 300.0, 1.0)
            norm_queue = min(queue / 50.0, 1.0)
            norm_cong = congested / max(lanes, 1.0)

            congestion = 0.4 * norm_wait + 0.3 * norm_queue + 0.3 * norm_cong
            print("score", congestion)

            congestion_scores.append(max(congestion, 0.0))
            client_params.append(parameters_to_ndarrays(fit_res.parameters))

        # ---- normalize weights ----
        total = sum(congestion_scores)
        if total == 0:
            weights = [1 / len(congestion_scores)] * len(congestion_scores)
        else:
            weights = [c / total for c in congestion_scores]

        print(
            f"\n[Server] Round {server_round} congestion weights:", np.round(weights, 3)
        )

        # ---- weighted average (same as your weighted_avg_params) ----
        new_params = []
        for layer in zip(*client_params):
            new_params.append(np.sum([w * p for w, p in zip(weights, layer)], axis=0))

        return ndarrays_to_parameters(new_params), {}
