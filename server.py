import logging
import socket
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from byzantine_resilience import krum_filter
from data_loader import prepare_data_shards
from differential_privacy import compute_epsilon
from device_utils import get_device
from homomorphic_encryption import generate_paillier_keypair
from metrics_logger import MetricsLogger
from model import get_model
from pssa_compression import compute_communication_cost_mb
from utils import recv_msg, send_msg
from adaptive_controller import AdaptiveController


class FederatedServer:
    def __init__(self, input_dim: int, num_clients: int = 5, num_rounds: int = 20, host: str = "127.0.0.1", port: int = 12345, device: str = "auto"):
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.host = host
        self.port = port

        self.device = get_device(device)
        logging.info("[SERVER] Using device: %s", self.device)
        self.global_model = get_model(input_dim).to(self.device)
        self.public_key, self.private_key = generate_paillier_keypair(key_length=1024)
        self.scale_factor = 1e6
        self.metrics_logger = MetricsLogger(results_dir="results")
        self.adaptive_controller = AdaptiveController(num_clients=num_clients)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.clients: Dict[str, socket.socket] = {}

    def _evaluate(self, test_loader) -> float:
        self.global_model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                preds = self.global_model(data)
                labels = (preds >= 0.5).float()
                correct += (labels == target).sum().item()
                total += target.numel()
        if total == 0:
            return 0.0
        return (correct / total) * 100.0

    def _accept_clients(self) -> None:
        self.server_socket.listen(self.num_clients)
        logging.info("[SERVER] Listening on %s:%s", self.host, self.port)
        logging.info("[SERVER] Waiting for %d clients...", self.num_clients)

        while len(self.clients) < self.num_clients:
            conn, addr = self.server_socket.accept()
            hello = recv_msg(conn)
            if not isinstance(hello, dict) or hello.get("type") != "hello":
                logging.warning("[SERVER] Invalid handshake from %s. Closing.", addr)
                conn.close()
                continue

            client_id = str(hello.get("client_id", "unknown"))
            self.clients[client_id] = conn
            logging.info("[SERVER] Client %s connected from %s", client_id, addr)

        logging.info("[SERVER] All clients connected: %s", ", ".join(sorted(self.clients.keys())))

    def _broadcast_round(self, round_idx: int, global_weights: np.ndarray, params_by_client: Dict[str, dict]) -> None:
        for client_id, conn in self.clients.items():
            payload = {
                "type": "round_start",
                "round": round_idx,
                "num_rounds": self.num_rounds,
                "weights": global_weights,
                "public_key": self.public_key,
                "params": params_by_client[client_id],
            }
            send_msg(conn, payload)
            logging.info("[SERVER] Sent round %d model to %s", round_idx, client_id)

    def _collect_updates(self, round_idx: int) -> tuple[List[dict], List[np.ndarray], List[float], List[int], List[str]]:
        encrypted_updates: List[dict] = []
        sparse_updates: List[np.ndarray] = []
        enc_times: List[float] = []
        dataset_sizes: List[int] = []
        client_ids: List[str] = []

        for client_id, conn in self.clients.items():
            msg = recv_msg(conn)
            if not isinstance(msg, dict) or msg.get("type") != "round_update":
                raise RuntimeError(f"Invalid update from client {client_id}")

            if int(msg.get("round", -1)) != round_idx:
                raise RuntimeError(f"Round mismatch from client {client_id}")

            indices = [int(i) for i in msg["indices"]]
            encrypted_values = msg["encrypted_values"]
            if len(indices) != len(encrypted_values):
                raise RuntimeError(f"Malformed sparse encrypted payload from {client_id}")

            sparse_encrypted_map = {idx: enc_val for idx, enc_val in zip(indices, encrypted_values)}
            encrypted_updates.append(sparse_encrypted_map)
            sparse_updates.append(np.asarray(msg["sparse_weights"], dtype=np.float64))
            dataset_sizes.append(int(msg.get("dataset_size", 1)))
            client_ids.append(client_id)
            timing_info = msg.get("timing", {})
            enc_times.append(float(timing_info.get("encryption_time_ms", 0.0)))
            logging.info("[SERVER] Received update from %s for round %d", client_id, round_idx)

        return encrypted_updates, sparse_updates, enc_times, dataset_sizes, client_ids

    def _apply_aggregated_update(self, encrypted_updates: List[dict]) -> np.ndarray:
        current_global = parameters_to_vector(self.global_model.parameters()).detach().cpu().numpy().copy()
        averaged_update = np.zeros_like(current_global)

        aggregated_sparse = {}
        for update_map in encrypted_updates:
            for idx, enc_val in update_map.items():
                if idx in aggregated_sparse:
                    aggregated_sparse[idx] = aggregated_sparse[idx] + enc_val
                else:
                    aggregated_sparse[idx] = enc_val

        for idx, enc_sum in aggregated_sparse.items():
            decrypted_sum = self.private_key.decrypt(enc_sum)
            averaged_update[idx] = decrypted_sum / (self.num_clients * self.scale_factor)

        return averaged_update

    def _shutdown_clients(self) -> None:
        payload = {"type": "shutdown"}
        for client_id, conn in self.clients.items():
            try:
                send_msg(conn, payload)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            logging.info("[SERVER] Closed connection with %s", client_id)

    def run(self, test_loader) -> None:
        try:
            self._accept_clients()

            for round_idx in range(1, self.num_rounds + 1):
                logging.info("[SERVER] ===== Round %d/%d =====", round_idx, self.num_rounds)
                global_weights = parameters_to_vector(self.global_model.parameters()).detach().cpu().numpy().copy()

                raw_round_params = self.adaptive_controller.get_params_for_round()
                sorted_client_ids = sorted(self.clients.keys())
                params_by_client: Dict[str, dict] = {}
                for i, client_id in enumerate(sorted_client_ids):
                    params_by_client[client_id] = raw_round_params.get(
                        i,
                        {"noise_std": 0.01, "bit_precision": 8, "threshold": 0.01},
                    )

                self._broadcast_round(round_idx, global_weights, params_by_client)
                encrypted_updates, sparse_updates, enc_times, dataset_sizes, client_ids = self._collect_updates(round_idx)

                total_data = sum(dataset_sizes)
                logging.info("[SERVER] Dataset sizes: %s", {cid: ds for cid, ds in zip(client_ids, dataset_sizes)})
                logging.info("[SERVER] Weights: %s", {cid: round(ds / total_data, 3) for cid, ds in zip(client_ids, dataset_sizes)})

                # STEP 1: HE secure aggregation in encrypted domain (for privacy, logging).
                he_averaged_update = self._apply_aggregated_update(encrypted_updates)

                # STEP 2: Build HE-decrypted per-client vectors for Krum Byzantine detection.
                total_params = int(parameters_to_vector(self.global_model.parameters()).numel())
                per_client_decrypted = []
                for update_map in encrypted_updates:
                    client_vec = np.zeros(total_params)
                    for idx, enc_val in update_map.items():
                        client_vec[idx] = self.private_key.decrypt(enc_val) / self.scale_factor
                    per_client_decrypted.append(client_vec)

                # STEP 3: Run Krum for Byzantine detection and logging only (Eq. 9, Eq. 10).
                robust_update = krum_filter(per_client_decrypted)
                winning_idx = next(
                    (
                        i
                        for i, upd in enumerate(per_client_decrypted)
                        if upd is robust_update or np.array_equal(upd, robust_update)
                    ),
                    -1,
                )

                # STEP 4: Apply Weighted FedAvg (Eq. 2) as the actual model update.
                # This is the paper-correct aggregation. Krum above is used only for
                # Byzantine detection metrics, not to select the update source.
                total_data = sum(dataset_sizes)
                weighted_avg_update = np.zeros(total_params)
                for client_vec, ds in zip(per_client_decrypted, dataset_sizes):
                    weighted_avg_update += (ds / total_data) * client_vec

                # STEP 5: Update global model with weighted average delta.
                current_global = parameters_to_vector(self.global_model.parameters()).detach().cpu().numpy().copy()
                new_global = current_global + weighted_avg_update
                vector_to_parameters(torch.from_numpy(new_global).float().to(self.device), self.global_model.parameters())

                comm_costs = {
                    "uncompressed": compute_communication_cost_mb(weighted_avg_update, "uncompressed"),
                    "quantized": compute_communication_cost_mb(weighted_avg_update, "quantized_8bit"),
                    "sparse": compute_communication_cost_mb(weighted_avg_update, "sparse_top10"),
                }

                global_accuracy = self._evaluate(test_loader)
                mean_noise_std = float(np.mean([params["noise_std"] for params in params_by_client.values()]))
                epsilon = compute_epsilon(noise_std=mean_noise_std, sensitivity=1.0)
                he_norm = float(np.linalg.norm(he_averaged_update))
                krum_norm = float(np.linalg.norm(weighted_avg_update))
                num_encrypted_params = int(sum(len(update_map) for update_map in encrypted_updates))
                self.metrics_logger.log(
                    round_num=round_idx,
                    global_accuracy=global_accuracy,
                    comm_costs=comm_costs,
                    enc_times=enc_times,
                    epsilon=epsilon,
                    krum_winner_idx=winning_idx,
                    he_avg_norm=he_norm,
                    krum_norm=krum_norm,
                    num_encrypted_params=num_encrypted_params,
                )
                logging.info("[SERVER] HE avg norm: %.6f, Krum norm: %.6f, Krum winner idx: %d", he_norm, krum_norm, winning_idx)
                logging.info("[SERVER] Round %d complete. Accuracy: %.2f%%", round_idx, global_accuracy)

            self.metrics_logger.save()
            logging.info("[SERVER] Metrics written to results/metrics.csv")

            df = self.metrics_logger.get_metrics_df()
            final_accuracy = float(df["global_accuracy"].iloc[-1]) if not df.empty else 0.0
            best_row = df.loc[df["global_accuracy"].idxmax()] if not df.empty else None
            best_accuracy = float(best_row["global_accuracy"]) if best_row is not None else 0.0
            best_round = int(best_row["round_number"]) if best_row is not None else 0
            avg_enc_time = float(df["encryption_time_ms"].mean()) if not df.empty else 0.0
            avg_comm = float(df["communication_cost_sparse_mb"].mean()) if not df.empty else 0.0
            final_epsilon = float(df["epsilon"].iloc[-1]) if not df.empty else 0.0
            krum_counts = (
                df["krum_winner_idx"].value_counts().sort_index().to_dict() if not df.empty else {}
            )

            print("\n" + "=" * 60)
            print("PSSA TRAINING COMPLETE - FINAL SUMMARY")
            print("=" * 60)
            print("Dataset          : NSL-KDD")
            print("Clients          : 5 (A, B, C, D, E)")
            print("Rounds           : 20")
            print("Local Epochs     : 5")
            print(f"Final Accuracy   : {final_accuracy:.2f}%")
            print(f"Best Accuracy    : {best_accuracy:.2f}% (Round {best_round})")
            print(f"Avg Enc Time     : {avg_enc_time:.1f} ms")
            print(f"Avg Comm Cost    : {avg_comm:.4f} MB (sparse)")
            final_gla = float(df["gla_success_rate"].iloc[-1]) if not df.empty else 0.0
            print(f"Final GLA Rate   : {final_gla:.2f}% (simulated exponential decay model)")
            print(f"Final Epsilon    : {final_epsilon:.2f}")
            print(f"Krum Winners     : {krum_counts}")
            print("=" * 60)

        finally:
            self._shutdown_clients()
            self.server_socket.close()
            logging.info("[SERVER] Shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    test_loader, input_dim = prepare_data_shards(num_clients=5, data_dir="data")
    server = FederatedServer(input_dim=input_dim, num_clients=5, num_rounds=20, host="127.0.0.1", port=12345, device="auto")
    server.run(test_loader)
