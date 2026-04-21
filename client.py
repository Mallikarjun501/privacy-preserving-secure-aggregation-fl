import argparse
import logging
import socket
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from data_loader import load_client_data
from differential_privacy import add_gaussian_noise
from device_utils import get_device
from homomorphic_encryption import encrypt_weights
from model import get_model
from pssa_compression import adaptive_quantization, sparse_gradient_sharing
from utils import recv_msg, send_msg


class FederatedClient:
    def __init__(self, client_id: str, local_loader, input_dim: int, host: str = "127.0.0.1", port: int = 12345, device: str = "auto"):
        self.client_id = client_id
        self.local_loader = local_loader
        self.device = get_device(device)
        self.model = get_model(input_dim).to(self.device)
        self.host = host
        self.port = port

        self.scale_factor = 1e6
        self.sock = None
        logging.info("[%s] Using device: %s", self.client_id, self.device)

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        send_msg(self.sock, {"type": "hello", "client_id": self.client_id})
        logging.info("[%s] Connected to server %s:%s", self.client_id, self.host, self.port)

    def set_model_weights(self, weights_vector: np.ndarray) -> None:
        vector_to_parameters(torch.from_numpy(weights_vector).float().to(self.device), self.model.parameters())

    def local_train(self, learning_rate: float = 0.01, local_epochs: int = 1) -> np.ndarray:
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for _ in range(local_epochs):
            for data, target in self.local_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return parameters_to_vector(self.model.parameters()).detach().cpu().numpy().copy()   

    def pssa_pipeline(self, global_weights: np.ndarray, public_key, noise_std: float, bit_precision: int, threshold: float):
        self.set_model_weights(global_weights)

        raw_weights = self.local_train(learning_rate=0.01, local_epochs=5)

        # Compute delta update relative to the received global model.
        delta_weights = raw_weights - global_weights

        # Apply PSSA pipeline on delta updates, not absolute weights.
        noisy_weights = add_gaussian_noise(delta_weights, noise_std)
        quantized_weights = adaptive_quantization(noisy_weights, bit_precision)
        sparse_weights = sparse_gradient_sharing(quantized_weights, threshold)

        # Encrypt all non-zero sparse values (paper-aligned behavior).
        nonzero_indices = np.nonzero(sparse_weights)[0]
        if len(nonzero_indices) == 0:
            return (
                [],
                [],
                sparse_weights,
                {"encryption_time_ms": 0.0},
                len(self.local_loader.dataset),
            )

        weights_to_encrypt = sparse_weights[nonzero_indices]
        weights_int = (weights_to_encrypt * self.scale_factor).astype(np.int64)

        start_time = time.time()
        encrypted_sparse_values = encrypt_weights(weights_int.tolist(), public_key)
        encryption_time_ms = (time.time() - start_time) * 1000.0

        return (
            nonzero_indices.tolist(),
            encrypted_sparse_values,
            sparse_weights,
            {"encryption_time_ms": encryption_time_ms},
            len(self.local_loader.dataset),
        )

    def run(self) -> None:
        try:
            self.connect()

            while True:
                msg = recv_msg(self.sock)
                if msg is None:
                    logging.warning("[%s] Server disconnected", self.client_id)
                    break

                msg_type = msg.get("type") if isinstance(msg, dict) else None
                if msg_type == "shutdown":
                    logging.info("[%s] Received shutdown", self.client_id)
                    break

                if msg_type != "round_start":
                    logging.warning("[%s] Unknown message: %s", self.client_id, msg_type)
                    continue

                round_idx = int(msg["round"])
                params = msg.get("params", {})
                noise_std = float(params.get("noise_std", 0.01))
                bit_precision = int(params.get("bit_precision", 8))
                threshold = float(params.get("threshold", 0.01))

                indices, encrypted_values, sparse_weights, timing_info, dataset_size = self.pssa_pipeline(
                    global_weights=np.asarray(msg["weights"], dtype=np.float64),
                    public_key=msg["public_key"],
                    noise_std=noise_std,
                    bit_precision=bit_precision,
                    threshold=threshold,
                )

                send_msg(
                    self.sock,
                    {
                        "type": "round_update",
                        "client_id": self.client_id,
                        "round": round_idx,
                        "indices": indices,
                        "encrypted_values": encrypted_values,
                        "sparse_weights": sparse_weights,
                        "dataset_size": dataset_size,
                        "timing": timing_info,
                    },
                )
                logging.info("[%s] Round %d update sent", self.client_id, round_idx)

        except Exception as exc:
            logging.error("[%s] Fatal error: %s", self.client_id, exc, exc_info=True)
        finally:
            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass
            logging.info("[%s] Client stopped", self.client_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated client")
    parser.add_argument("client_id", type=str, help="Client ID, e.g., A/B/C/D/E")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device to use")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    id_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    shard_id = id_map.get(args.client_id.upper())
    if shard_id is None:
        raise ValueError("client_id must be one of A, B, C, D, E")

    local_loader, input_dim = load_client_data(shard_id, data_dir="data")
    client = FederatedClient(client_id=args.client_id.upper(), local_loader=local_loader, input_dim=input_dim, host=args.host, port=args.port, device=args.device)
    client.run()
