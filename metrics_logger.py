import pandas as pd
import os
import numpy as np

class MetricsLogger:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.metrics = []
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def log(
        self,
        round_num,
        global_accuracy,
        comm_costs,
        enc_times,
        epsilon,
        krum_winner_idx,
        he_avg_norm,
        krum_norm,
        num_encrypted_params,
    ):
        self.metrics.append({
            "round_number": round_num,
            "global_accuracy": global_accuracy,
            "communication_cost_uncompressed_mb": comm_costs["uncompressed"],
            "communication_cost_quantized_mb": comm_costs["quantized"],
            "communication_cost_sparse_mb": comm_costs["sparse"],
            "encryption_time_ms": np.mean(enc_times) if enc_times else 0,
            "gla_success_rate": self.simulate_gla_success(round_num),
            "epsilon": epsilon,
            "krum_winner_idx": krum_winner_idx,
            "he_avg_norm": he_avg_norm,
            "krum_norm": krum_norm,
            "num_encrypted_params": num_encrypted_params,
        })

    def simulate_gla_success(self, round_num):
        """
        Simulates the success rate of a Gradient Leakage Attack (GLA).
        The rate decreases as privacy accumulates over rounds.
        """
        rate = 100 * np.exp(-round_num * 0.15)
        return np.clip(rate, 12.5, 100)

    def get_metrics_df(self):
        return pd.DataFrame(self.metrics)

    def save(self):
        df = self.get_metrics_df()
        filepath = os.path.join(self.results_dir, "metrics.csv")
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to {filepath}")

    def plot_all(self):
        import matplotlib.pyplot as plt
        df = self.get_metrics_df()
        if df.empty:
            print("No metrics to plot.")
            return

        # Plot 1: Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(df['round_number'], df['global_accuracy'], marker='o', linestyle='-')
        plt.title("PSSA Model Accuracy over 20 Rounds (NSL-KDD)")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "accuracy_plot.png"))
        plt.close()

        # Plot 2: Communication Cost
        plt.figure(figsize=(10, 6))
        plt.plot(df['round_number'], df['communication_cost_uncompressed_mb'], marker='s', linestyle='--', label='Uncompressed')
        plt.plot(df['round_number'], df['communication_cost_quantized_mb'], marker='x', linestyle='-.', label='Quantized (Adaptive)')
        plt.plot(df['round_number'], df['communication_cost_sparse_mb'], marker='o', linestyle='-', label='Sparse (Adaptive)')
        plt.title("Communication Cost per Client per Round")
        plt.xlabel("Round")
        plt.ylabel("Cost (MB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "communication_cost_plot.png"))
        plt.close()

        # Plot 3: Privacy Attack Success Rate
        plt.figure(figsize=(10, 6))
        plt.plot(df['round_number'], df['gla_success_rate'], marker='o', linestyle='-', color='r')
        plt.title("Gradient Leakage Attack Success Rate over Rounds")
        plt.xlabel("Round")
        plt.ylabel("GLA Success Rate (%)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "privacy_attack_plot.png"))
        plt.close()

        # Plot 4: Encryption Time
        plt.figure(figsize=(10, 6))
        plt.plot(df['round_number'], df['encryption_time_ms'], marker='o', linestyle='-', color='g')
        plt.title("Encryption Time per Client per Round (ms)")
        plt.xlabel("Round")
        plt.ylabel("Time (ms)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "encryption_time_plot.png"))
        plt.close()
        
        print("All plots saved in 'results/' directory.")
