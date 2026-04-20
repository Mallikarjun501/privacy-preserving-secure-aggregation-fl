import numpy as np

def adaptive_quantization(weights: np.ndarray, bit_precision: int) -> np.ndarray:
    """
    Applies adaptive quantization to the weights.
    Implements Eq. 7: Q(w) = floor(w * 2^b) / 2^b
    """
    scale = 2**bit_precision
    return np.floor(weights * scale) / scale

def sparse_gradient_sharing(weights: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies sparse gradient sharing by zeroing out weights below a threshold.
    Implements Eq. 8: S(w) = { w_j | |w_j| > τ }
    """
    sparse_weights = np.copy(weights)
    sparse_weights[np.abs(sparse_weights) < threshold] = 0
    return sparse_weights

def compute_communication_cost_mb(weights: np.ndarray, mode: str) -> float:
    """
    Computes the communication cost in megabytes for a given set of weights.
    """
    if mode == "uncompressed":
        # float32 is 4 bytes per parameter
        return (weights.size * 4) / 1e6
    elif mode == "quantized_8bit":
        # 8-bit quantization means 1 byte per parameter
        return weights.size / 1e6
    elif mode == "sparse_top10":
        # Cost of non-zero elements (float32)
        non_zero_weights = np.count_nonzero(weights)
        return (non_zero_weights * 4) / 1e6
    else:
        raise ValueError(f"Unknown mode: {mode}")
