import numpy as np

def add_gaussian_noise(weights: np.ndarray, noise_std: float) -> np.ndarray:
    """
    Adds Gaussian noise to a numpy array of weights.
    Implements Eq. 5: w̃ = w + N(0, σ²I)
    """
    noise = np.random.normal(0, noise_std, weights.shape)
    return weights + noise

def compute_epsilon(noise_std: float, sensitivity: float, delta: float = 1e-5) -> float:
    """
    Computes the approximate differential privacy budget (epsilon).
    This is based on the standard Gaussian mechanism privacy bound.
    """
    if noise_std == 0:
        return float('inf') # No privacy if no noise is added
    epsilon = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / noise_std
    return epsilon
