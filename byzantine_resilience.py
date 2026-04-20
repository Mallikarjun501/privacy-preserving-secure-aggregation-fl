import numpy as np
from typing import List

def krum_filter(updates: List[np.ndarray]) -> np.ndarray:
    """
    Implements the Krum filter for Byzantine resilience.
    Selects the single most representative update from a list.
    
    Implements Eq. 9 and 10:
    - score_i = Σ_{j≠i} ||w_i - w_j||²
    - w^{t+1} = argmin_{w_i} score_i
    """
    if not updates:
        raise ValueError("Update list cannot be empty.")
    
    num_updates = len(updates)
    if num_updates == 1:
        return updates[0]

    scores = []
    for i in range(num_updates):
        score = 0
        for j in range(num_updates):
            if i == j:
                continue
            # Calculate squared Euclidean distance
            distance_sq = np.sum(np.square(updates[i] - updates[j]))
            score += distance_sq
        scores.append(score)
    
    # Find the index of the update with the minimum score
    min_score_index = np.argmin(scores)
    
    # Return the most representative update
    return updates[min_score_index]
