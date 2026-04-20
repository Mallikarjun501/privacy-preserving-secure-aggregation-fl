import numpy as np

class AdaptiveController:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def get_params_for_round(self):
        """
        Simulates dynamic network conditions and returns adaptive parameters for each client.
        """
        params = {}
        for client_id in range(self.num_clients):
            network_load = np.random.rand()  # Simulate network load: 0.0 to 1.0

            if network_load < 0.33:  # Good network
                noise_std = 0.005
                bit_precision = 8
                threshold = 0.001
            elif network_load < 0.66:  # Medium network
                noise_std = 0.01
                bit_precision = 6
                threshold = 0.005
            else:  # Poor network
                noise_std = 0.02
                bit_precision = 4
                threshold = 0.01
            
            params[client_id] = {
                "noise_std": noise_std,
                "bit_precision": bit_precision,
                "threshold": threshold
            }
        return params
