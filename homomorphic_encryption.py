from phe import paillier

def generate_paillier_keypair(key_length=1024):
    """Generates a Paillier keypair."""
    print(f"Generating Paillier keypair with key length: {key_length} bits...")
    public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
    print("Keypair generated.")
    return public_key, private_key

def encrypt_weights(weights_int_list, public_key):
    """Encrypts a list of integers using the public key."""
    return [public_key.encrypt(w) for w in weights_int_list]

def secure_aggregation(list_of_encrypted_updates, public_key):
    """
    Aggregates encrypted model updates from multiple clients.
    Implements Eq. 4: E(w^{t+1}) = Π E(w_i^{t+1}) mod N²
    In phe, this corresponds to the sum of encrypted numbers.
    """
    if not list_of_encrypted_updates:
        return []
    
    # Ensure all lists have the same length
    num_params = len(list_of_encrypted_updates[0])
    if any(len(update) != num_params for update in list_of_encrypted_updates):
        raise ValueError("All client updates must have the same number of parameters.")

    # Sum the encrypted weights element-wise
    aggregated_update = list(list_of_encrypted_updates[0]) # Start with the first client's update
    for i in range(1, len(list_of_encrypted_updates)):
        for j in range(num_params):
            aggregated_update[j] += list_of_encrypted_updates[i][j]
            
    return aggregated_update

def decrypt_weights(encrypted_aggregated, private_key, num_clients, scale_factor=1e6):
    """
    Decrypts the aggregated update, averages it, and converts back to float.
    """
    if not encrypted_aggregated:
        return []
        
    decrypted_sum = [private_key.decrypt(w) for w in encrypted_aggregated]
    
    import numpy as np
    # Average the weights and rescale
    averaged_weights = np.array(decrypted_sum) / (num_clients * scale_factor)
    
    return averaged_weights
