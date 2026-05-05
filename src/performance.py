import numpy as np

def calculate_sinr_and_rate(H, allocated_powers, noise_variance, bandwidth=1e6):
    """
    Calculates SINR and the Shannon-Hartley capacity for each user.
    """
    num_users = H.shape[0]
    rates = np.zeros(num_users)
    
    for k in range(num_users):
        # FIX: Use np.abs() to get the magnitude squared for complex channels
        signal = (np.abs(H[k, k])**2) * allocated_powers[k]
        
        # FIX: Apply the same to the interference calculation
        interference = sum((np.abs(H[k, j])**2) * allocated_powers[j] for j in range(num_users) if j != k)
        
        # Calculate SINR
        sinr = signal / (interference + noise_variance)
        
        # Shannon-Hartley theorem calculation
        rates[k] = bandwidth * np.log2(1 + sinr)
        
    return rates