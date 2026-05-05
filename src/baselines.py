import numpy as np

def iterative_water_filling(channel_gains, noise_variance, P_max):
    """
    Allocates power across independent channels using the Water-Filling algorithm.
    """
    # The "vessel" bottom is determined by noise variance and inverse channel gain
    inverse_snr = noise_variance / channel_gains
    
    # Bisection search to find the optimal water level (mu)
    low = 0.0
    high = P_max + np.max(inverse_snr)
    tolerance = 1e-5
    
    while high - low > tolerance:
        mu = (low + high) / 2
        # Power is the water level minus the inverse SNR (bounded at 0)
        allocated_power = np.maximum(mu - inverse_snr, 0)
        
        # Check against the finite total power budget P_max
        if np.sum(allocated_power) > P_max:
            high = mu
        else:
            low = mu
            
    return np.maximum(low - inverse_snr, 0)

def wmmse_power_allocation(H, P_max, noise_variance, iterations=100):
    """
    Allocates power using rigorous WMMSE with a KKT bisection search for the power constraint.
    """
    num_users = H.shape[0]
    v = np.sqrt(P_max / num_users) * np.ones(num_users)
    u = np.zeros(num_users)
    w = np.zeros(num_users)
    
    for _ in range(iterations):
        # 1. Update Receiver Weights (u)
        for k in range(num_users):
            interference = sum((np.abs(H[k, j]) * v[j])**2 for j in range(num_users)) + noise_variance
            u[k] = (np.abs(H[k, k]) * v[k]) / interference
            
        # 2. Update MMSE Weights (w)
        for k in range(num_users):
            w[k] = 1 / (1 - u[k] * np.abs(H[k, k]) * v[k])
            
        # 3. Update Transmitter Powers (v) using Bisection Search (KKT condition)
        a = np.zeros(num_users)
        b = np.zeros(num_users)
        
        for k in range(num_users):
            a[k] = w[k] * u[k] * np.abs(H[k, k])
            b[k] = sum(w[j] * (u[j]**2) * (np.abs(H[j, k])**2) for j in range(num_users))
            
        # Bisection search for the Lagrange multiplier
        low, high = 0.0, 1e4
        tolerance = 1e-5
        
        # If the power constraint is naturally satisfied without a multiplier
        if np.sum((a / b)**2) <= P_max:
            v = a / b
        else:
            # Find the exact multiplier that makes the sum of powers equal to P_max
            while high - low > tolerance:
                mu = (low + high) / 2
                v_temp = a / (b + mu)
                if np.sum(v_temp**2) > P_max:
                    low = mu
                else:
                    high = mu
            v = a / (b + high)
            
    return v**2

def max_min_power_allocation(H, P_max, noise_variance):
    """
    Allocates power to guarantee the exact same SINR (and rate) for all users.
    Uses bisection search over the target SINR threshold.
    """
    num_users = H.shape[0]
    
    # Power gains
    G = np.abs(H)**2
    # Diagonal matrix of direct channels
    D_inv = np.diag(1.0 / np.diag(G))
    # Interference matrix (diagonal set to 0)
    Interference_Matrix = G - np.diag(np.diag(G))
    
    low_sinr = 0.0
    high_sinr = 1e4  # Upper bound for search
    tolerance = 1e-5
    
    optimal_powers = np.zeros(num_users)
    
    while high_sinr - low_sinr > tolerance:
        target_sinr = (low_sinr + high_sinr) / 2
        
        # We solve the matrix equation: P = target_sinr * D_inv * (Interference * P + Noise)
        # (I - target_sinr * D_inv * Interference) * P = target_sinr * D_inv * Noise
        A = np.eye(num_users) - target_sinr * np.dot(D_inv, Interference_Matrix)
        b = target_sinr * np.dot(D_inv, np.ones(num_users) * noise_variance)
        
        try:
            # Try to calculate the required power vector to hit this target SINR
            p_req = np.linalg.solve(A, b)
            
            # Check feasibility: Powers must be positive and sum must be <= P_max
            if np.all(p_req > 0) and np.sum(p_req) <= P_max:
                optimal_powers = p_req
                low_sinr = target_sinr  # We can aim higher
            else:
                high_sinr = target_sinr # Target is unachievable, aim lower
        except np.linalg.LinAlgError:
            # Matrix inversion failed (target too high, system is unstable)
            high_sinr = target_sinr
            
    return optimal_powers


def fractional_programming_power_allocation(H, P_max, noise_variance, iterations=100):
    """
    Allocates power using the FP Quadratic Transform method for Sum-Rate Maximization.
    """
    num_users = H.shape[0]
    
    # Initialize powers evenly
    p = (P_max / num_users) * np.ones(num_users)
    y = np.zeros(num_users)
    
    # Power gains matrix
    G = np.abs(H)**2
    
    for _ in range(iterations):
        # Step 1: Update the auxiliary variable y (Quadratic Transform)
        for k in range(num_users):
            interference_plus_noise = sum(G[k, j] * p[j] for j in range(num_users) if j != k) + noise_variance
            # y represents the decoupled fractional state
            y[k] = np.sqrt(G[k, k] * p[k]) / (interference_plus_noise + G[k, k] * p[k])
            
        # Step 2: Update the power vector p (Solving the pseudo-convex subproblem)
        # For sum-rate, this is updated using gradient descent on the transformed Lagrangian
        learning_rate = 0.1 
        
        for k in range(num_users):
            # Gradient of the transformed objective with respect to p[k]
            signal_term = (y[k] * np.sqrt(G[k, k])) / (2 * np.sqrt(p[k] + 1e-10))
            
            # Penalty from the interference caused to ALL other users
            interference_penalty = sum((y[j]**2) * G[j, k] for j in range(num_users))
            
            gradient = signal_term - interference_penalty
            
            # Update power (gradient ascent)
            p[k] = p[k] + learning_rate * gradient
            
        # Enforce physical constraints: Power must be positive
        p = np.maximum(p, 0)
        
        # Enforce sum-power constraint P_max
        total_p = np.sum(p)
        if total_p > P_max:
            # Simple projection onto the simplex budget
            p = p * (P_max / total_p)
            
    return p

def wmmse_pf_power_allocation(H, P_max, noise_variance, dynamic_weights, iterations=20):
    """Proportional Fairness Baseline"""
    num_users = H.shape[0]
    v = np.sqrt(P_max / num_users) * np.ones(num_users)
    u, w = np.zeros(num_users), np.zeros(num_users)
    
    for _ in range(iterations):
        for k in range(num_users):
            interference = sum((np.abs(H[k, j]) * v[j])**2 for j in range(num_users)) + noise_variance
            u[k] = (np.abs(H[k, k]) * v[k]) / interference
            
        for k in range(num_users):
            w[k] = 1 / (1 - u[k] * np.abs(H[k, k]) * v[k])
            
        a, b = np.zeros(num_users), np.zeros(num_users)
        for k in range(num_users):
            a[k] = dynamic_weights[k] * w[k] * u[k] * np.abs(H[k, k])
            b[k] = sum(dynamic_weights[j] * w[j] * (u[j]**2) * (np.abs(H[j, k])**2) for j in range(num_users))
            
        b = np.maximum(b, 1e-12) 
        
        if np.sum((a / b)**2) <= P_max:
            v = a / b
        else:
            low, high = 0.0, 1e4
            for _ in range(30):
                mu = (low + high) / 2
                if np.sum((a / (b + mu))**2) > P_max:
                    low = mu
                else:
                    high = mu
            v = a / (b + high)
    return v**2