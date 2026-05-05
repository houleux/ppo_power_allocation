import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import j0

class WirelessAllocationEnv(gym.Env):
    """
    Custom Gymnasium Environment for Multi-User Power Allocation.
    Enforces Proportional Fairness using a Logarithmic Reward.
    """
    def __init__(self, num_users=4, p_max=10.0, noise_var=0.1, bandwidth=1e6):
        super(WirelessAllocationEnv, self).__init__()
        
        self.num_users = num_users
        self.p_max = p_max
        self.noise_var = noise_var
        self.bandwidth = bandwidth
        
        # Physics Parameters for Dynamic CSI (60 km/h)
        speed_ms = 60 * (1000 / 3600)
        fd = speed_ms / (3e8 / 2e9) 
        self.rho = j0(2 * np.pi * fd * 1e-3) # 1ms TTI
        
        # --- Action Space ---
        # The agent outputs a value between 0 and 1 for each user.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_users,), dtype=np.float32)
        
        # --- Observation Space ---
        # State includes: Flattened Channel Matrix Magnitudes (N*N) + Historical Rates (N)
        obs_size = (self.num_users * self.num_users) + self.num_users
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.reset()

    def _generate_rayleigh(self):
        return (np.random.randn(self.num_users, self.num_users) + 
                1j * np.random.randn(self.num_users, self.num_users)) / np.sqrt(2)

    def _get_obs(self):
        # Flatten the magnitude of the channel matrix
        h_mag_flat = np.abs(self.H_current).flatten()
        # Normalize historical rates to keep neural network inputs stable
        norm_hist_rates = self.historical_rates / (self.bandwidth * 10) 
        return np.concatenate([h_mag_flat, norm_hist_rates]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.H_current = self._generate_rayleigh()
        self.historical_rates = np.ones(self.num_users) * 1e-6
        self.current_step = 0
        self.max_steps = 100 # An episode consists of 100 TTIs
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Scale Action to Power Constraints
        # Agent outputs [0, 1]. We scale to P_max.
        allocated_powers = action * self.p_max
        
        # Action Penalty: Teach the network to natively respect the sum-power constraint
        total_power = np.sum(allocated_powers)
        penalty = 0.0
        if total_power > self.p_max:
            # If it exceeds hardware limits, penalize and physically project it back
            penalty = -5.0 * (total_power - self.p_max)
            allocated_powers = allocated_powers * (self.p_max / total_power)

        # 2. Calculate SINR and Instantaneous Rates
        instant_rates = np.zeros(self.num_users)
        for k in range(self.num_users):
            signal = (np.abs(self.H_current[k, k])**2) * allocated_powers[k]
            interference = sum((np.abs(self.H_current[k, j])**2) * allocated_powers[j] for j in range(self.num_users) if j != k)
            sinr = signal / (interference + self.noise_var)
            instant_rates[k] = self.bandwidth * np.log2(1 + sinr)
            
        # 3. Update Historical Rates (EMA)
        self.historical_rates = 0.8 * self.historical_rates + 0.2 * instant_rates
        
        # 4. Calculate Proportional Fairness Reward
        # Logarithmic utility mathematically penalizes total data deprivation
        # We add 1e-6 to prevent log(0) if a user is completely starved
        pf_reward = np.sum(np.log(instant_rates + 1e-6)) 
        
        total_reward = pf_reward + penalty
        
        # 5. Step the Physical Environment (Gauss-Markov)
        innovation = self._generate_rayleigh()
        self.H_current = self.rho * self.H_current + np.sqrt(1 - self.rho**2) * innovation
        
        # 6. Check Episode Termination
        terminated = False
        truncated = bool(self.current_step >= self.max_steps)
        
        info = {
            "allocated_powers": allocated_powers,
            "instant_rates": instant_rates,
            "sum_rate_mbps": np.sum(instant_rates) / 1e6
        }
        
        return self._get_obs(), total_reward, terminated, truncated, info