import gymnasium as gym

# Register the environment
gym.register(
    id='WirelessAllocation-v0',
    entry_point='gym_env:WirelessAllocationEnv',
    kwargs={'num_users': 16, 'p_max': 10.0}
)
