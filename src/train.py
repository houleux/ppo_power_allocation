from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import WirelessAllocationEnv

# 1. Instantiate the Environment
env = WirelessAllocationEnv(num_users=4, p_max=10.0)

# Optional but highly recommended: Verify your custom environment meets Gymnasium standards
check_env(env, warn=True)

# 2. Initialize the PPO Agent
# MlpPolicy creates a standard Multi-Layer Perceptron (feed-forward neural network)
# verbose=1 prints the training metrics to the console
model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=3e-4, 
    n_steps=2048, 
    batch_size=64, 
    gamma=0.99, # High gamma values teach the agent to care about long-term fairness
    verbose=1
)

# 3. Train the Agent
print("--- Starting PPO Training ---")
# 100,000 timesteps is a good starting point for a 4-user continuous space.
# For a production model or research paper, this might be 1M - 5M timesteps.
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_pf_wireless_model")
print("Model Saved Successfully.")

# Load the trained model
trained_model = PPO.load("ppo_pf_wireless_model")

# Reset the environment for evaluation
obs, info = env.reset()

print("\n--- Evaluating Trained PPO Agent ---")
for tti in range(5):
    # The neural network performs a rapid forward-pass to determine the action
    # deterministic=True removes exploratory noise for optimal evaluation
    action, _states = trained_model.predict(obs, deterministic=True)
    
    # Execute the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"TTI {tti + 1}:")
    print(f"  PPO Action (0-1):  {action.round(3)}")
    print(f"  Actual Power (W):  {info['allocated_powers'].round(2)}")
    print(f"  Achieved Rates:    {(info['instant_rates'] / 1e6).round(2)} Mbps")
    print(f"  Total Sum-Rate:    {info['sum_rate_mbps']:.2f} Mbps\n")