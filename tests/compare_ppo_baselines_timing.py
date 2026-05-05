import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.baselines import fractional_programming_power_allocation, wmmse_power_allocation_pf, max_min_power_allocation
from src.performance import calculate_sinr_and_rate
from src.dynamic_channel import DynamicChannelSimulator

np.random.seed(101)

P_MAX = 10.0
NOISE_VAR = 0.1
BANDWIDTH = 1e6
NUM_TTI = 50
EMA_BETA = 0.2
P_FLOOR = 0.05
MODEL_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ppo_pf_wireless_model_user{}.zip")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")


def build_obs(H, historical_rates):
    h_mag_flat = np.abs(H).flatten()
    norm_hist_rates = historical_rates / (BANDWIDTH * 10)
    return np.concatenate([h_mag_flat, norm_hist_rates]).astype(np.float32)


def project_action_to_powers(action):
    allocated_powers = action * P_MAX
    total_power = np.sum(allocated_powers)
    if total_power > P_MAX:
        allocated_powers = allocated_powers * (P_MAX / total_power)
    return allocated_powers


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def run_for_users(num_users):
    model_path = MODEL_TEMPLATE.format(num_users)
    model = PPO.load(model_path)

    env = DynamicChannelSimulator(num_users, user_speed_kmh=60)
    ema_rates_pf = np.ones(num_users) * 1e-3
    ema_rates_ppo = np.ones(num_users) * 1e-6

    timing = {
        "FP": [],
        "PF-WMMSE": [],
        "Max-Min": [],
        "PPO": [],
    }

    for _ in range(NUM_TTI):
        H = env.step()

        _, dt = time_call(fractional_programming_power_allocation, H, P_MAX, NOISE_VAR)
        timing["FP"].append(dt)

        pf_weights = 1.0 / (ema_rates_pf + 1e-6)
        wmmse_powers, dt = time_call(wmmse_power_allocation_pf, H, P_MAX, NOISE_VAR, weights=pf_weights, p_floor=P_FLOOR)
        timing["PF-WMMSE"].append(dt)
        wmmse_rates = calculate_sinr_and_rate(H, wmmse_powers, NOISE_VAR, BANDWIDTH)
        ema_rates_pf = (1 - EMA_BETA) * ema_rates_pf + EMA_BETA * wmmse_rates

        _, dt = time_call(max_min_power_allocation, H, P_MAX, NOISE_VAR)
        timing["Max-Min"].append(dt)

        obs = build_obs(H, ema_rates_ppo)
        (action, _states), dt = time_call(model.predict, obs, deterministic=True)
        timing["PPO"].append(dt)
        action = np.asarray(action, dtype=float)
        ppo_powers = project_action_to_powers(action)
        ppo_rates = calculate_sinr_and_rate(H, ppo_powers, NOISE_VAR, BANDWIDTH)
        ema_rates_ppo = (1 - EMA_BETA) * ema_rates_ppo + EMA_BETA * ppo_rates

    return {key: np.mean(values) for key, values in timing.items()}


num_users_list = [3, 4, 5, 6]

avg_time_results = {
    "FP": [],
    "PF-WMMSE": [],
    "Max-Min": [],
    "PPO": [],
}

for num_users in num_users_list:
    metrics = run_for_users(num_users)
    for key in avg_time_results:
        avg_time_results[key].append(metrics[key] * 1e3)

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.figure(figsize=(8, 5))
for label, values in avg_time_results.items():
    plt.plot(num_users_list, values, marker="o", label=label)
plt.xlabel("Number of Users")
plt.ylabel("Average Execution Time per TTI (ms)")
plt.title("Average Execution Time vs Number of Users")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "avg_time_vs_users.png"), dpi=150)
plt.close()
