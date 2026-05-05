import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.baselines import fractional_programming_power_allocation, wmmse_power_allocation_pf, max_min_power_allocation
from src.performance import calculate_sinr_and_rate
from src.dynamic_channel import DynamicChannelSimulator

np.random.seed(101)

NUM_USERS = 4
P_MAX = 10.0
NOISE_VAR = 0.1
BANDWIDTH = 1e6
NUM_TTI = 10
EMA_BETA = 0.2
P_FLOOR = 0.05
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ppo_pf_wireless_model.zip")


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


model = PPO.load(MODEL_PATH)

env = DynamicChannelSimulator(NUM_USERS, user_speed_kmh=60)
ema_rates_pf = np.ones(NUM_USERS) * 1e-3
ema_rates_ppo = np.ones(NUM_USERS) * 1e-6
fp_sum_rates = []
pf_wmmse_sum_rates = []
max_min_sum_rates = []
ppo_sum_rates = []

for tti in range(NUM_TTI):
    print(f"\n[{'='*15} TTI {tti + 1} {'='*15}]")

    H = env.step()

    # Fractional Programming
    fp_powers = fractional_programming_power_allocation(H, P_MAX, NOISE_VAR)
    fp_rates = calculate_sinr_and_rate(H, fp_powers, NOISE_VAR, BANDWIDTH)
    fp_sum_rates.append(np.sum(fp_rates))

    # PF-WMMSE
    pf_weights = 1.0 / (ema_rates_pf + 1e-6)
    wmmse_powers = wmmse_power_allocation_pf(H, P_MAX, NOISE_VAR, weights=pf_weights, p_floor=P_FLOOR)
    wmmse_rates = calculate_sinr_and_rate(H, wmmse_powers, NOISE_VAR, BANDWIDTH)
    pf_wmmse_sum_rates.append(np.sum(wmmse_rates))
    ema_rates_pf = (1 - EMA_BETA) * ema_rates_pf + EMA_BETA * wmmse_rates

    # Max-Min
    max_min_powers = max_min_power_allocation(H, P_MAX, NOISE_VAR)
    max_min_rates = calculate_sinr_and_rate(H, max_min_powers, NOISE_VAR, BANDWIDTH)
    max_min_sum_rates.append(np.sum(max_min_rates))

    # PPO
    obs = build_obs(H, ema_rates_ppo)
    action, _states = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=float)
    ppo_powers = project_action_to_powers(action)
    ppo_rates = calculate_sinr_and_rate(H, ppo_powers, NOISE_VAR, BANDWIDTH)
    ppo_sum_rates.append(np.sum(ppo_rates))
    ema_rates_ppo = (1 - EMA_BETA) * ema_rates_ppo + EMA_BETA * ppo_rates

    print(f"FP Power Allocation:       {fp_powers.round(2)}")
    print(f"FP Sum-Rate:               {np.sum(fp_rates)/1e6:.2f} Mbps")

    print(f"PF-WMMSE Power Allocation: {wmmse_powers.round(2)}")
    print(f"PF-WMMSE Sum-Rate:          {np.sum(wmmse_rates)/1e6:.2f} Mbps")
    print(f"PF-WMMSE Rates per user:    {(wmmse_rates/1e6).round(2)} Mbps")

    print(f"Max-Min Power Allocation:  {max_min_powers.round(2)}")
    print(f"Max-Min Rates per user:    {(max_min_rates/1e6).round(2)} Mbps")
    print(f"Max-Min Sum-Rate:          {np.sum(max_min_rates)/1e6:.2f} Mbps")

    # print(f"PPO Action (0-1):           {action.round(3)}")
    print(f"PPO Power Allocation:      {ppo_powers.round(2)}")
    print(f"PPO Rates per user:        {(ppo_rates/1e6).round(2)} Mbps")
    print(f"PPO Sum-Rate:              {np.sum(ppo_rates)/1e6:.2f} Mbps")

print("\n[=============== Average Sum-Rate over TTIs ===============]")
print(f"FP Average Sum-Rate:       {np.mean(fp_sum_rates)/1e6:.2f} Mbps")
print(f"PF-WMMSE Average Sum-Rate: {np.mean(pf_wmmse_sum_rates)/1e6:.2f} Mbps")
print(f"Max-Min Average Sum-Rate:  {np.mean(max_min_sum_rates)/1e6:.2f} Mbps")
print(f"PPO Average Sum-Rate:      {np.mean(ppo_sum_rates)/1e6:.2f} Mbps")
