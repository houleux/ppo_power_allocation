import os
import sys

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


def jains_fairness(rates, eps=1e-12):
    numerator = np.sum(rates) ** 2
    denominator = len(rates) * np.sum(rates ** 2) + eps
    return numerator / denominator


def run_for_users(num_users):
    model_path = MODEL_TEMPLATE.format(num_users)
    model = PPO.load(model_path)

    env = DynamicChannelSimulator(num_users, user_speed_kmh=60)
    ema_rates_pf = np.ones(num_users) * 1e-3
    ema_rates_ppo = np.ones(num_users) * 1e-6

    fp_sum_rates = []
    pf_wmmse_sum_rates = []
    max_min_sum_rates = []
    ppo_sum_rates = []

    fp_fairness = []
    pf_wmmse_fairness = []
    max_min_fairness = []
    ppo_fairness = []

    for _ in range(NUM_TTI):
        H = env.step()

        fp_powers = fractional_programming_power_allocation(H, P_MAX, NOISE_VAR)
        fp_rates = calculate_sinr_and_rate(H, fp_powers, NOISE_VAR, BANDWIDTH)
        fp_sum_rates.append(np.sum(fp_rates))
        fp_fairness.append(jains_fairness(fp_rates))

        pf_weights = 1.0 / (ema_rates_pf + 1e-6)
        wmmse_powers = wmmse_power_allocation_pf(H, P_MAX, NOISE_VAR, weights=pf_weights, p_floor=P_FLOOR)
        wmmse_rates = calculate_sinr_and_rate(H, wmmse_powers, NOISE_VAR, BANDWIDTH)
        pf_wmmse_sum_rates.append(np.sum(wmmse_rates))
        pf_wmmse_fairness.append(jains_fairness(wmmse_rates))
        ema_rates_pf = (1 - EMA_BETA) * ema_rates_pf + EMA_BETA * wmmse_rates

        max_min_powers = max_min_power_allocation(H, P_MAX, NOISE_VAR)
        max_min_rates = calculate_sinr_and_rate(H, max_min_powers, NOISE_VAR, BANDWIDTH)
        max_min_sum_rates.append(np.sum(max_min_rates))
        max_min_fairness.append(jains_fairness(max_min_rates))

        obs = build_obs(H, ema_rates_ppo)
        action, _states = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float)
        ppo_powers = project_action_to_powers(action)
        ppo_rates = calculate_sinr_and_rate(H, ppo_powers, NOISE_VAR, BANDWIDTH)
        ppo_sum_rates.append(np.sum(ppo_rates))
        ppo_fairness.append(jains_fairness(ppo_rates))
        ema_rates_ppo = (1 - EMA_BETA) * ema_rates_ppo + EMA_BETA * ppo_rates

    return {
        "fp_sum_rate": np.mean(fp_sum_rates),
        "pf_wmmse_sum_rate": np.mean(pf_wmmse_sum_rates),
        "max_min_sum_rate": np.mean(max_min_sum_rates),
        "ppo_sum_rate": np.mean(ppo_sum_rates),
        "fp_fairness": np.mean(fp_fairness),
        "pf_wmmse_fairness": np.mean(pf_wmmse_fairness),
        "max_min_fairness": np.mean(max_min_fairness),
        "ppo_fairness": np.mean(ppo_fairness),
    }


num_users_list = [3, 4, 5, 6]

sum_rate_results = {
    "FP": [],
    "PF-WMMSE": [],
    "Max-Min": [],
    "PPO": [],
}

fairness_results = {
    "FP": [],
    "PF-WMMSE": [],
    "Max-Min": [],
    "PPO": [],
}

for num_users in num_users_list:
    metrics = run_for_users(num_users)

    sum_rate_results["FP"].append(metrics["fp_sum_rate"] / 1e6)
    sum_rate_results["PF-WMMSE"].append(metrics["pf_wmmse_sum_rate"] / 1e6)
    sum_rate_results["Max-Min"].append(metrics["max_min_sum_rate"] / 1e6)
    sum_rate_results["PPO"].append(metrics["ppo_sum_rate"] / 1e6)

    fairness_results["FP"].append(metrics["fp_fairness"])
    fairness_results["PF-WMMSE"].append(metrics["pf_wmmse_fairness"])
    fairness_results["Max-Min"].append(metrics["max_min_fairness"])
    fairness_results["PPO"].append(metrics["ppo_fairness"])

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.figure(figsize=(8, 5))
for label, values in sum_rate_results.items():
    plt.plot(num_users_list, values, marker="o", label=label)
plt.xlabel("Number of Users")
plt.ylabel("Average Sum-Rate (Mbps)")
plt.title("Average Sum-Rate vs Number of Users")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sum_rate_vs_users.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
for label, values in fairness_results.items():
    plt.plot(num_users_list, values, marker="o", label=label)
plt.xlabel("Number of Users")
plt.ylabel("Jain's Fairness Index")
plt.title("Jain's Fairness vs Number of Users")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "jain_fairness_vs_users.png"), dpi=150)
plt.close()
