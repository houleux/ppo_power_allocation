import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.baselines import iterative_water_filling, wmmse_power_allocation_pf, fractional_programming_power_allocation, max_min_power_allocation
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


env = DynamicChannelSimulator(NUM_USERS, user_speed_kmh=60)
ema_rates = np.ones(NUM_USERS) * 1e-3

for tti in range(NUM_TTI):
    print(f"\n[{'='*15} TTI {tti + 1} {'='*15}]")

    H = env.step()

    # Test Fractional Programming power allocation
    fp_powers = fractional_programming_power_allocation(H, P_MAX, NOISE_VAR)
    fp_rates = calculate_sinr_and_rate(H, fp_powers, NOISE_VAR, BANDWIDTH)

    # Test PF-WMMSE power allocation
    pf_weights = 1.0 / (ema_rates + 1e-6)
    wmmse_powers = wmmse_power_allocation_pf(H, P_MAX, NOISE_VAR, weights=pf_weights, p_floor=P_FLOOR)
    wmmse_rates = calculate_sinr_and_rate(H, wmmse_powers, NOISE_VAR, BANDWIDTH)
    ema_rates = (1 - EMA_BETA) * ema_rates + EMA_BETA * wmmse_rates

    # Test Max-Min power allocation
    max_min_powers = max_min_power_allocation(H, P_MAX, NOISE_VAR)
    max_min_rates = calculate_sinr_and_rate(H, max_min_powers, NOISE_VAR, BANDWIDTH)


    print(f"FP Power Allocation:      {fp_powers.round(2)}")
    print(f"FP Sum-Rate:              {np.sum(fp_rates)/1e6:.2f} Mbps")
    print(f"PF rates per user:         {(wmmse_rates/1e6).round(2)} Mbps") 

    print(f"PF-WMMSE Power Allocation: {wmmse_powers.round(2)}")
    print(f"PF-WMMSE Sum-Rate: {np.sum(wmmse_rates)/1e6:.2f} Mbps")
    print(f"PF-WMMSE rates per user:         {(wmmse_rates/1e6).round(2)} Mbps")

    print(f"Max-Min Power Allocation: {max_min_powers.round(2)}")
    print(f"Max-Min Rates per user:   {(max_min_rates/1e6).round(2)} Mbps")
    print(f"Max-Min Sum-Rate:         {np.sum(max_min_rates)/1e6:.2f} Mbps")
