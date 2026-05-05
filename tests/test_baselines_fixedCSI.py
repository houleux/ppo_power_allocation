import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.baselines import iterative_water_filling, wmmse_power_allocation, fractional_programming_power_allocation, max_min_power_allocation
from src.performance import calculate_sinr_and_rate


np.random.seed(42)

NUM_USERS = 4
P_MAX = 10.0
NOISE_VARIANCE = 0.1
BANDWIDTH = 1e6


# generate random channel gains for testing
H = np.abs(np.random.randn(NUM_USERS, NUM_USERS) + 1j * np.random.randn(NUM_USERS, NUM_USERS)) / np.sqrt(2)  

# Test iterative water-filling
direct_gains = np.diag(H)**2
iwf_powers = iterative_water_filling(direct_gains, NOISE_VARIANCE, P_MAX)
iwf_rates = calculate_sinr_and_rate(H, iwf_powers, NOISE_VARIANCE, BANDWIDTH)


# Test WMMSE power allocation
wmmse_powers = wmmse_power_allocation(H, P_MAX, NOISE_VARIANCE)
wmmse_rates = calculate_sinr_and_rate(H, wmmse_powers, NOISE_VARIANCE, BANDWIDTH)

# Test Max-Min power allocation
max_min_powers = max_min_power_allocation(H, P_MAX, NOISE_VARIANCE)
max_min_rates = calculate_sinr_and_rate(H, max_min_powers, NOISE_VARIANCE, BANDWIDTH)

# Test Fractional Programming power allocation
fp_powers = fractional_programming_power_allocation(H, P_MAX, NOISE_VARIANCE)
fp_rates = calculate_sinr_and_rate(H, fp_powers, NOISE_VARIANCE, BANDWIDTH)

print("--- Simulation Results ---")
print(f"IWF Power Allocation: {iwf_powers.round(2)}")
print(f"IWF Sum-Rate: {np.sum(iwf_rates)/1e6:.2f} Mbps\n")

print(f"WMMSE Power Allocation: {wmmse_powers.round(2)}")
print(f"WMMSE Sum-Rate: {np.sum(wmmse_rates)/1e6:.2f} Mbps")

print(f"Max-Min Power Allocation: {max_min_powers.round(2)}")
print(f"Max-Min Rates per user:   {(max_min_rates/1e6).round(2)} Mbps")
print(f"Max-Min Sum-Rate:         {np.sum(max_min_rates)/1e6:.2f} Mbps\n")

print(f"FP Power Allocation:      {fp_powers.round(2)}")
print(f"FP Sum-Rate:              {np.sum(fp_rates)/1e6:.2f} Mbps")