# Proportional-Fair WMMSE Baseline

## 1. Overview
This report documents the new proportional-fair WMMSE (PF-WMMSE) baseline used for power allocation in the multi-user interference network. The goal is to reduce user starvation observed in the pure sum-rate WMMSE by introducing fairness weights derived from long-term average rates.

## 2. Algorithm Summary
PF-WMMSE solves a weighted sum-rate problem by iteratively updating receiver filters, MSE weights, and transmitter powers. The fairness weights are based on an exponential moving average (EMA) of each user rate.

Key updates:
- **Complex-valued receiver update** to correctly handle complex channels.
- **Weighted MSE update** using proportional-fair weights.
- **Optional power floor** to prevent complete starvation.

The MSE for user $k$ is computed as:
$$
 e_k = 1 - 2\Re\{u_k h_{kk} v_k\} + |u_k|^2\left(\sum_j |h_{kj}|^2 v_j^2 + \sigma^2\right)
$$
with weights $w_k = \alpha_k / \max(e_k, \epsilon)$ and $
\alpha_k = 1/(\bar{R}_k + \epsilon)$.

## 3. Implementation Details
File: [src/baselines.py](src/baselines.py)

New function:
- `wmmse_power_allocation_pf(H, P_max, noise_variance, weights=None, iterations=100, p_floor=0.0)`

Features:
- Uses complex `u` and the full MSE expression.
- Accepts external weights (proportional-fair by default in the test).
- Includes a configurable `p_floor` to avoid zero allocations.

## 4. Test Updates
File: [tests/test_baselines_dynamicCSI.py](tests/test_baselines_dynamicCSI.py)

The dynamic CSI test now:
- Tracks EMA rates per user.
- Converts EMA rates to proportional-fair weights.
- Calls PF-WMMSE with a small power floor.

## 5. Suggested Parameters
- `EMA_BETA = 0.1 to 0.3` for smoother or faster fairness adaptation.
- `p_floor` set to a small fraction of $P_{max}$ to avoid zero power.
- `iterations = 50 to 100` for convergence under dynamic channels.

## 6. Expected Behavior
PF-WMMSE should reduce power starvation while keeping a good sum-rate. It is not as strict as max-min fairness, but should provide more balanced allocations than sum-rate WMMSE in most TTIs.
