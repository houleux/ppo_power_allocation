# MAML-PPO for User-Count Adaptable Power Allocation

## 1. Overview
The original PPO implementation trains a separate policy for a fixed number of wireless users. This works for a known network size, but it does not directly adapt when the number of users changes because both the observation dimension and action dimension depend on `num_users`.

The new implementation in [src/maml_ppo.py](../src/maml_ppo.py) adds a meta-learning layer using Model-Agnostic Meta-Learning (MAML). The goal is to train a single initialization of an actor-critic policy that can quickly adapt to different numbers of users, such as 3, 4, 5, and 6 users, using only a small amount of task-specific rollout data.

The high-level idea is:
- Each user count is treated as a separate task.
- The policy is trained to adapt quickly from a shared initialization.
- During deployment, the model can perform a few gradient updates on rollouts from the current user count before inference.

## 2. Motivation
In the existing Stable-Baselines3 PPO setup, the policy input is:

$$
|H| \in \mathbb{R}^{N \times N}, \quad \bar{R} \in \mathbb{R}^{N}
$$

so the flattened observation has size:

$$
N^2 + N
$$

The action vector also has size:

$$
N
$$

This means a 3-user PPO model and a 6-user PPO model have different neural network input/output sizes and cannot share weights directly. MAML requires a common model architecture across tasks, so the implementation introduces a padded and masked representation.

## 3. Fixed-Size Padded Task Representation
The MAML implementation defines a maximum supported number of users, `max_users`. For example, when training on 3 to 6 users:

```bash
--user-counts 3 4 5 6 --max-users 6
```

All tasks are represented using the same fixed dimensions:

$$
\text{obs_dim} = max\_users^2 + 2 \cdot max\_users
$$

The observation contains:
- Padded channel magnitudes: `max_users x max_users`
- Padded historical average rates: `max_users`
- Binary active-user mask: `max_users`

For a task with `K < max_users`, only the first `K` users are active. The remaining users are padded with zeros. The mask is used to ignore inactive actions and inactive log-probability terms during PPO loss computation.

Example for `K = 4`, `max_users = 6`:

```text
mask = [1, 1, 1, 1, 0, 0]
```

This design allows a single actor-critic network to support multiple user counts.

## 4. Wireless Task Environment
The class `PaddedWirelessAllocationTask` implements the wireless environment used by MAML. It follows the same core logic as `WirelessAllocationEnv`:

- Rayleigh fading channel generation.
- Dynamic CSI evolution using a first-order Gauss-Markov process.
- Continuous power-allocation actions in `[0, 1]`.
- Scaling actions by `P_max`.
- Projection back to the sum-power constraint when total power exceeds `P_max`.
- SINR and Shannon rate calculation.
- Proportional-fair reward using logarithmic utility.

The reward is:

$$
r_t = \sum_{k=1}^{K} \log(R_k(t) + \epsilon) + penalty
$$

where `penalty` is negative if the action exceeds the total power budget before projection.

## 5. Policy Architecture
The implementation uses a custom PyTorch actor-critic model named `MAMLActorCritic`.

The network contains:
- Shared MLP backbone.
- Actor head.
- Critic value head.

The actor outputs parameters of a Beta distribution:

$$
a_k \sim Beta(\alpha_k, \beta_k)
$$

This is useful because the Beta distribution naturally produces actions in `[0, 1]`, matching the normalized power-allocation action space. The deterministic action used during evaluation is the mean:

$$
a_k = \frac{\alpha_k}{\alpha_k + \beta_k}
$$

Inactive users are masked out after sampling.

## 6. PPO Loss Used Inside MAML
The implementation uses a PPO-style clipped objective for the inner and outer losses. For each rollout, the policy computes:

- Log probability of selected actions.
- Entropy bonus.
- Value estimate.
- Generalized Advantage Estimation (GAE).

The clipped PPO policy loss is:

$$
L^{PPO} = -\mathbb{E}\left[
\min(r_t A_t, clip(r_t, 1-\epsilon, 1+\epsilon)A_t)
\right]
$$

where:

$$
r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

The full loss is:

$$
L = L^{PPO} + c_v L^{value} - c_e H(\pi)
$$

where:
- `L_value` is the critic mean-squared error.
- `H(pi)` is the entropy bonus.
- `c_v` is `vf_coef`.
- `c_e` is `ent_coef`.

## 7. MAML Algorithm
MAML trains a model initialization rather than only a final task-specific policy. The learned parameters should be close to good solutions for all training tasks, so one or a few gradient steps can adapt the policy to a new task.

In this project, each task is a user count:

$$
\mathcal{T}_K, \quad K \in \{3,4,5,6\}
$$

For every meta-iteration:

1. Sample a batch of user-count tasks.
2. For each task, collect a support rollout.
3. Compute the PPO support loss.
4. Perform one or more inner gradient updates:

$$
\theta'_K = \theta - \alpha \nabla_\theta L_{\mathcal{T}_K}^{support}(\theta)
$$

5. Collect a query rollout using the adapted parameters `theta'_K`.
6. Compute the query PPO loss.
7. Update the original initialization `theta` using the average query loss:

$$
\theta \leftarrow \theta - \beta \nabla_\theta
\sum_K L_{\mathcal{T}_K}^{query}(\theta'_K)
$$

Here:
- `alpha` is `inner_lr`.
- `beta` is `meta_lr`.
- Support rollouts are used for adaptation.
- Query rollouts are used for meta-optimization.

The script supports full MAML by default. It also supports first-order MAML using `--first-order`, which avoids second-order gradients and is usually faster with lower memory cost.

## 8. Code Structure
Main file:

```text
src/maml_ppo.py
```

Important components:

- `WirelessTaskConfig`: stores wireless task configuration.
- `PaddedWirelessAllocationTask`: padded wireless environment for variable user counts.
- `MAMLActorCritic`: PyTorch actor-critic policy.
- `collect_rollout`: gathers support/query trajectories.
- `add_gae_returns`: computes GAE advantages and returns.
- `ppo_loss`: PPO clipped actor-critic objective.
- `maml_adapt`: performs the inner MAML update.
- `train_maml`: runs the meta-training loop.
- `evaluate_after_adaptation`: adapts and evaluates the policy for a given user count.
- `load_checkpoint`: loads a saved MAML checkpoint.
- `predict_power_allocation`: helper for using the trained model on a channel matrix.

## 9. Training Command
Basic training across 3 to 6 users:

```bash
python3 src/maml_ppo.py \
  --user-counts 3 4 5 6 \
  --max-users 6 \
  --meta-iters 200 \
  --output maml_ppo_wireless.pt
```

Lower-memory first-order training:

```bash
python3 src/maml_ppo.py \
  --user-counts 3 4 5 6 \
  --max-users 6 \
  --meta-iters 200 \
  --first-order \
  --output maml_ppo_wireless.pt
```

Useful larger-run settings:

```bash
python3 src/maml_ppo.py \
  --user-counts 3 4 5 6 \
  --max-users 6 \
  --meta-iters 1000 \
  --meta-batch-size 4 \
  --support-steps 128 \
  --query-steps 128 \
  --inner-steps 1 \
  --output maml_ppo_wireless.pt
```

## 10. Inference Usage
After training, a checkpoint can be loaded and used to predict powers for a channel matrix:

```python
import numpy as np

from src.maml_ppo import load_checkpoint, predict_power_allocation

model, args = load_checkpoint("maml_ppo_wireless.pt")

H = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
historical_rates = np.ones(4) * 1e-6

powers, raw_action = predict_power_allocation(
    model,
    H,
    historical_rates,
    max_users=args["max_users"],
    p_max=args["p_max"],
    bandwidth=args["bandwidth"],
)
```

The returned `powers` vector has length equal to the active number of users in `H`, not `max_users`.

## 11. Difference From the Existing PPO Code
The existing [src/train.py](../src/train.py) uses Stable-Baselines3 PPO. That is appropriate for training one policy for one fixed user count.

The MAML implementation does not directly wrap the SB3 `PPO` object because MAML requires differentiable inner-loop parameter updates. SB3 hides the training internals inside its algorithm class, while MAML needs functional access to the actor-critic parameters before and after adaptation.

Therefore, the new implementation reimplements the needed PPO pieces in PyTorch:

- Action distribution.
- Log-probability evaluation.
- Advantage estimation.
- PPO clipped loss.
- Value loss.
- Entropy regularization.
- Inner-loop gradient update.
- Meta-gradient update.

This keeps the algorithm explicit and makes the MAML adaptation step possible.

## 12. Expected Behavior
The MAML policy is expected to learn an initialization that adapts faster than training a fresh PPO model for each user count. Its main benefit is adaptability, not necessarily immediate superiority over a fully converged PPO model trained from scratch for a single fixed user count.

Expected outcomes:

- Better transfer across user counts than independent fixed-size PPO models.
- Faster adaptation when the number of users changes.
- One checkpoint supporting all user counts up to `max_users`.
- Small loss of specialization compared with a fully trained single-user-count PPO policy may occur.

## 13. Practical Notes
Recommended starting settings:

- Use `--first-order` if memory usage is high.
- Increase `--meta-iters` for meaningful performance.
- Increase `--support-steps` and `--query-steps` for more stable PPO gradients.
- Keep `max_users` only as large as needed, because observation size grows as `max_users^2`.
- Compare against fixed-user PPO and PF-WMMSE using sum-rate, Jain's fairness, and runtime.

The smoke tests performed during implementation used very small rollout lengths only to verify correctness of shapes, gradients, checkpointing, and evaluation flow. Full research results require longer training runs.
