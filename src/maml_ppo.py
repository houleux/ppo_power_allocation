import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import j0
from torch.distributions import Beta

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - for older PyTorch versions
    from torch.nn.utils.stateless import functional_call


TensorDict = OrderedDict[str, torch.Tensor]


@dataclass
class WirelessTaskConfig:
    num_users: int
    max_users: int
    p_max: float = 10.0
    noise_var: float = 0.1
    bandwidth: float = 1e6
    ema_beta: float = 0.2
    max_steps: int = 100
    carrier_freq_hz: float = 2e9
    user_speed_kmh: float = 60.0
    tti_duration_s: float = 1e-3


class PaddedWirelessAllocationTask:
    """
    Wireless power-allocation task with fixed-size padded observations/actions.

    MAML needs one policy shape across tasks. The active task may have K users,
    while the policy always observes/actions max_users. Inactive users are
    padded with zeros and excluded by a binary mask.
    """

    def __init__(self, config: WirelessTaskConfig, seed: Optional[int] = None):
        if config.num_users > config.max_users:
            raise ValueError("num_users cannot exceed max_users")
        self.config = config
        self.rng = np.random.default_rng(seed)

        speed_ms = config.user_speed_kmh * (1000 / 3600)
        wavelength = 3e8 / config.carrier_freq_hz
        fd = speed_ms / wavelength
        self.rho = j0(2 * np.pi * fd * config.tti_duration_s)

        self.H_current: np.ndarray
        self.historical_rates: np.ndarray
        self.current_step = 0
        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.config.max_users * self.config.max_users + 2 * self.config.max_users

    @property
    def action_dim(self) -> int:
        return self.config.max_users

    def _generate_rayleigh(self) -> np.ndarray:
        k = self.config.num_users
        real = self.rng.standard_normal((k, k))
        imag = self.rng.standard_normal((k, k))
        return (real + 1j * imag) / np.sqrt(2)

    def _active_mask(self) -> np.ndarray:
        mask = np.zeros(self.config.max_users, dtype=np.float32)
        mask[: self.config.num_users] = 1.0
        return mask

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        padded_h = np.zeros((cfg.max_users, cfg.max_users), dtype=np.float32)
        padded_h[: cfg.num_users, : cfg.num_users] = np.abs(self.H_current).astype(np.float32)

        padded_rates = np.zeros(cfg.max_users, dtype=np.float32)
        padded_rates[: cfg.num_users] = (
            self.historical_rates / (cfg.bandwidth * 10.0)
        ).astype(np.float32)

        return np.concatenate(
            [padded_h.reshape(-1), padded_rates, self._active_mask()]
        ).astype(np.float32)

    def reset(self) -> np.ndarray:
        cfg = self.config
        self.H_current = self._generate_rayleigh()
        self.historical_rates = np.ones(cfg.num_users) * 1e-6
        self.current_step = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        cfg = self.config
        self.current_step += 1

        active_action = np.clip(np.asarray(action[: cfg.num_users], dtype=np.float64), 0.0, 1.0)
        allocated_powers = active_action * cfg.p_max

        total_power = float(np.sum(allocated_powers))
        penalty = 0.0
        if total_power > cfg.p_max:
            penalty = -5.0 * (total_power - cfg.p_max)
            allocated_powers = allocated_powers * (cfg.p_max / total_power)

        instant_rates = np.zeros(cfg.num_users, dtype=np.float64)
        abs_h_sq = np.abs(self.H_current) ** 2
        for k in range(cfg.num_users):
            signal = abs_h_sq[k, k] * allocated_powers[k]
            interference = sum(
                abs_h_sq[k, j] * allocated_powers[j]
                for j in range(cfg.num_users)
                if j != k
            )
            sinr = signal / (interference + cfg.noise_var)
            instant_rates[k] = cfg.bandwidth * np.log2(1.0 + sinr)

        self.historical_rates = (
            (1.0 - cfg.ema_beta) * self.historical_rates + cfg.ema_beta * instant_rates
        )

        pf_reward = float(np.sum(np.log(instant_rates + 1e-6)))
        total_reward = pf_reward + penalty

        innovation = self._generate_rayleigh()
        self.H_current = self.rho * self.H_current + np.sqrt(1.0 - self.rho**2) * innovation

        done = self.current_step >= cfg.max_steps
        info = {
            "allocated_powers": allocated_powers,
            "instant_rates": instant_rates,
            "sum_rate_mbps": np.array(np.sum(instant_rates) / 1e6),
            "num_users": np.array(cfg.num_users),
        }
        return self._get_obs(), total_reward, done, info


class MAMLActorCritic(nn.Module):
    def __init__(self, obs_dim: int, max_users: int, hidden_dim: int = 128):
        super().__init__()
        self.max_users = max_users
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.alpha_beta_head = nn.Linear(hidden_dim, 2 * max_users)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        raw_alpha_beta = self.alpha_beta_head(features)
        raw_alpha, raw_beta = torch.chunk(raw_alpha_beta, 2, dim=-1)

        # Beta(alpha, beta) is bounded on [0, 1], matching the power-allocation action.
        alpha = F.softplus(raw_alpha) + 1.0
        beta = F.softplus(raw_beta) + 1.0
        value = self.value_head(features).squeeze(-1)
        return alpha, beta, value


def module_forward(
    model: MAMLActorCritic,
    obs: torch.Tensor,
    params: Optional[TensorDict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if params is None:
        return model(obs)
    return functional_call(model, params, (obs,))


def evaluate_actions(
    model: MAMLActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    params: Optional[TensorDict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha, beta, values = module_forward(model, obs, params)
    dist = Beta(alpha, beta)

    safe_actions = torch.where(
        masks > 0.0,
        actions.clamp(1e-6, 1.0 - 1e-6),
        torch.full_like(actions, 0.5),
    )
    log_probs = (dist.log_prob(safe_actions) * masks).sum(dim=-1)
    entropies = (dist.entropy() * masks).sum(dim=-1)
    return log_probs, entropies, values


def build_padded_observation(
    H: np.ndarray,
    historical_rates: np.ndarray,
    max_users: int,
    bandwidth: float = 1e6,
) -> np.ndarray:
    num_users = H.shape[0]
    if H.shape != (num_users, num_users):
        raise ValueError("H must be a square channel matrix")
    if num_users > max_users:
        raise ValueError("H has more users than max_users")
    if historical_rates.shape[0] != num_users:
        raise ValueError("historical_rates length must match H.shape[0]")

    padded_h = np.zeros((max_users, max_users), dtype=np.float32)
    padded_h[:num_users, :num_users] = np.abs(H).astype(np.float32)

    padded_rates = np.zeros(max_users, dtype=np.float32)
    padded_rates[:num_users] = (historical_rates / (bandwidth * 10.0)).astype(np.float32)

    mask = np.zeros(max_users, dtype=np.float32)
    mask[:num_users] = 1.0
    return np.concatenate([padded_h.reshape(-1), padded_rates, mask]).astype(np.float32)


def project_action_to_powers(
    action: np.ndarray,
    num_users: int,
    p_max: float = 10.0,
) -> np.ndarray:
    allocated_powers = np.clip(action[:num_users], 0.0, 1.0) * p_max
    total_power = np.sum(allocated_powers)
    if total_power > p_max:
        allocated_powers = allocated_powers * (p_max / total_power)
    return allocated_powers


@torch.no_grad()
def predict_power_allocation(
    model: MAMLActorCritic,
    H: np.ndarray,
    historical_rates: np.ndarray,
    max_users: int,
    p_max: float = 10.0,
    bandwidth: float = 1e6,
    device: str = "cpu",
    params: Optional[TensorDict] = None,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = build_padded_observation(H, historical_rates, max_users, bandwidth)
    mask = np.zeros(max_users, dtype=np.float32)
    mask[: H.shape[0]] = 1.0
    action, _log_prob, _value = select_action(
        model,
        obs,
        mask,
        torch.device(device),
        params=params,
        deterministic=deterministic,
    )
    powers = project_action_to_powers(action, H.shape[0], p_max)
    return powers, action


@torch.no_grad()
def select_action(
    model: MAMLActorCritic,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    params: Optional[TensorDict] = None,
    deterministic: bool = False,
) -> Tuple[np.ndarray, float, float]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    alpha, beta, value = module_forward(model, obs_t, params)
    dist = Beta(alpha, beta)

    if deterministic:
        action_t = alpha / (alpha + beta)
    else:
        action_t = dist.sample()
    action_t = action_t * mask_t

    safe_actions = torch.where(
        mask_t > 0.0,
        action_t.clamp(1e-6, 1.0 - 1e-6),
        torch.full_like(action_t, 0.5),
    )
    log_prob = (dist.log_prob(safe_actions) * mask_t).sum(dim=-1)
    return (
        action_t.squeeze(0).cpu().numpy(),
        float(log_prob.item()),
        float(value.squeeze(0).item()),
    )


def collect_rollout(
    task: PaddedWirelessAllocationTask,
    model: MAMLActorCritic,
    n_steps: int,
    device: torch.device,
    params: Optional[TensorDict] = None,
) -> Dict[str, torch.Tensor]:
    obs = task.reset()
    mask = task._active_mask()
    obs_buf: List[np.ndarray] = []
    action_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    reward_buf: List[float] = []
    done_buf: List[float] = []
    value_buf: List[float] = []
    log_prob_buf: List[float] = []

    for _ in range(n_steps):
        action, log_prob, value = select_action(model, obs, mask, device, params=params)
        next_obs, reward, done, _info = task.step(action)

        obs_buf.append(obs)
        action_buf.append(action)
        mask_buf.append(mask)
        reward_buf.append(reward)
        done_buf.append(float(done))
        value_buf.append(value)
        log_prob_buf.append(log_prob)

        obs = task.reset() if done else next_obs
        mask = task._active_mask()

    if done_buf[-1] > 0.0:
        last_value = 0.0
    else:
        _action, _log_prob, last_value = select_action(
            model, obs, mask, device, params=params, deterministic=True
        )

    rollout = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(action_buf), dtype=torch.float32, device=device),
        "masks": torch.as_tensor(np.asarray(mask_buf), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(reward_buf, dtype=torch.float32, device=device),
        "dones": torch.as_tensor(done_buf, dtype=torch.float32, device=device),
        "values": torch.as_tensor(value_buf, dtype=torch.float32, device=device),
        "old_log_probs": torch.as_tensor(log_prob_buf, dtype=torch.float32, device=device),
        "last_value": torch.as_tensor(last_value, dtype=torch.float32, device=device),
    }
    return rollout


def add_gae_returns(
    rollout: Dict[str, torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> Dict[str, torch.Tensor]:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    next_value = rollout["last_value"]

    for t in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    rollout["advantages"] = advantages.detach()
    rollout["returns"] = returns.detach()
    return rollout


def ppo_loss(
    model: MAMLActorCritic,
    rollout: Dict[str, torch.Tensor],
    params: Optional[TensorDict],
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_probs, entropies, values = evaluate_actions(
        model,
        rollout["obs"],
        rollout["actions"],
        rollout["masks"],
        params=params,
    )
    ratio = torch.exp(log_probs - rollout["old_log_probs"])
    unclipped = ratio * rollout["advantages"]
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * rollout["advantages"]

    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, rollout["returns"])
    entropy = entropies.mean()
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "mean_reward": float(rollout["rewards"].mean().detach().cpu().item()),
    }
    return total_loss, metrics


def clone_parameters(model: MAMLActorCritic) -> TensorDict:
    return OrderedDict((name, param) for name, param in model.named_parameters())


def maml_adapt(
    model: MAMLActorCritic,
    params: TensorDict,
    support_rollout: Dict[str, torch.Tensor],
    inner_lr: float,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    first_order: bool,
) -> Tuple[TensorDict, Dict[str, float]]:
    inner_loss, inner_metrics = ppo_loss(
        model,
        support_rollout,
        params=params,
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
    )
    grads = torch.autograd.grad(
        inner_loss,
        tuple(params.values()),
        create_graph=not first_order,
        retain_graph=not first_order,
    )
    adapted = OrderedDict(
        (name, param - inner_lr * grad)
        for (name, param), grad in zip(params.items(), grads)
    )
    return adapted, inner_metrics


def make_task(
    num_users: int,
    max_users: int,
    p_max: float,
    noise_var: float,
    bandwidth: float,
    max_steps: int,
    seed: Optional[int],
) -> PaddedWirelessAllocationTask:
    config = WirelessTaskConfig(
        num_users=num_users,
        max_users=max_users,
        p_max=p_max,
        noise_var=noise_var,
        bandwidth=bandwidth,
        max_steps=max_steps,
    )
    return PaddedWirelessAllocationTask(config, seed=seed)


def train_maml(args: argparse.Namespace) -> MAMLActorCritic:
    if args.max_users < max(args.user_counts):
        raise ValueError("--max-users must be at least the largest value in --user-counts")

    device = torch.device(args.device)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    obs_dim = args.max_users * args.max_users + 2 * args.max_users
    model = MAMLActorCritic(obs_dim, args.max_users, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    for meta_iter in range(1, args.meta_iters + 1):
        optimizer.zero_grad()
        query_losses: List[torch.Tensor] = []
        support_rewards: List[float] = []
        query_rewards: List[float] = []

        sampled_counts = np_rng.choice(args.user_counts, size=args.meta_batch_size, replace=True)
        for num_users in sampled_counts:
            params = clone_parameters(model)

            for _inner_step in range(args.inner_steps):
                support_task = make_task(
                    int(num_users),
                    args.max_users,
                    args.p_max,
                    args.noise_var,
                    args.bandwidth,
                    args.episode_steps,
                    seed=int(np_rng.integers(0, 2**31 - 1)),
                )
                support_rollout = collect_rollout(
                    support_task,
                    model,
                    args.support_steps,
                    device,
                    params=params,
                )
                support_rollout = add_gae_returns(support_rollout, args.gamma, args.gae_lambda)
                params, support_metrics = maml_adapt(
                    model,
                    params,
                    support_rollout,
                    args.inner_lr,
                    args.clip_range,
                    args.vf_coef,
                    args.ent_coef,
                    args.first_order,
                )
                support_rewards.append(support_metrics["mean_reward"])

            query_task = make_task(
                int(num_users),
                args.max_users,
                args.p_max,
                args.noise_var,
                args.bandwidth,
                args.episode_steps,
                seed=int(np_rng.integers(0, 2**31 - 1)),
            )
            query_rollout = collect_rollout(
                query_task,
                model,
                args.query_steps,
                device,
                params=params,
            )
            query_rollout = add_gae_returns(query_rollout, args.gamma, args.gae_lambda)
            query_loss, query_metrics = ppo_loss(
                model,
                query_rollout,
                params=params,
                clip_range=args.clip_range,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
            )
            query_losses.append(query_loss)
            query_rewards.append(query_metrics["mean_reward"])

        meta_loss = torch.stack(query_losses).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if meta_iter == 1 or meta_iter % args.log_every == 0:
            print(
                f"iter={meta_iter:04d} "
                f"meta_loss={meta_loss.item():.3f} "
                f"support_reward={np.mean(support_rewards):.3f} "
                f"query_reward={np.mean(query_rewards):.3f} "
                f"tasks={list(map(int, sampled_counts))}"
            )

        if args.save_every > 0 and meta_iter % args.save_every == 0:
            save_checkpoint(model, args, args.output)

    save_checkpoint(model, args, args.output)
    return model


@torch.no_grad()
def evaluate_after_adaptation(
    model: MAMLActorCritic,
    args: argparse.Namespace,
    num_users: int,
    seed: int,
) -> Dict[str, float]:
    device = torch.device(args.device)
    task = make_task(
        num_users,
        args.max_users,
        args.p_max,
        args.noise_var,
        args.bandwidth,
        args.episode_steps,
        seed=seed,
    )
    params = clone_parameters(model)

    # Evaluation uses no meta-gradient. Adaptation is done with first-order grads.
    with torch.enable_grad():
        support_rollout = collect_rollout(
            task,
            model,
            args.support_steps,
            device,
            params=params,
        )
        support_rollout = add_gae_returns(support_rollout, args.gamma, args.gae_lambda)
        params, _metrics = maml_adapt(
            model,
            params,
            support_rollout,
            args.inner_lr,
            args.clip_range,
            args.vf_coef,
            args.ent_coef,
            first_order=True,
        )

    eval_task = make_task(
        num_users,
        args.max_users,
        args.p_max,
        args.noise_var,
        args.bandwidth,
        args.episode_steps,
        seed=seed + 1,
    )
    obs = eval_task.reset()
    mask = eval_task._active_mask()

    rewards = []
    sum_rates = []
    for _ in range(args.eval_steps):
        action, _log_prob, _value = select_action(
            model,
            obs,
            mask,
            device,
            params=params,
            deterministic=True,
        )
        obs, reward, done, info = eval_task.step(action)
        rewards.append(reward)
        sum_rates.append(float(info["sum_rate_mbps"]))
        if done:
            obs = eval_task.reset()
        mask = eval_task._active_mask()

    return {
        "reward": float(np.mean(rewards)),
        "sum_rate_mbps": float(np.mean(sum_rates)),
    }


def save_checkpoint(model: MAMLActorCritic, args: argparse.Namespace, path: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "task_config": asdict(
            WirelessTaskConfig(
                num_users=max(args.user_counts),
                max_users=args.max_users,
                p_max=args.p_max,
                noise_var=args.noise_var,
                bandwidth=args.bandwidth,
                max_steps=args.episode_steps,
            )
        ),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[MAMLActorCritic, Dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch does not support weights_only
        checkpoint = torch.load(path, map_location=device)
    saved_args = checkpoint["args"]
    obs_dim = saved_args["max_users"] * saved_args["max_users"] + 2 * saved_args["max_users"]
    model = MAMLActorCritic(
        obs_dim,
        saved_args["max_users"],
        hidden_dim=saved_args["hidden_dim"],
    ).to(torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, saved_args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAML + PPO meta-training for wireless power allocation across user counts."
    )
    parser.add_argument("--user-counts", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--max-users", type=int, default=6)
    parser.add_argument("--meta-iters", type=int, default=200)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--support-steps", type=int, default=64)
    parser.add_argument("--query-steps", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--episode-steps", type=int, default=100)

    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--meta-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--p-max", type=float, default=10.0)
    parser.add_argument("--noise-var", type=float, default=0.1)
    parser.add_argument("--bandwidth", type=float, default=1e6)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--first-order", action="store_true", help="Use first-order MAML for lower memory.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--output", type=str, default="maml_ppo_wireless.pt")
    parser.add_argument("--no-eval", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model = train_maml(args)
    if args.no_eval:
        return

    print("\n--- Adapted policy evaluation ---")
    for num_users in args.user_counts:
        metrics = evaluate_after_adaptation(
            model,
            args,
            num_users=num_users,
            seed=args.seed + 1000 + num_users,
        )
        print(
            f"users={num_users}: "
            f"reward={metrics['reward']:.3f}, "
            f"sum_rate={metrics['sum_rate_mbps']:.3f} Mbps"
        )


if __name__ == "__main__":
    main()
