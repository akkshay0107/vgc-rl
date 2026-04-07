import functools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from lookups import ACT_SIZE, OBS_DIM
from policy import PolicyNet


@dataclass
class PPOConfig:
    num_episodes: int = 10
    n_jobs: int = 2
    rollouts_per_episode: int = 64

    gamma: float = 0.96  # effective horizon = ~25 turns
    gae_lambda: float = 0.98  # sparse reward counteraction
    lr: float = 1e-4
    batch_size: int = 64
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    ppo_epochs: int = 4

    checkpoint_path: Path = Path(__file__).parent.parent / "checkpoints" / "ppo_checkpoint.pt"
    pool_dir: Path = Path(__file__).parent.parent / "checkpoints" / "pool"
    pool_size: int = 10
    snapshot_interval: int = 10
    # EMA smoothing factor for per-opponent win-rate tracking (lower = smoother).
    pool_win_rate_smoothing: float = 0.1
    # Minimum sampling weight for any opponent (prevents starvation).
    min_pool_win_rate_weight: float = 0.1
    pool_cache_size: int = 5
    self_play_prob: float = 0.5


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []

    def add(self, data: dict):
        # make sure all the data on the cpu beforehand
        # pin memory when using gpus
        self.obs.append(data["obs"])
        self.actions.append(data["actions"])
        self.log_probs.append(data["log_probs"])
        self.values.append(data["values"])
        self.rewards.append(data["rewards"])
        self.dones.append(data["dones"])
        self.action_masks.append(data["action_masks"])

    def get_batches(self, device: torch.device, config: PPOConfig):
        rewards = torch.stack(self.rewards).to(device)
        values = torch.stack(self.values).to(device)
        dones = torch.stack(self.dones).to(device)

        T, B = rewards.shape
        advantages = torch.zeros(T, B, dtype=torch.float32, device=device)
        gae = torch.zeros(B, dtype=torch.float32, device=device)

        for t in reversed(range(T)):
            next_val = torch.zeros_like(values[t]) if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + config.gamma * next_val * mask - values[t]
            gae = delta + config.gamma * config.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_obs = torch.stack(self.obs).reshape(T * B, *OBS_DIM).to(device, non_blocking=True)
        flat_actions = torch.stack(self.actions).reshape(T * B, 2).to(device, non_blocking=True)
        flat_log_probs = torch.stack(self.log_probs).reshape(-1).to(device, non_blocking=True)
        flat_action_masks = (
            torch.stack(self.action_masks).reshape(T * B, 2, ACT_SIZE).to(device, non_blocking=True)
        )

        return {
            "obs": flat_obs,
            "actions": flat_actions,
            "log_probs": flat_log_probs,
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "action_masks": flat_action_masks,
        }


def save_checkpoint(path: Path, episode: int, policy: PolicyNet, optimizer=None, scheduler=None):
    state = {
        "episode": episode,
        "model_state_dict": policy.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def save_best_checkpoint(path: Path, episode: int, win_rate: float, policy: PolicyNet):
    state = {
        "episode": episode,
        "win_rate": win_rate,
        "model_state_dict": policy.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path: Path, policy: PolicyNet, optimizer=None, scheduler=None):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("episode", None)


# AlphaStar style pool of checkpoints for diverse training
class OpponentPool:
    def __init__(self, pool_dir: Path, config: PPOConfig):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        # Ordered list of opponent ids (strings); first = oldest.
        self.opponent_ids: list[str] = []
        # EMA win-rate of the training policy against each opponent.
        # Starts at 0.5 (neutral) for newly added opponents.
        self.win_rates: dict[str, float] = {}
        # LRU cache wrapper for loading policies
        self._load_policy_cached = functools.lru_cache(maxsize=config.pool_cache_size)(
            self._load_policy_impl
        )

    def save_state(self) -> None:
        state = {
            "opponent_ids": self.opponent_ids,
            "win_rates": self.win_rates,
        }
        with open(self.pool_dir / "pool_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        path = self.pool_dir / "pool_state.json"
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.opponent_ids = state.get("opponent_ids", [])
            self.win_rates = state.get("win_rates", {})

        # Sync with filesystem for any missing .pt files (e.g. if json was lost)
        for pt_file in sorted(self.pool_dir.glob("*.pt")):
            opponent_id = pt_file.stem
            if opponent_id not in self.opponent_ids:
                self.opponent_ids.append(opponent_id)
                if opponent_id not in self.win_rates:
                    self.win_rates[opponent_id] = 0.5

        # Cleanup: remove any IDs that no longer have a corresponding .pt file
        existing_ids = {pt.stem for pt in self.pool_dir.glob("*.pt")}
        self.opponent_ids = [oid for oid in self.opponent_ids if oid in existing_ids]
        self.win_rates = {oid: wr for oid, wr in self.win_rates.items() if oid in existing_ids}

    @classmethod
    def load_or_create(cls, pool_dir: Path, config: PPOConfig) -> Self:
        """Load an existing pool from disk, or create an empty one."""
        pool = cls(pool_dir, config)
        pool._load_state()
        return pool

    def add(self, policy: PolicyNet, opponent_id: str) -> None:
        # Evict oldest if at capacity.
        while len(self.opponent_ids) >= self.config.pool_size:
            evicted_id = self.opponent_ids.pop(0)
            self.win_rates.pop(evicted_id, None)
            evicted_path = self.pool_dir / f"{evicted_id}.pt"
            if evicted_path.exists():
                evicted_path.unlink()

        # Save checkpoint.
        torch.save({"model_state_dict": policy.state_dict()}, self.pool_dir / f"{opponent_id}.pt")
        self.opponent_ids.append(opponent_id)
        # Neutral starting win-rate.
        self.win_rates[opponent_id] = 0.5

    def update_win_rate(self, opponent_id: str, won: bool) -> None:
        if opponent_id not in self.win_rates:
            return
        alpha = self.config.pool_win_rate_smoothing
        self.win_rates[opponent_id] = (1 - alpha) * self.win_rates[opponent_id] + alpha * float(won)

    def sample(self) -> tuple[PolicyNet, str]:
        # Returns frozen snapshot of a policy from the pool (can't train it)
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample().")

        floor = self.config.min_pool_win_rate_weight
        # Weight = 1 - ema_win_rate, clamped to [floor, 1.0].
        # Opponents we beat easily get low weight; hard ones get high weight.
        weights = [max(floor, 1.0 - self.win_rates.get(oid, 0.5)) for oid in self.opponent_ids]
        (opponent_id,) = random.choices(self.opponent_ids, weights=weights, k=1)
        return self._load_policy_cached(opponent_id), opponent_id

    def _load_policy_impl(self, opponent_id: str) -> PolicyNet:
        checkpoint = torch.load(
            self.pool_dir / f"{opponent_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        net = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    def __len__(self) -> int:
        return len(self.opponent_ids)

    def __repr__(self) -> str:
        return f"OpponentPool(size={len(self)}/{self.config.pool_size}, ids={self.opponent_ids})"
