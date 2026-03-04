from dataclasses import dataclass
from pathlib import Path

import torch

from constants import ACT_SIZE, OBS_DIM


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

    best_update_threshold: float = 0.54
    checkpoint_path: Path = Path(__file__).parent.parent / "checkpoints" / "ppo_checkpoint.pt"
    best_checkpoint_path: Path = (
        Path(__file__).parent.parent / "checkpoints" / "ppo_best_checkpoint.pt"
    )


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


def get_winrate(buffer: RolloutBuffer):
    terminal_rewards = [r.item() for r, d in zip(buffer.rewards, buffer.dones) if d.item() == 1.0]
    if not terminal_rewards:
        return {"win_rate": 0.0, "wins": 0, "losses": 0}

    wins = sum(1 for r in terminal_rewards if r > 0)
    losses = sum(1 for r in terminal_rewards if r < 0)
    return {
        "win_rate": wins / len(terminal_rewards),
        "wins": wins,
        "losses": losses,
    }


def save_checkpoint(path, episode, policy, optimizer=None, scheduler=None):
    state = {
        "episode": episode,
        "model_state_dict": policy.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def save_best_checkpoint(path, episode, win_rate, policy):
    state = {
        "episode": episode,
        "win_rate": win_rate,
        "model_state_dict": policy.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, policy, optimizer=None, scheduler=None):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("episode", None)
