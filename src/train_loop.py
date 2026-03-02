import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import observation_builder
from env import SimEnv
from observation_builder import ACT_SIZE, OBS_DIM
from policy import PolicyNet


@dataclass
class PPOConfig:
    num_episodes: int = 10
    n_jobs: int = 8
    rollouts_per_episode: int = 16

    gamma: float = 0.95
    gae_lambda: float = 0.98  # sparse reward counteraction
    lr: float = 1e-4
    batch_size: int = 64
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    ppo_epochs: int = 4

    best_update_threshold: float = 0.55
    checkpoint_path: str = "ppo_checkpoint.pt"
    best_checkpoint_path: str = "ppo_best_checkpoint.pt"


config = PPOConfig()
_inference_lock = threading.Lock()
_buffer_lock = threading.Lock()

policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
best_policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.AdamW(policy.parameters(), lr=config.lr, eps=1e-5)

if policy.device.type == "cuda":
    policy.compile()

if best_policy.device.type == "cuda":
    best_policy.compile()


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

    def add(self, data):
        self.obs.append(data["obs"].cpu())
        self.actions.append(data["actions"].cpu())
        self.log_probs.append(data["log_probs"].cpu())
        self.values.append(data["values"].cpu())
        self.rewards.append(data["rewards"].cpu())
        self.dones.append(data["dones"].cpu())
        self.action_masks.append(data["action_masks"].cpu())

    def get_batches(self, device):
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

        flat_obs = torch.stack(self.obs).reshape(T * B, *OBS_DIM).to(device)
        flat_actions = torch.stack(self.actions).reshape(T * B, 2).to(device)
        flat_log_probs = torch.stack(self.log_probs).reshape(-1).to(device)
        flat_action_masks = torch.stack(self.action_masks).reshape(T * B, 2, ACT_SIZE).to(device)

        return {
            "obs": flat_obs,
            "actions": flat_actions,
            "log_probs": flat_log_probs,
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "action_masks": flat_action_masks,
        }


def save_checkpoint(path, episode):
    torch.save(
        {
            "episode": episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_best_checkpoint(path, episode, win_rate):
    torch.save(
        {
            "episode": episode,
            "win_rate": win_rate,
            "model_state_dict": best_policy.state_dict(),
        },
        path,
    )


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("episode", None)


def collect_rollout(env, buffer):
    policy.eval()
    best_policy.eval()
    obs, _ = env.reset()
    local_transitions = []

    while True:
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        action_mask_agent1 = observation_builder.get_action_mask(env.battle1)
        action_mask_agent2 = observation_builder.get_action_mask(env.battle2)

        obs_batch = obs_agent1.unsqueeze(0).to(policy.device)
        action_mask_batch = action_mask_agent1.unsqueeze(0).to(policy.device)

        with _inference_lock:
            with torch.no_grad():
                _, log_probs, action1, values = policy(
                    obs_batch,
                    action_mask_batch,
                )
                _, _, action2, _ = best_policy(
                    obs_agent2.unsqueeze(0).to(best_policy.device),
                    action_mask_agent2.unsqueeze(0).to(best_policy.device),
                )

        action1_np = action1[0].cpu().numpy()
        action2_np = action2[0].cpu().numpy()

        actions = {
            env.agent1.username: action1_np,
            env.agent2.username: action2_np,
        }

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done1 = terminated[env.agent1.username] or truncated[env.agent1.username]
        done2 = terminated[env.agent2.username] or truncated[env.agent2.username]

        local_transitions.append(
            {
                "obs": obs_batch,
                "actions": action1.to(policy.device),
                "log_probs": log_probs,
                "values": values,
                "rewards": torch.tensor(
                    [rewards[env.agent1.username]],
                    dtype=torch.float32,
                    device=policy.device,
                ),
                "dones": torch.tensor([done1], dtype=torch.float32, device=policy.device),
                "action_masks": action_mask_batch,
            }
        )

        obs = next_obs
        if done1 or done2:
            break

    with _buffer_lock:
        for t in local_transitions:
            buffer.add(t)


def collect_all_rollouts(envs, buffer):
    env_queue = queue.Queue()
    for env in envs:
        env_queue.put(env)

    def worker():
        env = env_queue.get()
        try:
            collect_rollout(env, buffer)
        finally:
            env_queue.put(env)

    with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
        futures = [executor.submit(worker) for _ in range(config.rollouts_per_episode)]
        for f in as_completed(futures):
            f.result()


def ppo_update(rollout_data):
    policy.train()
    t0 = time.time()

    observations = rollout_data["obs"]
    actions = rollout_data["actions"]
    old_log_probs = rollout_data["log_probs"]
    advantages = rollout_data["advantages"]
    returns = rollout_data["returns"]
    action_masks = rollout_data["action_masks"]

    n_samples = len(observations)

    # overall metrics
    tot_avg_policy_loss = 0.0
    tot_avg_value_loss = 0.0
    tot_avg_entropy_loss = 0.0
    tot_avg_kl_div = 0.0
    epochs_done = 0

    for epoch_idx in range(config.ppo_epochs):
        # per epoch stats
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        avg_entropy_loss = 0.0
        avg_kl_div = 0.0

        minibatch_indices = np.random.permutation(n_samples)

        for minibatch_start in range(0, n_samples, config.batch_size):
            minibatch_end = minibatch_start + config.batch_size
            mb_idx = minibatch_indices[minibatch_start:minibatch_end]

            mb_observations = observations[mb_idx].to(policy.device)
            mb_actions = actions[mb_idx].to(policy.device)
            mb_old_log_probs = old_log_probs[mb_idx].to(policy.device)
            mb_advantages = advantages[mb_idx].to(policy.device)
            mb_returns = returns[mb_idx].to(policy.device)
            mb_action_masks = action_masks[mb_idx].to(policy.device)

            new_log_probs, entropy, new_values = policy.evaluate_actions(
                mb_observations, mb_actions, mb_action_masks
            )

            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            policy_loss_unclipped = mb_advantages * ratio
            policy_loss_clipped = mb_advantages * torch.clamp(
                ratio, 1.0 - config.clip_range, 1.0 + config.clip_range
            )

            policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
            value_loss = F.mse_loss(new_values, mb_returns)
            entropy_loss = -entropy.mean()

            total_loss = (
                policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = (mb_old_log_probs - new_log_probs).mean().item()

            avg_policy_loss += policy_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy_loss += entropy_loss.item()
            avg_kl_div += kl

        num_mb = (n_samples + config.batch_size - 1) // config.batch_size
        epochs_done += 1

        avg_policy_loss /= num_mb
        avg_value_loss /= num_mb
        avg_entropy_loss /= num_mb
        avg_kl_div /= num_mb

        tot_avg_policy_loss += avg_policy_loss
        tot_avg_value_loss += avg_value_loss
        tot_avg_entropy_loss += avg_entropy_loss
        tot_avg_kl_div += avg_kl_div

        if avg_kl_div > config.target_kl:
            print(
                f"Early stop at epoch {epoch_idx + 1}/{config.ppo_epochs} (KL={avg_kl_div:.4f} > {config.target_kl})"
            )
            break

    t1 = time.time()

    return {
        "policy_loss": tot_avg_policy_loss / epochs_done,
        "value_loss": tot_avg_value_loss / epochs_done,
        "entropy_loss": tot_avg_entropy_loss / epochs_done,
        "kl_divergence": tot_avg_kl_div / epochs_done,
        "time": t1 - t0,
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


def main():
    envs = [SimEnv.build_env(env_id=i) for i in range(config.n_jobs)]
    buffer = RolloutBuffer()

    start = load_checkpoint(config.checkpoint_path) or 0
    best_policy.load_state_dict(policy.state_dict())
    if os.path.exists(config.best_checkpoint_path):
        best_checkpoint = torch.load(config.best_checkpoint_path, map_location=best_policy.device)
        best_policy.load_state_dict(best_checkpoint["model_state_dict"])

    print(f"Starting training from episode {start + 1}")
    for episode in range(start, config.num_episodes):
        print(f"Collecting rollout for episode {episode + 1}")

        t0_rollout = time.time()
        buffer.reset()
        collect_all_rollouts(envs, buffer)
        rollout_time = time.time() - t0_rollout
        processed_rollout_data = buffer.get_batches(policy.device)
        winrate_stats = get_winrate(buffer)

        if winrate_stats["win_rate"] >= config.best_update_threshold:
            best_policy.load_state_dict(policy.state_dict())
            print(f"Updated best_policy (win rate {winrate_stats['win_rate']:.2%}).")
            save_best_checkpoint(
                config.best_checkpoint_path, episode + 1, winrate_stats["win_rate"]
            )
            print("Best policy checkpoint saved.")

        stats = ppo_update(processed_rollout_data)

        print("=" * 60)
        print(f"Episode {episode + 1}/{config.num_episodes}:")
        print(
            f"  Win Rate: {winrate_stats['win_rate']:.2%} (W/L: {winrate_stats['wins']}/{winrate_stats['losses']})"
        )
        print(f"  Rollout Time: {rollout_time:.4f} s")
        print(f"  Policy Loss: {stats['policy_loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Entropy Loss: {stats['entropy_loss']:.4f}")
        print(f"  KL Divergence: {stats['kl_divergence']:.4f}")
        print(f"  Update Time: {stats['time']:.4f} s")
        print("=" * 60)

        if (episode + 1) % 10 == 0:
            print(f"Saving checkpoint at episode {episode + 1}")
            save_checkpoint(config.checkpoint_path, episode + 1)
            print("Checkpoint saved.")


if __name__ == "__main__":
    main()
