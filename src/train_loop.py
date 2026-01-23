import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import observation_builder
from env import SimEnv
from observation_builder import ACT_SIZE, OBS_DIM
from policy import PolicyNet

num_episodes = 1000
gamma = 0.95  # avg vgc game length = 8 turns (0.95)^8 ~ 0.66
gae_lambda = 0.95
lr = 1e-4
batch_size = 64
clip_range = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
target_kl = 0.015  # for early KL stopping

env = SimEnv.build_env()
policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.Adam(policy.parameters(), lr=lr)

rollout_buffer = []
replay_buffer = deque(maxlen=10000)


def save_checkpoint(path, episode):
    torch.save(
        {
            "episode": episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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


def compute_gae(rewards, values, dones, gamma=gamma, gae_lambda=gae_lambda):
    T, B = rewards.shape
    advantages = torch.zeros(T, B, dtype=torch.float32, device=rewards.device)
    gae = torch.zeros(B, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        next_val = torch.zeros_like(values[t]) if t == T - 1 else values[t + 1]  # (B,)
        mask = 1.0 - dones[t]  # (B,)
        delta = rewards[t] + gamma * next_val * mask - values[t]  # (B,)
        gae = delta + gamma * gae_lambda * mask * gae  # (B,)
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def collect_rollout():
    obs, _ = env.reset()
    episode_data = []

    while True:
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        action_mask_agent1 = observation_builder.get_action_mask(env.battle1)  # type: ignore
        action_mask_agent2 = observation_builder.get_action_mask(env.battle2)  # type: ignore

        obs_batch = [obs_agent1, obs_agent2]
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0).to(
            policy.device
        )

        with torch.no_grad():
            logits, log_probs, sampled_actions, values = policy.batch_forward(
                obs_batch, action_mask_batch
            )

        assert sampled_actions is not None and log_probs is not None
        action1_np = sampled_actions[0].cpu().numpy()
        action2_np = sampled_actions[1].cpu().numpy()

        actions = {
            env.agent1.username: action1_np,
            env.agent2.username: action2_np,
        }

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done1 = terminated[env.agent1.username] or truncated[env.agent1.username]
        done2 = terminated[env.agent2.username] or truncated[env.agent2.username]

        # each batch contains both player and opponents perspective
        episode_data.append(
            {
                "obs": obs_batch,
                "actions": sampled_actions,
                "log_probs": log_probs,
                "values": values,  # (2,)
                "rewards": torch.tensor(
                    [rewards[env.agent1.username], rewards[env.agent2.username]],
                    dtype=torch.float32,
                    device=policy.device,
                ),  # (2,)
                "dones": torch.tensor(
                    [done1, done2], dtype=torch.float32, device=policy.device
                ),  # (2,)
                "action_masks": action_mask_batch,
            }
        )

        obs = next_obs
        if done1 or done2:
            break

    return episode_data


def main():
    pass


if __name__ == "__main__":
    main()
