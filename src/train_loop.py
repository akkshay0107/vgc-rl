import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from encoder import ACT_SIZE, OBS_DIM, Encoder
from env import SimEnv
from policy import PolicyNet

num_episodes = 2500
max_steps_per_episode = 100
gamma = 0.99
lr = 3e-4
batch_size = 64
buffer_size = 10000
entropy_coef = 0.01

env = SimEnv.build_env()
policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.Adam(policy.parameters(), lr=lr)
replay_buffer = deque(maxlen=buffer_size)


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
    checkpoint = torch.load(path)
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("episode", None)


def store_transition(obs, action, reward, next_obs, action_mask, done):
    replay_buffer.append((obs, action, reward, next_obs, action_mask, done))


def update_policy():
    if len(replay_buffer) < batch_size:
        return

    batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
    transitions = [replay_buffer[i] for i in batch]

    obs_batch = torch.stack([t[0] for t in transitions])
    action_batch = torch.stack([torch.tensor(t[1], dtype=torch.long) for t in transitions])
    reward_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    next_obs_batch = torch.stack([t[3] for t in transitions])
    mask_batch = torch.stack([t[4] for t in transitions])
    done_batch = torch.tensor([t[5] for t in transitions], dtype=torch.float32)

    with torch.no_grad():
        _, _, _, next_values = policy(next_obs_batch, mask_batch, sample_actions=False)

    _, _, _, values = policy(obs_batch, mask_batch, sample_actions=False)

    target_values = (
        reward_batch.unsqueeze(-1) + gamma * (1 - done_batch.unsqueeze(-1)) * next_values
    )
    critic_loss = F.mse_loss(values, target_values.detach())

    log_probs, _ = policy.evaluate_actions(obs_batch, action_batch, mask_batch)
    advantages = (target_values - values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = -(log_probs * advantages.unsqueeze(-1)).mean() + entropy_coef * log_probs.mean()

    loss = critic_loss + actor_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()


# optional code below to load a checkpoint
start_episode = (
    load_checkpoint("./checkpoints/checkpoint_<episode_num>.pt") or 0
)  # fails here since path doesnt exist

for episode in range(start_episode, num_episodes):
    obs, _ = env.reset()
    episode_rewards = []

    while True:
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        action_mask_agent1 = Encoder.get_action_mask(env.battle1)  # type: ignore
        action_mask_agent2 = Encoder.get_action_mask(env.battle2)  # type: ignore

        # Stack along batch dim for batch processing by policy network
        obs_batch = torch.stack([obs_agent1, obs_agent2], dim=0)  # shape (2, *obs_shape)
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0)

        with torch.no_grad():
            _, _, sampled_actions, values = policy.forward(obs_batch, action_mask_batch)

        assert sampled_actions is not None
        action1_np = sampled_actions[0].cpu().numpy()
        action2_np = sampled_actions[1].cpu().numpy()

        actions = {
            env.agent1.username: action1_np,
            env.agent2.username: action2_np,
        }

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done1 = terminated[env.agent1.username] or truncated[env.agent1.username]
        done2 = terminated[env.agent2.username] or truncated[env.agent2.username]
        episode_rewards.append(rewards[env.agent1.username])

        store_transition(
            obs_agent1,
            action1_np,
            rewards[env.agent1.username],
            next_obs[env.agent1.username],
            action_mask_agent1,
            done1,
        )

        store_transition(
            obs_agent2,
            action2_np,
            rewards[env.agent2.username],
            next_obs[env.agent2.username],
            action_mask_agent2,
            done2,
        )

        obs = next_obs

        if done1 or done2:
            break

    if episode % 10 == 0:
        update_policy()
        save_checkpoint(f"./checkpoints/checkpoint_{episode}.pt", episode + 1)
        print(f"Checkpoint saved at episode {episode}")

    print(
        f"Episode {episode + 1}/{num_episodes}, reward: {sum(episode_rewards):.3f}, buffer: {len(replay_buffer)}"
    )
