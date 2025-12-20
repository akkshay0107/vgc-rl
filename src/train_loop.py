import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from encoder import ACT_SIZE, OBS_DIM, Encoder
from env import SimEnv
from policy import PolicyNet

# PPG Hyperparameters
num_episodes = 2500
gamma = 0.99
gae_lambda = 0.95
lr = 5e-4
n_policy_iterations = 32
n_aux_epochs = 6
ppo_epochs = 1
value_epochs = 1
batch_size = 64
clip_range = 0.2
entropy_coef = 0.01
value_coef = 0.5
beta_clone = 1.0
max_grad_norm = 0.5

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
    checkpoint = torch.load(path)
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("episode", None)


def compute_gae(rewards, values, dones, next_value, gamma=gamma, gae_lambda=gae_lambda):
    T, B = rewards.shape
    advantages = torch.zeros(T, B, dtype=torch.float32, device=rewards.device)
    gae = torch.zeros(B, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]  # (B,)
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

        action_mask_agent1 = Encoder.get_action_mask(env.battle1)  # type: ignore
        action_mask_agent2 = Encoder.get_action_mask(env.battle2)  # type: ignore

        obs_batch = torch.stack([obs_agent1, obs_agent2], dim=0)  # (2, ...)
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0)

        with torch.no_grad():
            logits, log_probs, sampled_actions, values = policy(obs_batch, action_mask_batch)

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

        episode_data.append(
            {
                "obs": obs_batch,
                "actions": sampled_actions,
                "log_probs": log_probs,
                "values": values,  # (2,)
                "rewards": torch.tensor(
                    [rewards[env.agent1.username], rewards[env.agent2.username]],
                    dtype=torch.float32,
                ),  # (2,)
                "dones": torch.tensor([done1, done2], dtype=torch.float32),  # (2,)
                "action_masks": action_mask_batch,
            }
        )

        obs = next_obs

        if done1 or done2:
            with torch.no_grad():
                next_obs_batch = torch.stack(
                    [next_obs[env.agent1.username], next_obs[env.agent2.username]], dim=0
                )
                next_mask_batch = torch.stack(
                    [Encoder.get_action_mask(env.battle1), Encoder.get_action_mask(env.battle2)],  # type: ignore
                    dim=0,
                )
                _, _, _, next_values = policy(next_obs_batch, next_mask_batch, sample_actions=False)
            break

    return episode_data, next_values


def policy_phase():
    episode_data, next_values = collect_rollout()

    obs_list = torch.stack([d["obs"] for d in episode_data])
    actions_list = torch.stack([d["actions"] for d in episode_data])
    old_log_probs_list = torch.stack([d["log_probs"] for d in episode_data])
    values_list = torch.stack([d["values"] for d in episode_data])
    rewards_list = torch.stack([d["rewards"] for d in episode_data])
    dones_list = torch.stack([d["dones"] for d in episode_data])
    masks_list = torch.stack([d["action_masks"] for d in episode_data])

    advantages, returns = compute_gae(
        rewards=rewards_list,
        values=values_list,
        dones=dones_list,
        next_value=next_values,
    )  # advantages, returns: (T, 2)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    with torch.no_grad():
        # Flatten turns and batch
        T = obs_list.shape[0]
        B = obs_list.shape[1]
        flat_obs = obs_list.view(T * B, *obs_list.shape[2:])
        flat_masks = masks_list.view(T * B, *masks_list.shape[2:])
        policy_probs = policy.get_policy_probs(flat_obs, flat_masks)  # (T*B, act_dim)

    for t in range(len(episode_data)):
        for agent_idx in range(2):
            idx = t * 2 + agent_idx
            replay_buffer.append(
                {
                    "obs": obs_list[t, agent_idx],
                    "action_mask": masks_list[t, agent_idx],
                    "returns": returns[t, agent_idx],
                    "old_policy_probs": policy_probs[idx].detach(),
                }
            )

    # Flatten rollout for PPO / value updates
    T = obs_list.shape[0]
    B = obs_list.shape[1]
    flat_obs = obs_list.view(T * B, *obs_list.shape[2:])
    flat_actions = actions_list.view(T * B, *actions_list.shape[2:])
    flat_old_log_probs = old_log_probs_list.view(T * B, *old_log_probs_list.shape[2:])
    flat_advantages = advantages.view(T * B)
    flat_returns = returns.view(T * B)
    flat_masks = masks_list.view(T * B, *masks_list.shape[2:])

    # Policy update
    num_samples = T * B
    for _ in range(ppo_epochs):
        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, batch_size):
            idx = indices[start : start + batch_size]

            obs_batch = flat_obs[idx]
            actions_batch = flat_actions[idx]
            old_log_probs_batch = flat_old_log_probs[idx]
            adv_batch = flat_advantages[idx]
            masks_batch = flat_masks[idx]

            log_probs, _ = policy.evaluate_actions(obs_batch, actions_batch, masks_batch)
            ratio = torch.exp(log_probs - old_log_probs_batch)
            surr1 = ratio * adv_batch.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv_batch.unsqueeze(-1)

            policy_loss = -torch.min(surr1, surr2).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

    # Value function update
    for _ in range(value_epochs):
        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, batch_size):
            idx = indices[start : start + batch_size]

            obs_batch = flat_obs[idx]
            returns_batch = flat_returns[idx]
            masks_batch = flat_masks[idx]

            _, _, _, values = policy(
                obs_batch, masks_batch, sample_actions=False
            )  # values: (batch,) or (batch,1)
            value_loss = F.mse_loss(values.squeeze(-1), returns_batch)

            optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

    total_reward = rewards_list.sum().item()
    return total_reward


def auxiliary_phase():
    if len(replay_buffer) < batch_size:
        return

    for _ in range(n_aux_epochs):
        indices = np.random.choice(
            len(replay_buffer), min(batch_size, len(replay_buffer)), replace=False
        )
        batch = [replay_buffer[i] for i in indices]

        obs_batch = torch.stack([b["obs"] for b in batch])
        mask_batch = torch.stack([b["action_mask"] for b in batch])
        returns_batch = torch.stack([b["returns"] for b in batch])
        old_policy_probs_batch = torch.stack([b["old_policy_probs"] for b in batch])

        # Auxiliary value loss (aux head)
        aux_values = policy.get_aux_value(obs_batch)
        aux_value_loss = F.mse_loss(aux_values.squeeze(-1), returns_batch)

        # KL divergence loss KL(old || new)
        current_policy_probs = policy.get_policy_probs(obs_batch, mask_batch)
        kl_loss = F.kl_div(
            torch.log(current_policy_probs + 1e-10),  # log(new)
            old_policy_probs_batch,  # old
            reduction="batchmean",
        )

        joint_loss = aux_value_loss + beta_clone * kl_loss

        optimizer.zero_grad()
        joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        # Additional value head training
        _, _, _, values = policy(obs_batch, mask_batch, sample_actions=False)
        value_loss = F.mse_loss(values.squeeze(-1), returns_batch)

        optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()


def main():
    start_episode = load_checkpoint("./checkpoints/checkpoint_latest.pt") or 0
    for episode in range(start_episode, num_episodes):
        # Policy phase
        avg_rew = 0
        for _ in range(n_policy_iterations):
            avg_rew += policy_phase()
        avg_rew /= n_policy_iterations

        # Auxiliary phase
        auxiliary_phase()

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{num_episodes}, reward: {avg_rew:.3f}, buffer: {len(replay_buffer)}"
            )

        if episode % 50 == 0:
            save_checkpoint(f"./checkpoints/checkpoint_{episode}.pt", episode)
            save_checkpoint("./checkpoints/checkpoint_latest.pt", episode)
            print(f"Checkpoint saved at episode {episode}")


if __name__ == "__main__":
    main()
