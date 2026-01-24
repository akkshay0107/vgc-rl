import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import observation_builder
from env import SimEnv
from observation_builder import ACT_SIZE, OBS_DIM
from policy import PolicyNet

# Hyperparameters
num_episodes = 1000
gamma = 0.95
gae_lambda = 0.95
lr = 1e-4
batch_size = 2  # for testing (since my cpu doesnt have enough memory), increase it later
clip_range = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
target_kl = 0.015
ppo_epochs = 4
n_jobs = 4

policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.Adafactor(policy.parameters(), lr=lr)


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
        next_val = torch.zeros_like(values[t]) if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def collect_rollout(env):
    obs, _ = env.reset()
    episode_data = []

    while True:
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        action_mask_agent1 = observation_builder.get_action_mask(env.battle1)
        action_mask_agent2 = observation_builder.get_action_mask(env.battle2)

        obs_batch = [obs_agent1, obs_agent2]
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0).to(
            policy.device
        )

        with torch.no_grad():
            _, log_probs, sampled_actions, values = policy.batch_forward(
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

        episode_data.append(
            {
                "obs": obs_batch,
                "actions": sampled_actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": torch.tensor(
                    [rewards[env.agent1.username], rewards[env.agent2.username]],
                    dtype=torch.float32,
                    device=policy.device,
                ),
                "dones": torch.tensor([done1, done2], dtype=torch.float32, device=policy.device),
                "action_masks": action_mask_batch,
            }
        )

        obs = next_obs
        if done1 or done2:
            break

    return episode_data


def ppo_update(rollout_data):
    obs_data = rollout_data["obs"]
    actions_data = rollout_data["actions"]
    log_probs_data = rollout_data["log_probs"]
    advantages_data = rollout_data["advantages"]
    returns_data = rollout_data["returns"]
    action_masks_data = rollout_data["action_masks"]

    advantages_data = (advantages_data - advantages_data.mean()) / (advantages_data.std() + 1e-8)

    policy_losses, value_losses, entropy_losses, approx_kl_divs = [], [], [], []

    for _ in range(ppo_epochs):
        indices = np.random.permutation(len(obs_data))
        for start in range(0, len(obs_data), batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            mb_obs_list = [obs_data[i] for i in mb_indices]
            mb_actions = actions_data[mb_indices].to(policy.device)
            mb_action_masks = action_masks_data[mb_indices].to(policy.device)

            new_log_probs, entropy, new_values = policy.evaluate_actions(
                mb_obs_list, mb_actions, mb_action_masks
            )

            mb_advantages = advantages_data[mb_indices].to(policy.device)
            mb_old_log_probs = log_probs_data[mb_indices].to(policy.device)
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)
            policy_loss1 = mb_advantages * ratio
            policy_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

            mb_returns = returns_data[mb_indices].to(policy.device)
            value_loss = F.mse_loss(new_values, mb_returns)

            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().cpu().numpy()
            approx_kl_divs.append(approx_kl)

        if np.mean(approx_kl_divs) > target_kl:
            print(f"Early stopping at epoch due to high KL divergence: {np.mean(approx_kl_divs)}")
            break

    return {
        "policy_loss": np.mean(policy_losses),
        "value_loss": np.mean(value_losses),
        "entropy_loss": np.mean(entropy_losses),
        "approx_kl": np.mean(approx_kl_divs),
    }


def main():
    checkpoint_path = "ppo_checkpoint.pt"
    env = SimEnv.build_env()

    start = load_checkpoint(checkpoint_path) or 0
    print(f"Starting training from episode {start + 1}")
    for episode in range(start, num_episodes):
        print(f"Collecting rollout for episode {episode + 1}...")

        rollout_data = []
        for _ in range(n_jobs):
            rollout_data.extend(collect_rollout(env))

        rewards = torch.stack([d["rewards"] for d in rollout_data])
        values = torch.stack([d["values"] for d in rollout_data])
        dones = torch.stack([d["dones"] for d in rollout_data])
        advantages, returns = compute_gae(rewards, values, dones)

        T, B = rewards.shape
        flat_obs = [obs for d in rollout_data for obs in d["obs"]]
        flat_actions = torch.stack([d["actions"] for d in rollout_data]).reshape(T * B, 2)
        flat_log_probs = torch.stack([d["log_probs"] for d in rollout_data]).reshape(-1)
        flat_action_masks = torch.stack([d["action_masks"] for d in rollout_data]).reshape(
            T * B, 2, ACT_SIZE
        )

        processed_rollout_data = {
            "obs": flat_obs,
            "actions": flat_actions,
            "log_probs": flat_log_probs,
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "action_masks": flat_action_masks,
        }

        stats = ppo_update(processed_rollout_data)

        print("=" * 60)
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Policy Loss: {stats['policy_loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Entropy Loss: {stats['entropy_loss']:.4f}")
        print(f"  Approx KL: {stats['approx_kl']:.4f}")
        print("=" * 60)

        if (episode + 1) % 50 == 0:
            print(f"Saving checkpoint at episode {episode + 1}")
            save_checkpoint(checkpoint_path, episode + 1)
            print("Checkpoint saved.")


if __name__ == "__main__":
    main()
