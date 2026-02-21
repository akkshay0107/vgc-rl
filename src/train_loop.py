import os
import time

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
gae_lambda = 0.98  # sparse reward counteraction
lr = 1e-4
batch_size = 64
clip_range = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
target_kl = 0.015
ppo_epochs = 4
n_jobs = 4
eval_episodes = 10
best_update_threshold = 0.55

policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
best_policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.AdamW(policy.parameters(), lr=lr, eps=1e-5)

if policy.device.type == "cuda":
    policy.compile()

if best_policy.device.type == "cuda":
    best_policy.compile()


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
    policy.eval()

    obs, _ = env.reset()
    episode_data = []

    while True:
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        action_mask_agent1 = observation_builder.get_action_mask(env.battle1)
        action_mask_agent2 = observation_builder.get_action_mask(env.battle2)

        obs_batch = torch.stack([obs_agent1, obs_agent2], dim=0).to(policy.device)
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0).to(
            policy.device
        )

        with torch.no_grad():
            _, _, action1, _ = policy(
                obs_agent1.unsqueeze(0).to(policy.device),
                action_mask_agent1.unsqueeze(0).to(policy.device),
            )
            _, _, action2, _ = best_policy(
                obs_agent2.unsqueeze(0).to(best_policy.device),
                action_mask_agent2.unsqueeze(0).to(best_policy.device),
            )

            sampled_actions = torch.cat([action1, action2], dim=0)
            log_probs, _, values = policy.evaluate_actions(
                obs_batch, sampled_actions.to(policy.device), action_mask_batch
            )

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

    for epoch_idx in range(ppo_epochs):
        # per epoch stats
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        avg_entropy_loss = 0.0
        avg_kl_div = 0.0

        minibatch_indices = np.random.permutation(n_samples)

        for minibatch_start in range(0, n_samples, batch_size):
            minibatch_end = minibatch_start + batch_size
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
                ratio, 1.0 - clip_range, 1.0 + clip_range
            )

            policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
            value_loss = F.mse_loss(new_values, mb_returns)
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            #TODO: save best policy
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = (mb_old_log_probs - new_log_probs).mean().item()

            avg_policy_loss += policy_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy_loss += entropy_loss.item()
            avg_kl_div += kl

        num_mb = (n_samples + batch_size - 1) // batch_size
        epochs_done += 1

        avg_policy_loss /= num_mb
        avg_value_loss /= num_mb
        avg_entropy_loss /= num_mb
        avg_kl_div /= num_mb

        tot_avg_policy_loss += avg_policy_loss
        tot_avg_value_loss += avg_value_loss
        tot_avg_entropy_loss += avg_entropy_loss
        tot_avg_kl_div += avg_kl_div

        if avg_kl_div > target_kl:
            print(
                f"Early stop at epoch {epoch_idx + 1}/{ppo_epochs} (KL={avg_kl_div} > {target_kl})"
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


def compare_winrate(env, policy_a, policy_b, n_episodes=eval_episodes):
    policy_a.eval()
    policy_b.eval()

    wins_a = 0
    losses_a = 0
    ties = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        while True:
            obs_agent1 = obs[env.agent1.username]
            obs_agent2 = obs[env.agent2.username]

            action_mask_agent1 = observation_builder.get_action_mask(env.battle1)
            action_mask_agent2 = observation_builder.get_action_mask(env.battle2)

            with torch.no_grad():
                _, _, action1, _ = policy_a(
                    obs_agent1.unsqueeze(0).to(policy_a.device),
                    action_mask_agent1.unsqueeze(0).to(policy_a.device),
                )
                _, _, action2, _ = policy_b(
                    obs_agent2.unsqueeze(0).to(policy_b.device),
                    action_mask_agent2.unsqueeze(0).to(policy_b.device),
                )

            actions = {
                env.agent1.username: action1[0].cpu().numpy(),
                env.agent2.username: action2[0].cpu().numpy(),
            }

            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            done1 = terminated[env.agent1.username] or truncated[env.agent1.username]
            done2 = terminated[env.agent2.username] or truncated[env.agent2.username]

            if done1 or done2:
                r1 = rewards[env.agent1.username]
                r2 = rewards[env.agent2.username]
                if r1 > r2:
                    wins_a += 1
                elif r2 > r1:
                    losses_a += 1
                else:
                    ties += 1
                break

            obs = next_obs

    total = wins_a + losses_a + ties
    win_rate = wins_a / total if total > 0 else 0.0
    return {"win_rate": win_rate, "wins": wins_a, "losses": losses_a, "ties": ties}


def main():
    checkpoint_path = "ppo_checkpoint.pt"
    best_checkpoint_path = "ppo_best_checkpoint.pt"
    env = SimEnv.build_env()

    start = load_checkpoint(checkpoint_path) or 0
    best_policy.load_state_dict(policy.state_dict())
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=best_policy.device)
        best_policy.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"Starting training from episode {start + 1}")
    for episode in range(start, num_episodes):
        print(f"Collecting rollout for episode {episode + 1}...")

        rollout_data = []
        for _ in range(n_jobs):
            rollout_data.extend(collect_rollout(env))  # should be in parallel

        rewards = torch.stack([d["rewards"] for d in rollout_data])
        values = torch.stack([d["values"] for d in rollout_data])
        dones = torch.stack([d["dones"] for d in rollout_data])
        advantages, returns = compute_gae(rewards, values, dones)

        # normalize advantages before ppo update
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T, B = rewards.shape
        flat_obs = torch.stack([d["obs"] for d in rollout_data]).reshape(T * B, *OBS_DIM)
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

        winrate_stats = compare_winrate(env, policy, best_policy)
        print(
            "Pre-update win rate "
            f"(policy vs best_policy): {winrate_stats['win_rate']:.2%} "
            f"(W/L/T: {winrate_stats['wins']}/{winrate_stats['losses']}/{winrate_stats['ties']})"
        )
        if winrate_stats["win_rate"] >= 0.5:
            best_policy.load_state_dict(policy.state_dict())
            print(f"Updated best_policy (win rate {winrate_stats['win_rate']:.2%}).")
            save_best_checkpoint(best_checkpoint_path, episode + 1, winrate_stats["win_rate"])
            print("Best policy checkpoint saved.")

        stats = ppo_update(processed_rollout_data)

        print("=" * 60)
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Policy Loss: {stats['policy_loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Entropy Loss: {stats['entropy_loss']:.4f}")
        print(f"  KL Divergence: {stats['kl_divergence']:.4f}")
        print(f"  Time taken: {stats['time']:.4f} s")
        print("=" * 60)

        if (episode + 1) % 50 == 0:
            print(f"Saving checkpoint at episode {episode + 1}")
            save_checkpoint(checkpoint_path, episode + 1)
            print("Checkpoint saved.")


if __name__ == "__main__":
    main()
