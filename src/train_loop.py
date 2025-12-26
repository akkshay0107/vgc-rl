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
num_episodes = 100  # num of times entire loop (PPO + Aux phase) takes place
gamma = 0.95  # avg vgc game length = 8 turns (0.95)^8 ~ 0.66
gae_lambda = 0.95
lr = 1e-4
n_policy_iterations = 32  # Number of PPO iters before the Aux phase
n_aux_epochs = 6  # Number of minibatches sampled in the Aux phase
ppo_epochs = 1  # Number of minibatches sampled per PPO iter
batch_size = 64
clip_range = 0.2
entropy_coef = 0.01
value_coef = 0.5
beta_clone = 1.0  # weight given to KL divergence term for Aux phase
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
                ),  # (2,)
                "dones": torch.tensor([done1, done2], dtype=torch.float32),  # (2,)
                "action_masks": action_mask_batch,
            }
        )

        obs = next_obs
        if done1 or done2:
            break

    return episode_data


# TODO: parallelize the policy phase
def policy_phase():
    episode_data = collect_rollout()

    def flat_collect(by):
        res = torch.stack([d[by] for d in episode_data])
        if len(res.shape) <= 2:
            return res.view(-1)
        return res.view(-1, *res.shape[2:])

    # all below of shape (turns, 2, *)
    values_list = torch.stack([d["values"] for d in episode_data])
    rewards_list = torch.stack([d["rewards"] for d in episode_data])
    dones_list = torch.stack([d["dones"] for d in episode_data])

    advantages, returns = compute_gae(
        rewards=rewards_list,
        values=values_list,
        dones=dones_list,
    )  # advantages, returns: (turns, 2)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    flat_obs = flat_collect("obs")
    flat_act = flat_collect("actions")
    flat_masks = flat_collect("action_masks")
    flat_old_log_prob = flat_collect("log_probs")
    flat_adv = advantages.view(-1)
    flat_ret = returns.view(-1)
    with torch.no_grad():
        policy_log_probs = policy.get_policy_log_probs(flat_obs, flat_act, flat_masks)

    # Filling out replay buffer
    sz = flat_obs.shape[0]
    for t in range(sz):
        replay_buffer.append(
            {
                "obs": flat_obs[t],
                "action": flat_act[t],
                "action_mask": flat_masks[t],
                "returns": flat_ret[t],
                "old_policy_log_probs": policy_log_probs[t].detach(),
            }
        )

    # Sample minibatches and train on minibatch
    for _ in range(ppo_epochs):
        indices = torch.randperm(sz)
        for start in range(0, sz, batch_size):
            minibatch = indices[start : start + batch_size]
            mb_obs = flat_obs[minibatch]
            mb_act = flat_act[minibatch]
            mb_old_log_prob = flat_old_log_prob[minibatch]
            mb_mask = flat_masks[minibatch]
            mb_adv = flat_adv[minibatch]
            mb_ret = flat_ret[minibatch]

            new_log_prob, entropy, values = policy.evaluate_actions(
                mb_obs, mb_act, mb_mask
            )  # add entropy term later

            ratio = (new_log_prob - mb_old_log_prob).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()  # maximize clip objective

            value_loss = F.mse_loss(values.squeeze(-1), mb_ret)

            joint_loss = (
                policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            )  # -ve for entropy since we want to maximize entropy term

            optimizer.zero_grad()
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

    return rewards_list.sum().item()


def auxiliary_phase():
    if len(replay_buffer) < batch_size:
        return

    for _ in range(n_aux_epochs):
        indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]

        obs_batch = torch.stack([b["obs"] for b in batch])
        act_batch = torch.stack([b["action"] for b in batch])
        mask_batch = torch.stack([b["action_mask"] for b in batch])
        returns_batch = torch.stack([b["returns"] for b in batch])
        old_policy_log_probs_batch = torch.stack([b["old_policy_log_probs"] for b in batch])

        # Auxiliary value loss (aux head)
        aux_values = policy.get_aux_value(obs_batch)
        aux_value_loss = F.mse_loss(aux_values.squeeze(-1), returns_batch)

        # KL divergence loss KL(old || new)
        current_policy_log_probs = policy.get_policy_log_probs(obs_batch, act_batch, mask_batch)
        kl1 = F.kl_div(
            current_policy_log_probs[:, 0],  # log(new) (y_pred)
            old_policy_log_probs_batch[:, 0],  # old (y_true)
            reduction="batchmean",
            log_target=True,
        )

        kl2 = F.kl_div(
            current_policy_log_probs[:, 1],  # log(new) (y_pred)
            old_policy_log_probs_batch[:, 1],  # old (y_true)
            reduction="batchmean",
            log_target=True,
        )

        joint_loss = aux_value_loss + beta_clone * (kl1 + kl2)

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

        print(
            f"Episode {episode + 1}/{num_episodes}, reward: {avg_rew:.3f}, buffer: {len(replay_buffer)}"
        )

        if (episode + 1) % 10 == 0:
            save_checkpoint(f"./checkpoints/checkpoint_{episode + 1}.pt", episode)
            save_checkpoint("./checkpoints/checkpoint_latest.pt", episode)
            print(f"Checkpoint saved at episode {episode + 1}")


if __name__ == "__main__":
    main()
