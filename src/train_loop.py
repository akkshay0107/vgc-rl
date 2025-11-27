import os
import sys
from math import prod
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from encoder import ACT_SIZE, OBS_DIM, Encoder
from env import SimEnv
from policy import PolicyNet

NUM_EPISODES = 2500
MAX_STEPS_PER_EPISODE = 200
GAMMA = 0.99
LR = 3e-4
VALUE_COEF = 0.5
GRAD_CLIP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_discounted_rewards(rewards: List[float], gamma: float) -> List[float]:
    discounted = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted


def to_tensor(x, dtype=torch.float32, device=DEVICE):
    return torch.tensor(x, dtype=dtype, device=device)


env = SimEnv.build_env()
policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE).to(DEVICE)
optimizer = Adam(policy.parameters(), lr=LR)

agent_usernames = (env.agent1.username, env.agent2.username)

print("Device:", DEVICE)
print("Action size:", ACT_SIZE)
print("Obs dim:", OBS_DIM)


for episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    traj = {
        agent_usernames[0]: {
            "obs": [],
            "masks": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
        },
        agent_usernames[1]: {
            "obs": [],
            "masks": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
        },
    }

    total_reward_agent0 = 0.0
    total_reward_agent1 = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        obs_agent1 = obs[agent_usernames[0]]
        obs_agent2 = obs[agent_usernames[1]]

        if not isinstance(obs_agent1, torch.Tensor):
            obs_agent1 = torch.tensor(obs_agent1, dtype=torch.float32)
        if not isinstance(obs_agent2, torch.Tensor):
            obs_agent2 = torch.tensor(obs_agent2, dtype=torch.float32)

        obs_batch = torch.stack([obs_agent1.to(DEVICE), obs_agent2.to(DEVICE)], dim=0)

        mask_a1 = Encoder.get_action_mask(env.battle1)  # type: ignore
        mask_a2 = Encoder.get_action_mask(env.battle2)  # type: ignore

        action_mask_batch = torch.stack([mask_a1, mask_a2], dim=0).to(DEVICE)

        policy_logits, values = policy(obs_batch, action_mask=action_mask_batch)

        logits_pos0 = policy_logits[:, 0, :]
        logits_pos1 = policy_logits[:, 1, :]

        dist0 = Categorical(logits=logits_pos0)
        dist1 = Categorical(logits=logits_pos1)

        sampled0 = dist0.sample()
        sampled1 = dist1.sample()

        logp0 = dist0.log_prob(sampled0)
        logp1 = dist1.log_prob(sampled1)

        actions_for_env = {}
        for i, username in enumerate(agent_usernames):
            action_pair = np.array(
                [int(sampled0[i].item()), int(sampled1[i].item())], dtype=np.int64
            )
            actions_for_env[username] = action_pair

            traj[username]["obs"].append(obs_batch[i].detach().cpu())
            traj[username]["masks"].append(action_mask_batch[i].detach().cpu())
            traj[username]["actions"].append(action_pair)
            traj[username]["log_probs"].append(logp0[i] + logp1[i])
            traj[username]["values"].append(values[i])

        obs, rewards, terminated, truncated, info = env.step(actions_for_env)

        r0 = float(rewards[agent_usernames[0]])
        r1 = float(rewards[agent_usernames[1]])
        traj[agent_usernames[0]]["rewards"].append(r0)
        traj[agent_usernames[1]]["rewards"].append(r1)

        total_reward_agent0 += r0
        total_reward_agent1 += r1

        done0 = terminated[agent_usernames[0]] or truncated[agent_usernames[0]]
        done1 = terminated[agent_usernames[1]] or truncated[agent_usernames[1]]
        if done0 or done1:
            break

    returns0 = compute_discounted_rewards(traj[agent_usernames[0]]["rewards"], GAMMA)
    returns1 = compute_discounted_rewards(traj[agent_usernames[1]]["rewards"], GAMMA)

    all_log_probs = []
    all_values = []
    all_returns = []

    for returns, username in zip([returns0, returns1], agent_usernames):
        ret = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        lp = torch.stack(traj[username]["log_probs"]).to(DEVICE)
        vals = torch.stack(traj[username]["values"]).to(DEVICE)

        all_log_probs.append(lp)
        all_values.append(vals)
        all_returns.append(ret)

    logp_batch = torch.cat(all_log_probs, dim=0)
    values_batch = torch.cat(all_values, dim=0)
    returns_batch = torch.cat(all_returns, dim=0)

    advantages = returns_batch - values_batch

    policy_loss = -(logp_batch * advantages.detach()).mean()
    value_loss = F.mse_loss(values_batch, returns_batch)
    loss = policy_loss + VALUE_COEF * value_loss

    logp_batch = torch.cat(all_log_probs, dim=0)
    values_batch = torch.cat(all_values, dim=0)
    returns_batch = torch.cat(all_returns, dim=0)

    advantages = returns_batch - values_batch

    policy_loss = -(logp_batch * advantages.detach()).mean()

    value_loss = F.mse_loss(values_batch, returns_batch)

    loss = policy_loss + VALUE_COEF * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    optimizer.step()

    avg_return = (
        sum(traj[agent_usernames[0]]["rewards"]) + sum(traj[agent_usernames[1]]["rewards"])
    ) / 2.0
    print(
        f"Episode {episode}/{NUM_EPISODES} | Step {len(traj[agent_usernames[0]]['rewards'])} "
        f"| AvgReward {avg_return:.3f} | PolicyLoss {policy_loss.item():.4f} | ValueLoss {value_loss.item():.4f}"
    )

print("done")
