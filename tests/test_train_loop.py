import os

# Adding src to path
import sys
from math import prod

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pseudo_policy import PseudoPolicy

from encoder import ACT_SIZE, OBS_DIM, Encoder
from env import SimEnv

num_episodes = 2500
max_steps_per_episode = 100
gamma = 0.99

env = SimEnv.build_env()
policy = PseudoPolicy(observation_dim=prod(OBS_DIM), action_dim=ACT_SIZE)


def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


EMBEDDING_TEST = 1  # change to zero to test policy
if EMBEDDING_TEST:
    num_episodes = 10

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_rewards = []

    for step in range(max_steps_per_episode):
        actions = {}

        # Get observation tensors for both agents (convert to float tensor)
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        if EMBEDDING_TEST:
            embedding = Encoder.encode_battle_state(env.battle1)  # type: ignore
            print(
                embedding.shape
            )  # should be (11, 650)
            break

        action_mask_agent1 = Encoder.get_action_mask(env.battle1)  # type: ignore
        action_mask_agent2 = Encoder.get_action_mask(env.battle2)  # type: ignore

        # Stack along batch dim for batch processing by policy network
        obs_batch = torch.stack([obs_agent1, obs_agent2], dim=0)  # shape (2, *obs_shape)
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0)

        # print(obs_batch.shape)
        # print(action_mask_batch.shape)

        with torch.no_grad():
            sampled_actions, _, _ = policy.forward(obs_batch, action_mask_batch)

        actions[env.agent1.username] = sampled_actions[0]
        actions[env.agent2.username] = sampled_actions[1]

        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards.append(rewards[env.agent1.username])

        if (
            terminated[env.agent1.username]
            or terminated[env.agent2.username]
            or truncated[env.agent1.username]
            or truncated[env.agent2.username]
        ):
            break

    discounted_rewards = compute_discounted_rewards(episode_rewards, gamma)
    print(f"Episode {episode + 1}/{num_episodes} finished, reward: {sum(episode_rewards):.3f}")
