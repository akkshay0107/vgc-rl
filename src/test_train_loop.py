from math import prod
import torch

from encoder import ACT_SIZE, BATTLE_STATE_DIMS, Encoder
from env import SimEnv
from pseudo_policy import PseudoPolicy

num_episodes = 1000
max_steps_per_episode = 100
gamma = 0.99

env = SimEnv()
policy = PseudoPolicy(observation_dim=prod(BATTLE_STATE_DIMS), action_dim=ACT_SIZE)


def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_rewards = []

    for step in range(max_steps_per_episode):
        actions = {}

        # Get observation tensors for both agents (convert to float tensor)
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        action_mask_agent1 = Encoder.get_action_mask(env.battle1)  # type: ignore
        action_mask_agent2 = Encoder.get_action_mask(env.battle2)  # type: ignore

        # Stack along batch dim for batch processing by policy network
        obs_batch = torch.stack([obs_agent1, obs_agent2], dim=0)  # shape (2, *obs_shape)
        action_mask_batch = torch.stack([action_mask_agent1, action_mask_agent2], dim=0)

        # print(obs_batch.shape)
        # print(action_mask_batch.shape)

        with torch.no_grad():
            sampled_actions = policy.forward(obs_batch, action_mask_batch)

        actions[env.agent1.username] = sampled_actions[0]
        actions[env.agent2.username] = sampled_actions[1]

        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards.append(rewards[env.agent1.username])

        if terminated[env.agent1.username] or terminated[env.agent2.username]:
            break

    discounted_rewards = compute_discounted_rewards(episode_rewards, gamma)
    print(f"Episode {episode + 1}/{num_episodes} finished, reward: {sum(episode_rewards):.3f}")
