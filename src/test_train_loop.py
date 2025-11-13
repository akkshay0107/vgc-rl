from math import prod
from torch.optim import Adam

from encoder import ACT_SIZE, BATTLE_STATE_DIMS
from env import SimEnv
from pseudo_policy import PseudoPolicy
from rl_player import RLPlayer

# Parameters
num_episodes = 1000
max_steps_per_episode = 100
learning_rate = 1e-3
gamma = 0.99

env = SimEnv()
policy = PseudoPolicy(observation_dim=prod(BATTLE_STATE_DIMS), action_dim=ACT_SIZE)
player1 = RLPlayer(policy=policy)
player2 = RLPlayer(policy=policy)

optimizer = Adam(policy.parameters(), lr=learning_rate)


# -- THIS CODE IS JUST TO TEST THAT THE ENVIRONMENT WORKS --
# -- IT IS NOT AN ACTUAL TRAINING LOOP --
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


for episode in range(num_episodes):
    obs_dict, _ = env.reset()
    episode_rewards = []
    episode_log_probs = []

    for step in range(max_steps_per_episode):
        actions = {}
        # Use RLPlayer to choose moves given each internal battle state
        actions[env.agent1.username] = player1._get_action(env.battle1)  # type: ignore
        actions[env.agent2.username] = player2._get_action(env.battle2)  # type: ignore

        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards.append(rewards[env.agent1.username])
        if terminated[env.agent1.username] or terminated[env.agent2.username]:
            break

    discounted_rewards = compute_discounted_rewards(episode_rewards, gamma)
    print(f"Episode {episode + 1}/{num_episodes} finished, reward: {sum(episode_rewards):.3f}")
