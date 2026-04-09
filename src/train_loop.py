import logging
import queue
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import observation_builder
from env import SimEnv
from lookups import ACT_SIZE, OBS_DIM
from policy import PolicyNet
from ppo_utils import (
    OpponentPool,
    PPOConfig,
    RolloutBuffer,
    load_checkpoint,
    save_checkpoint,
)

config = PPOConfig()
_buffer_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training_continuous.log"), logging.StreamHandler(sys.stdout)],
)

shutdown_requested = False


def handle_sigterm(signum, frame):
    """Signal handler for graceful shutdown on SIGTERM from SLURM."""
    global shutdown_requested
    logging.warning("SIGTERM received, requesting shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)  # Also handle Ctrl+C for local runs

policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.AdamW(policy.parameters(), lr=config.lr, eps=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_episodes, eta_min=1e-6)

if policy.device.type == "cuda":
    policy = cast(PolicyNet, torch.compile(policy))


def collect_rollout(env, buffer, opponent_policy: PolicyNet, is_self_play: bool = False) -> bool:
    """Run one full battle episode.

    Returns True if the training agent (agent1) won, False otherwise.
    """
    obs, _ = env.reset()
    local_transitions = []
    agent1_won = False

    while True:
        obs_agent1 = obs[env.agent1.username]
        obs_agent2 = obs[env.agent2.username]

        mask1 = observation_builder.get_action_mask(env.battle1).unsqueeze(0).to(policy.device)
        mask2 = observation_builder.get_action_mask(env.battle2).unsqueeze(0).to(policy.device)

        obs_batch1 = obs_agent1.unsqueeze(0).to(policy.device)
        obs_batch2 = obs_agent2.unsqueeze(0).to(policy.device)

        with torch.no_grad():
            if is_self_play:
                # Batch both agents together for efficiency
                combined_obs = torch.cat([obs_batch1, obs_batch2], dim=0)
                combined_masks = torch.cat([mask1, mask2], dim=0)
                _, combined_log_probs, combined_actions, combined_values = policy(
                    combined_obs, combined_masks
                )
                log_probs1, log_probs2 = combined_log_probs[0:1], combined_log_probs[1:2]
                action1, action2 = combined_actions[0:1], combined_actions[1:2]
                values1, values2 = combined_values[0:1], combined_values[1:2]
            else:
                _, log_probs1, action1, values1 = policy(obs_batch1, mask1)
                _, _, action2, _ = opponent_policy(
                    obs_agent2.unsqueeze(0).to(opponent_policy.device), mask2
                )

        action1_np = action1[0].cpu().numpy()
        action2_np = action2[0].cpu().numpy()

        actions = {
            env.agent1.username: action1_np,
            env.agent2.username: action2_np,
        }

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done1 = terminated[env.agent1.username] or truncated[env.agent1.username]
        done2 = terminated[env.agent2.username] or truncated[env.agent2.username]

        reward1 = rewards[env.agent1.username]
        local_transitions.append(
            {
                "obs": obs_batch1.cpu(),
                "actions": action1.cpu(),
                "log_probs": log_probs1.cpu(),
                "values": values1.cpu(),
                "rewards": torch.tensor([reward1], dtype=torch.float32),
                "dones": torch.tensor([done1], dtype=torch.float32),
                "action_masks": mask1.cpu(),
            }
        )

        if is_self_play:
            reward2 = rewards[env.agent2.username]
            local_transitions.append(
                {
                    "obs": obs_batch2.cpu(),
                    "actions": action2.cpu(),
                    "log_probs": log_probs2.cpu(),
                    "values": values2.cpu(),
                    "rewards": torch.tensor([reward2], dtype=torch.float32),
                    "dones": torch.tensor([done2], dtype=torch.float32),
                    "action_masks": mask2.cpu(),
                }
            )

        if done1 or done2:
            agent1_won = reward1 > 0
            break

        obs = next_obs

    with _buffer_lock:
        for t in local_transitions:
            buffer.add(t)

    return agent1_won


def collect_all_rollouts(envs, buffer, executor, pool: OpponentPool):
    """
    Collect config.rollouts_per_episode rollouts in parallel.
    Each rollout samples a (potentially different) opponent from the pool.
    """
    env_queue = queue.Queue()
    for env in envs:
        env_queue.put(env)

    # Pre-sample one opponent per rollout slot.
    sampled_opponents = []
    for _ in range(config.rollouts_per_episode):
        if random.random() < config.self_play_prob:
            sampled_opponents.append((policy, "latest"))
        else:
            sampled_opponents.append(pool.sample())

    def worker(opponent_policy: PolicyNet, opponent_id: str):
        env = env_queue.get()
        try:
            is_self_play = opponent_id == "latest"
            won = collect_rollout(env, buffer, opponent_policy, is_self_play=is_self_play)
            return opponent_id, won
        finally:
            env_queue.put(env)

    pool_wins = 0
    pool_total = 0
    self_wins = 0
    self_total = 0

    futures = [
        executor.submit(worker, opp_policy, opp_id) for opp_policy, opp_id in sampled_opponents
    ]
    for f in as_completed(futures):
        opp_id, won = f.result()
        if opp_id == "latest":
            self_wins += int(won)
            self_total += 1
        else:
            pool_wins += int(won)
            pool_total += 1
            pool.update_win_rate(opp_id, won)

    return {
        "pool_win_rate": pool_wins / pool_total if pool_total > 0 else 0.0,
        "self_win_rate": self_wins / self_total if self_total > 0 else 0.0,
    }


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

    for epoch_idx in range(config.ppo_epochs):
        # per epoch stats
        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        avg_entropy_loss = 0.0
        avg_kl_div = 0.0

        minibatch_indices = np.random.permutation(n_samples)

        for minibatch_start in range(0, n_samples, config.batch_size):
            minibatch_end = minibatch_start + config.batch_size
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
                ratio, 1.0 - config.clip_range, 1.0 + config.clip_range
            )

            policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
            value_loss = F.mse_loss(new_values, mb_returns)
            entropy_loss = -entropy.mean()

            total_loss = (
                policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = (mb_old_log_probs - new_log_probs).mean().item()

            avg_policy_loss += policy_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy_loss += entropy_loss.item()
            avg_kl_div += kl

        num_mb = (n_samples + config.batch_size - 1) // config.batch_size
        epochs_done += 1

        avg_policy_loss /= num_mb
        avg_value_loss /= num_mb
        avg_entropy_loss /= num_mb
        avg_kl_div /= num_mb

        tot_avg_policy_loss += avg_policy_loss
        tot_avg_value_loss += avg_value_loss
        tot_avg_entropy_loss += avg_entropy_loss
        tot_avg_kl_div += avg_kl_div

        if avg_kl_div > config.target_kl:
            logging.info(
                f"Early stop at epoch {epoch_idx + 1}/{config.ppo_epochs} (KL={avg_kl_div:.4f} > {config.target_kl})"
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


def main():
    envs = [SimEnv.build_env(env_id=i) for i in range(config.n_jobs)]
    buffer = RolloutBuffer()
    executor = ThreadPoolExecutor(max_workers=config.n_jobs)

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.pool_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir="runs/ppo_training_1")

    # to guarantee executor shutdown
    try:
        start = load_checkpoint(config.checkpoint_path, policy, optimizer, scheduler) or 0

        pool = OpponentPool.load_or_create(config.pool_dir, config)
        if len(pool) == 0:
            logging.info(
                "Opponent pool is empty. Please run `uv run python src/seed_pool.py` first to create the initial seeds."
            )
            logging.info("For now, adding a random policy as 'ep0' so training can proceed.")
            pool.add(policy, "ep0")
            pool.save_state()

        logging.info(f"Starting training from episode {start + 1}")
        logging.info(f"Opponent pool: {pool}")

        for episode in range(start, config.num_episodes):
            if shutdown_requested:
                logging.warning("Shutdown requested, stopping training loop...")
                save_checkpoint(config.checkpoint_path, episode, policy, optimizer, scheduler)
                break

            logging.info(f"Collecting rollout for episode {episode + 1}")

            t0_rollout = time.time()
            buffer.reset()

            policy.eval()
            rollout_stats = collect_all_rollouts(envs, buffer, executor, pool)

            rollout_time = time.time() - t0_rollout
            processed_rollout_data = buffer.get_batches(policy.device, config)

            # Snapshot the current policy into the pool periodically.
            if (episode + 1) % config.snapshot_interval == 0:
                snap_id = f"ep{episode + 1}"
                pool.add(policy, snap_id)
                pool.save_state()
                logging.info(f"Snapshot '{snap_id}' added to opponent pool. Pool: {pool}")

            stats = ppo_update(processed_rollout_data)
            scheduler.step()
            del processed_rollout_data

            current_lr = scheduler.get_last_lr()[0]
            tb_writer.add_scalar("WinRate/Pool", rollout_stats["pool_win_rate"], episode + 1)
            tb_writer.add_scalar("WinRate/Self", rollout_stats["self_win_rate"], episode + 1)
            tb_writer.add_scalar("Loss/Policy", stats["policy_loss"], episode + 1)
            tb_writer.add_scalar("Loss/Value", stats["value_loss"], episode + 1)
            tb_writer.add_scalar("Loss/Entropy", stats["entropy_loss"], episode + 1)
            tb_writer.add_scalar("Training/KL_Divergence", stats["kl_divergence"], episode + 1)
            tb_writer.add_scalar("Training/LearningRate", current_lr, episode + 1)

            logging.info(
                f"Episode {episode + 1}/{config.num_episodes} | "
                f"Win Rate (Pool): {rollout_stats['pool_win_rate']:.2%} | "
                f"Rollout Time: {rollout_time:.2f}s | "
                f"Policy Loss: {stats['policy_loss']:.4f} | "
                f"KL: {stats['kl_divergence']:.4f}"
            )

            """
            # Log per-opponent win-rates every 10 episodes.
            if (episode + 1) % 10 == 0:
                print("  Opponent win-rates (training policy vs pool):")
                for oid, wr in pool.win_rates.items():
                    print(f"    {oid}: {wr:.2%}")

            print("=" * 60)
            print(f"Episode {episode + 1}/{config.num_episodes}:")
            print(
                f"  Win Rate: {winrate_stats['win_rate']:.2%} (W/L: {winrate_stats['wins']}/{winrate_stats['losses']})"
            )
            print(f"  Rollout Time: {rollout_time:.4f} s")
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")
            print(f"  Value Loss: {stats['value_loss']:.4f}")
            print(f"  Entropy Loss: {stats['entropy_loss']:.4f}")
            print(f"  KL Divergence: {stats['kl_divergence']:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            print(f"  Update Time: {stats['time']:.4f} s")
            print("=" * 60)
            """

            if (episode + 1) % 10 == 0:
                logging.info(f"Saving checkpoint at episode {episode + 1}")
                save_checkpoint(config.checkpoint_path, episode + 1, policy, optimizer, scheduler)
                logging.info("Checkpoint saved.")
    finally:
        executor.shutdown(wait=True)
        tb_writer.close()
        logging.info("Training loop terminated successfully.")


if __name__ == "__main__":
    main()
