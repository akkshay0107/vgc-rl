import logging
import queue
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

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
    unwrap_policy,
)

config = PPOConfig()
_buffer_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)

shutdown_requested = False


def handle_sigterm(signum, frame):
    global shutdown_requested
    logging.warning("SIGTERM received, requesting shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.AdamW(policy.parameters(), lr=config.lr, eps=3e-5)


def lr_lambda(episode):
    if episode < config.warmup_episodes:
        return 1.0
    decay_total = config.num_episodes - config.warmup_episodes
    if decay_total <= 0:
        return 1.0
    decay_progress = (episode - config.warmup_episodes) / decay_total
    end_factor = config.min_lr / config.lr
    return max(end_factor, 1.0 - (1.0 - end_factor) * decay_progress)


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

if config.compile_policy and policy.device.type == "cuda":
    policy = cast(PolicyNet, torch.compile(policy))


def reducer_of(model: PolicyNet):
    return unwrap_policy(model).reducer


def initial_state(model: PolicyNet, batch_size: int, device: torch.device):
    reducer = reducer_of(model)
    cls = reducer.cls_base.detach().expand(batch_size, -1, -1).squeeze(1).to(device)
    hg = reducer.hg_init.detach().expand(batch_size, -1, -1).to(device)
    return cls, hg


def state_to_cpu(state):
    cls, hg = state
    return cls.detach().cpu(), hg.detach().cpu()


def cat_states(state_a, state_b):
    return torch.cat([state_a[0], state_b[0]], dim=0), torch.cat([state_a[1], state_b[1]], dim=0)


def split_state(state, idx: int):
    cls, hg = state
    return cls[idx : idx + 1], hg[idx : idx + 1]


@torch.inference_mode()
def collect_rollout(
    env,
    buffer: RolloutBuffer,
    opponent_policy: PolicyNet,
    is_self_play: bool = False,
) -> tuple[bool, bool]:
    obs, _ = env.reset()
    agent1 = env.agent1.username
    agent2 = env.agent2.username

    traj1 = []
    traj2 = []

    state1 = initial_state(policy, 1, policy.device)
    if is_self_play:
        state2 = initial_state(policy, 1, policy.device)
    else:
        state2 = initial_state(opponent_policy, 1, opponent_policy.device)

    final_rewards = None

    while True:
        obs1 = obs[agent1].unsqueeze(0).to(policy.device, non_blocking=True)
        mask1 = (
            observation_builder.get_action_mask(env.battle1)
            .unsqueeze(0)
            .to(policy.device, non_blocking=True)
        )

        # store both sides of the game if self play
        if is_self_play:
            obs2 = obs[agent2].unsqueeze(0).to(policy.device, non_blocking=True)
            mask2 = (
                observation_builder.get_action_mask(env.battle2)
                .unsqueeze(0)
                .to(policy.device, non_blocking=True)
            )

            combined_obs = torch.cat([obs1, obs2], dim=0)
            combined_masks = torch.cat([mask1, mask2], dim=0)
            combined_state = cat_states(state1, state2)

            _, combined_log_probs, combined_actions, combined_values, next_combined_state = policy(
                combined_obs,
                combined_state,
                combined_masks,
                sample_actions=True,
            )

            action1 = combined_actions[0:1]
            action2 = combined_actions[1:2]
            log_probs1 = combined_log_probs[0:1]
            log_probs2 = combined_log_probs[1:2]
            values1 = combined_values[0:1]
            values2 = combined_values[1:2]
            next_state1 = split_state(next_combined_state, 0)
            next_state2 = split_state(next_combined_state, 1)
        else:
            obs2 = obs[agent2].unsqueeze(0).to(opponent_policy.device, non_blocking=True)
            mask2 = (
                observation_builder.get_action_mask(env.battle2)
                .unsqueeze(0)
                .to(opponent_policy.device, non_blocking=True)
            )

            _, log_probs1, action1, values1, next_state1 = policy(
                obs1,
                state1,
                mask1,
                sample_actions=True,
            )
            _, _, action2, _, next_state2 = opponent_policy(
                obs2,
                state2,
                mask2,
                sample_actions=True,
            )

        actions = {
            agent1: action1[0].cpu().numpy(),
            agent2: action2[0].cpu().numpy(),
        }

        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        done = bool(
            terminated[agent1] or truncated[agent1] or terminated[agent2] or truncated[agent2]
        )

        traj1.append(
            {
                "obs": obs1.cpu(),
                "actions": action1.cpu(),
                "log_probs": log_probs1.cpu(),
                "values": values1.cpu(),
                "rewards": torch.tensor([rewards[agent1]], dtype=torch.float32),
                "dones": torch.tensor([done], dtype=torch.float32),
                "action_masks": mask1.cpu(),
            }
        )

        if is_self_play:
            traj2.append(
                {
                    "obs": obs2.cpu(),
                    "actions": action2.cpu(),
                    "log_probs": log_probs2.cpu(),
                    "values": values2.cpu(),
                    "rewards": torch.tensor([rewards[agent2]], dtype=torch.float32),
                    "dones": torch.tensor([done], dtype=torch.float32),
                    "action_masks": mask2.cpu(),
                }
            )

        if done or shutdown_requested:
            final_rewards = rewards
            break

        obs = next_obs
        state1 = next_state1
        state2 = next_state2

    if shutdown_requested:
        # If interrupted, we don't return a valid win/loss as the game didn't finish.
        return False, is_self_play

    with _buffer_lock:
        buffer.add_episode(traj1)
        if is_self_play:
            buffer.add_episode(traj2)

    # winner gets final reward = +1, other gets -1
    return bool(final_rewards[agent1] > final_rewards[agent2]), is_self_play


def collect_all_rollouts(envs, buffer: RolloutBuffer, executor, pool: OpponentPool):
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
            won, was_self_play = collect_rollout(
                env,
                buffer,
                opponent_policy,
                is_self_play=(opponent_id == "latest"),
            )
            return opponent_id, won, was_self_play
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
        opp_id, won, was_self_play = f.result()
        if was_self_play:
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


def _run_episode_ppo(ep: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, float], int]:
    """
    Run one PPO episode with BPTT. Carries recurrent state step-to-step.
    Returns:
        total_loss: summed loss over all steps (scalar tensor, grad attached)
        step_metrics: dict of summed per-step scalar metrics for logging
        T: number of steps in the episode
    """
    obs = ep["obs"].to(device, non_blocking=True)
    actions = ep["actions"].to(device, non_blocking=True)
    old_log_probs = ep["log_probs"].to(device, non_blocking=True)
    advantages = ep["advantages"].to(device, non_blocking=True)
    returns = ep["returns"].to(device, non_blocking=True)
    action_masks = ep["action_masks"].to(device, non_blocking=True)

    T = obs.shape[0]
    state = initial_state(policy, 1, device)

    total_loss = torch.tensor(0.0, device=device)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "kl_div": 0.0}

    for t in range(T):
        curr_log_prob, curr_entropy, curr_val, state = policy.evaluate_actions(
            obs[t : t + 1],
            actions[t : t + 1],
            action_masks[t : t + 1],
            state=state,
        )

        log_ratio = curr_log_prob - old_log_probs[t : t + 1]
        ratio = torch.exp(log_ratio)

        mb_adv = advantages[t : t + 1]
        surr1 = ratio * mb_adv
        surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * mb_adv

        step_policy_loss = -torch.min(surr1, surr2).mean()
        step_value_loss = F.mse_loss(curr_val, returns[t : t + 1])
        step_entropy_loss = -curr_entropy.mean()

        step_loss = (
            step_policy_loss
            + config.value_coef * step_value_loss
            + config.entropy_coef * step_entropy_loss
        )
        total_loss = total_loss + step_loss

        with torch.no_grad():
            metrics["policy_loss"] += step_policy_loss.item()
            metrics["value_loss"] += step_value_loss.item()
            metrics["entropy_loss"] += step_entropy_loss.item()
            metrics["kl_div"] += (old_log_probs[t : t + 1] - curr_log_prob).mean().item()

    return total_loss, metrics, T


def ppo_update(episodes: list) -> dict:
    policy.train()
    t0 = time.time()

    tot_policy_loss = 0.0
    tot_value_loss = 0.0
    tot_entropy_loss = 0.0
    tot_kl_div = 0.0
    tot_steps = 0
    epochs_done = 0

    for epoch_idx in range(config.ppo_epochs):
        if shutdown_requested:
            break
        random.shuffle(episodes)

        epoch_steps = 0
        epoch_kl = 0.0

        for batch_start in range(0, len(episodes), config.batch_size):
            if shutdown_requested:
                break
            batch = episodes[batch_start : batch_start + config.batch_size]

            optimizer.zero_grad(set_to_none=True)

            # Accumulate loss and graphs for all episodes in batch before backprop,
            # normalizing by total steps across the batch (same scheme as BC).
            batch_loss = torch.tensor(0.0, device=policy.device)
            batch_steps = 0

            for ep in batch:
                ep_loss, ep_metrics, T = _run_episode_ppo(ep, policy.device)
                batch_loss = batch_loss + ep_loss
                batch_steps += T

                tot_policy_loss += ep_metrics["policy_loss"]
                tot_value_loss += ep_metrics["value_loss"]
                tot_entropy_loss += ep_metrics["entropy_loss"]
                epoch_kl += ep_metrics["kl_div"]

            if batch_steps > 0:
                (batch_loss / batch_steps).backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

            epoch_steps += batch_steps

        tot_steps += epoch_steps
        tot_kl_div += epoch_kl
        epochs_done += 1

        if epoch_steps > 0:
            avg_kl = epoch_kl / epoch_steps
            if avg_kl > config.target_kl:
                logging.info(
                    f"Early stop at epoch {epoch_idx + 1}/{config.ppo_epochs} "
                    f"(KL={avg_kl:.4f} > {config.target_kl:.4f})"
                )
                break

    if epochs_done == 0 or tot_steps == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "kl_divergence": 0.0,
            "time": time.time() - t0,
        }

    return {
        "policy_loss": tot_policy_loss / tot_steps,
        "value_loss": tot_value_loss / tot_steps,
        "entropy_loss": tot_entropy_loss / tot_steps,
        "kl_divergence": tot_kl_div / tot_steps,
        "time": time.time() - t0,
    }


def main():
    envs = [SimEnv.build_env(env_id=i) for i in range(config.n_jobs)]
    buffer = RolloutBuffer()
    executor = ThreadPoolExecutor(max_workers=config.n_jobs)

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.pool_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir="runs/ppo_training")

    # to guarantee executor shutdown
    try:
        start = load_checkpoint(config.checkpoint_path, policy, optimizer, scheduler)

        if start is not None:
            logging.info(f"Resuming training from episode {start + 1}")
        else:
            seed_path = config.pool_dir / "seed_mixed.pt"
            if seed_path.exists():
                logging.info(f"No checkpoint found. Seeding policy from {seed_path}")
                load_checkpoint(seed_path, policy)
            else:
                logging.info(
                    "No checkpoint or seed policy found. Starting from random initialization."
                )
            start = 0

        pool = OpponentPool.load_or_create(config.pool_dir, config)
        if len(pool) == 0:
            logging.info("Opponent pool empty, seeding with current policy as ep0")
            pool.add(policy, "ep0")
            pool.save_state()

        logging.info(f"Opponent pool: {pool}")

        for episode in range(start, config.num_episodes):
            if shutdown_requested:
                logging.warning("Shutdown requested, saving checkpoint and exiting")
                save_checkpoint(config.checkpoint_path, episode, policy, optimizer, scheduler)
                break

            buffer.reset()
            policy.eval()

            t0_rollout = time.time()
            rollout_stats = collect_all_rollouts(envs, buffer, executor, pool)
            rollout_time = time.time() - t0_rollout

            if not buffer.trajectories:
                logging.warning("No trajectories collected, skipping update")
                continue

            rollout_data = buffer.get_batches(policy.device, config)
            stats = ppo_update(rollout_data)
            scheduler.step()

            if (episode + 1) % config.snapshot_interval == 0:
                snap_id = f"ep{episode + 1}"
                pool.add(policy, snap_id)
                pool.save_state()
                logging.info(f"Snapshot '{snap_id}' added to opponent pool. Pool: {pool}")

            current_lr = scheduler.get_last_lr()[0]
            tb_writer.add_scalar("WinRate/Pool", rollout_stats["pool_win_rate"], episode + 1)
            tb_writer.add_scalar("WinRate/Self", rollout_stats["self_win_rate"], episode + 1)
            tb_writer.add_scalar("Loss/Policy", stats["policy_loss"], episode + 1)
            tb_writer.add_scalar("Loss/Value", stats["value_loss"], episode + 1)
            tb_writer.add_scalar("Loss/Entropy", stats["entropy_loss"], episode + 1)
            tb_writer.add_scalar("Training/KL_Divergence", stats["kl_divergence"], episode + 1)
            tb_writer.add_scalar("Training/LearningRate", current_lr, episode + 1)
            tb_writer.add_scalar("Timing/Rollout", rollout_time, episode + 1)
            tb_writer.add_scalar("Timing/Update", stats["time"], episode + 1)
            tb_writer.add_scalar("Buffer/NumTrajectories", len(buffer.trajectories), episode + 1)

            logging.info(
                f"Episode {episode + 1}/{config.num_episodes} | "
                f"Pool WR: {rollout_stats['pool_win_rate']:.2%} | "
                f"Self WR: {rollout_stats['self_win_rate']:.2%} | "
                f"Rollout: {rollout_time:.2f}s | "
                f"Policy Loss: {stats['policy_loss']:.4f} | "
                f"KL: {stats['kl_divergence']:.4f}"
            )

            if (episode + 1) % 10 == 0:
                save_checkpoint(config.checkpoint_path, episode + 1, policy, optimizer, scheduler)
                logging.info("Checkpoint saved.")

    finally:
        executor.shutdown(wait=True)
        tb_writer.close()
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        logging.info("Training loop terminated successfully.")


if __name__ == "__main__":
    main()
