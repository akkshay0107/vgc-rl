import sys
from pathlib import Path

import torch

from behaviour_cloning import ReplayDataset, train_behavior_cloning
from ppo_utils import OpponentPool, PPOConfig


def _get_dataset(replays_base: Path, subdir: str) -> ReplayDataset | None:
    path = replays_base / subdir
    if not path.exists() or not list(path.rglob("*.replay")):
        print(f"No replays found in {path}. Skipping.")
        return None
    return ReplayDataset(str(path))


def main():
    root_dir = Path(__file__).resolve().parent.parent
    replays_base = root_dir / "replays"

    if not replays_base.exists():
        print(f"Replays directory {replays_base} does not exist. Run replay_gen.py first.")
        sys.exit(1)

    config = PPOConfig()
    pool = OpponentPool.load_or_create(config.pool_dir, config)
    added_seeds = []

    bc_kwargs = {
        "batch_size": 128,
        "num_epochs": 15,
        "learning_rate": 5e-4,
        "val_split_ratio": 0.1,
    }

    print("\n" + "=" * 60)
    print("Seeding pool with 4 requested policies via Behavior Cloning")
    print("=" * 60)

    # 1. Max Base Power
    if "seed_max_base_power" not in pool.opponent_ids:
        print("\n--- Training seed_max_base_power ---")
        ds_mbp = _get_dataset(replays_base, "max_base_power")
        if ds_mbp:
            policy = train_behavior_cloning(ds_mbp, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_max_base_power")
                added_seeds.append("seed_max_base_power")
    else:
        print("\nseed_max_base_power already exists.")

    # 2. Simple Heuristic
    if "seed_simple_heuristic" not in pool.opponent_ids:
        print("\n--- Training seed_simple_heuristic ---")
        ds_sh = _get_dataset(replays_base, "simple_heuristic")
        if ds_sh:
            policy = train_behavior_cloning(ds_sh, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_simple_heuristic")
                added_seeds.append("seed_simple_heuristic")
    else:
        print("\nseed_simple_heuristic already exists.")

    # 3 & 4. Interactive 70 / 30
    needs_70 = "seed_interactive_70" not in pool.opponent_ids
    needs_30 = "seed_interactive_30" not in pool.opponent_ids

    if needs_70 or needs_30:
        print("\n--- Preparing Interactive Replay Splits ---")
        ds_inter = _get_dataset(replays_base, "interactive")
        if ds_inter is not None and len(ds_inter) > 0:
            total_size = len(ds_inter)
            size_70 = int(0.7 * total_size)
            size_30 = total_size - size_70

            ds_70, ds_30 = torch.utils.data.random_split(ds_inter, [size_70, size_30])

            if needs_70:
                print(f"\n--- Training seed_interactive_70 ({size_70} samples) ---")
                policy_70 = train_behavior_cloning(ds_70, **bc_kwargs)
                if policy_70:
                    pool.add(policy_70, "seed_interactive_70")
                    added_seeds.append("seed_interactive_70")
            else:
                print("\nseed_interactive_70 already exists.")

            if needs_30:
                print(f"\n--- Training seed_interactive_30 ({size_30} samples) ---")
                policy_30 = train_behavior_cloning(ds_30, **bc_kwargs)
                if policy_30:
                    pool.add(policy_30, "seed_interactive_30")
                    added_seeds.append("seed_interactive_30")
            else:
                print("\nseed_interactive_30 already exists.")
    else:
        print("\nseed_interactive_70 and seed_interactive_30 already exist.")

    if added_seeds:
        pool.save_state()
        print(f"\nPool state saved. Added {len(added_seeds)} seeds: {', '.join(added_seeds)}")
    else:
        print("\nNo new seeds were added to the pool.")

    print(f"\nCurrent pool state: {pool}")


if __name__ == "__main__":
    main()
