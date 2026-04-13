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
        "num_epochs": 10,
        "learning_rate": 5e-4,
        "val_split_ratio": 0.1,
    }

    print("\n" + "=" * 60)
    print("Seeding pool with behaviour cloning policies")
    print("=" * 60)

    # 1. Max Base Power
    if "seed_max_base_power" not in pool.opponent_ids:
        print("--- Training seed_max_base_power ---")
        ds_mbp = _get_dataset(replays_base, "max_base_power")
        if ds_mbp:
            policy = train_behavior_cloning(ds_mbp, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_max_base_power")
                added_seeds.append("seed_max_base_power")
    else:
        print("seed_max_base_power already exists.")

    # 2. Simple Heuristic
    if "seed_simple_heuristic" not in pool.opponent_ids:
        print("--- Training seed_simple_heuristic ---")
        ds_sh = _get_dataset(replays_base, "simple_heuristic")
        if ds_sh:
            policy = train_behavior_cloning(ds_sh, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_simple_heuristic")
                added_seeds.append("seed_simple_heuristic")
    else:
        print("seed_simple_heuristic already exists.")

    # 3. Fuzzy Heuristic
    if "seed_fuzzy_heuristic" not in pool.opponent_ids:
        print("--- Training seed_fuzzy_heuristic ---")
        ds_fuzzy = _get_dataset(replays_base, "fuzzy_heuristic")
        if ds_fuzzy:
            policy = train_behavior_cloning(ds_fuzzy, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_fuzzy_heuristic")
                added_seeds.append("seed_fuzzy_heuristic")
    else:
        print("seed_fuzzy_heuristic already exists.")

    # 4. Fuzzy into human tuning
    if "seed_mixed" not in pool.opponent_ids:
        print("--- Training seed_mixed ---")
        ds_fuzzy = _get_dataset(replays_base, "fuzzy_heuristic")
        ds_inter = _get_dataset(replays_base, "interactive")

        if ds_fuzzy and ds_inter:
            policy = train_behavior_cloning(ds_fuzzy, **bc_kwargs)
            if policy is None:
                print("Error: Mixed dataset could not be created.")
                return

            # retrain on interactive
            new_kwargs = bc_kwargs
            new_kwargs["learning_rate"] /= 10.0
            policy = train_behavior_cloning(ds_inter, policy=policy, **new_kwargs)
            if policy is None:
                print("Error: Interactive dataset missing.")
                return

            pool.add(policy, "seed_mixed")
            added_seeds.append("seed_mixed")
        else:
            print("Error: Mixed dataset could not be created.")
    else:
        print("seed_mixed already exists.")

    if added_seeds:
        pool.save_state()
        print(f"\nPool state saved. Added {len(added_seeds)} seeds: {', '.join(added_seeds)}")
    else:
        print("\nNo new seeds were added to the pool.")

    print(f"Current pool state: {pool}")


if __name__ == "__main__":
    main()
