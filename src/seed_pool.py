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

    # 4. Interactive
    if "seed_interactive" not in pool.opponent_ids:
        print("--- Training seed_interactive ---")
        ds_inter = _get_dataset(replays_base, "interactive")
        if ds_inter:
            policy = train_behavior_cloning(ds_inter, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_interactive")
                added_seeds.append("seed_interactive")
    else:
        print("seed_interactive already exists.")

    # 5. Mixed
    if "seed_mixed" not in pool.opponent_ids:
        print("--- Training seed_mixed ---")
        ds_mbp = _get_dataset(replays_base, "max_base_power")
        ds_sh = _get_dataset(replays_base, "simple_heuristic")
        ds_fuzzy = _get_dataset(replays_base, "fuzzy_heuristic")
        ds_inter = _get_dataset(replays_base, "interactive")

        if ds_mbp and ds_sh and ds_fuzzy and ds_inter:
            # We want to create a mixed dataset with 25/25/50 split
            # n_fuzzy = 2 * n_mbp = 2 * n_sh
            n_mbp_avail = len(ds_mbp)
            n_sh_avail = len(ds_sh)
            n_fuzzy_avail = len(ds_fuzzy)

            # Maximize the number of samples we can use
            n = min(n_mbp_avail, n_sh_avail, n_fuzzy_avail // 2)

            if n > 0:
                sub_mbp, _ = torch.utils.data.random_split(ds_mbp, [n, n_mbp_avail - n])
                sub_sh, _ = torch.utils.data.random_split(ds_sh, [n, n_sh_avail - n])
                sub_fuzzy, _ = torch.utils.data.random_split(
                    ds_fuzzy, [2 * n, n_fuzzy_avail - 2 * n]
                )

                ds_mixed_init = torch.utils.data.ConcatDataset([sub_mbp, sub_sh, sub_fuzzy])
                policy = train_behavior_cloning(ds_mixed_init, **bc_kwargs)
                if policy is None:
                    print("Error: Mixed dataset could not be created.")
                    return

                # retrain on interactive
                new_kwargs = bc_kwargs
                new_kwargs["learning_rate"] /= 2.0
                policy = train_behavior_cloning(ds_inter, policy=policy, **new_kwargs)
                if policy is None:
                    print("Error: Interactive dataset missing.")
                    return

                pool.add(policy, "seed_mixed")
                added_seeds.append("seed_mixed")

            else:
                print("Not enough samples to create mixed dataset.")
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
