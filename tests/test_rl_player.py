import asyncio
import sys
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

# Add src to python path
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))

from policy import PolicyNet
from rl_player import RLPlayer
from teampreview import TeamPreviewHandler
from teams import RandomTeamFromPool


async def evaluate_against_opponents(player, opponents, n_battles=20):
    results = {}
    for opponent in opponents:
        print(f"\n--- Battling {player.username} vs {opponent.username} for {n_battles} games ---")
        await player.battle_against(opponent, n_battles=n_battles)
        results[opponent.username] = player.win_rate
        print(f"Result: {player.username} Win Rate vs {opponent.username} -> {player.win_rate:.2%}")
        player.reset_battles()
        opponent.reset_battles()
    return results


async def main():
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
    if not teams_dir.exists():
        print(f"Teams directory not found: {teams_dir}")
        return

    team_files = [
        path.read_text(encoding="utf-8") for path in Path(teams_dir).iterdir() if path.is_file()
    ]
    if not team_files:
        print("No team files found. Please ensure there are team files in the teams directory.")
        return

    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"
    tp_handler = TeamPreviewHandler()

    rl_policy = PolicyNet()
    ppo_checkpoint_path = root_dir / "checkpoints" / "ppo_checkpoint.pt"

    if ppo_checkpoint_path.exists():
        print(f"Loading PPO Policy from {ppo_checkpoint_path}...")
        ppo_checkpoint = torch.load(ppo_checkpoint_path, weights_only=False, map_location="cpu")
        rl_policy.load_state_dict(ppo_checkpoint["model_state_dict"])
    else:
        print(f"Warning: PPO Checkpoint not found at {ppo_checkpoint_path}.")
        print("Running with a randomly initialized model.")

    rl_player = RLPlayer(
        policy=rl_policy,
        teampreview_handler=tp_handler,
        account_configuration=AccountConfiguration("RLPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=2,
        team=team,
        accept_open_team_sheet=True,
    )

    # Create opponents
    random_player = RandomPlayer(
        account_configuration=AccountConfiguration("RandomPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=2,
        team=team,
        accept_open_team_sheet=True,
    )

    max_power_player = MaxBasePowerPlayer(
        account_configuration=AccountConfiguration("MaxPowerPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=2,
        team=team,
        accept_open_team_sheet=True,
    )

    simple_heuristics_player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("SimpleHeurPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=2,
        team=team,
        accept_open_team_sheet=True,
    )

    opponents = [random_player, max_power_player, simple_heuristics_player]
    ppo_results = await evaluate_against_opponents(rl_player, opponents)

    print("\n==================================")
    print("PPO Policy Winrates")
    print("==================================")

    for opp in opponents:
        opp_name = opp.username
        print(f"Against {opp_name}:")
        print(f"  - PPO Policy Win Rate: {ppo_results[opp_name]:.2%}")

    # Clean up
    for p in [rl_player] + opponents:
        await p.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
