import asyncio
import sys
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

# Add src to python path
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))

from behaviour_cloning import ReplayDataset, train_behavior_cloning
from policy import PolicyNet
from rl_player import RLPlayer
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
    replays_dir = root_dir / "replays"
    teams_dir = root_dir / "teams"

    if not teams_dir.exists():
        print(f"Teams directory not found: {teams_dir}")
        return

    team_files = [
        path.read_text(encoding="utf-8")
        for path in Path(teams_dir).iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    if not team_files:
        print("No team files found. Please ensure there are team files in the teams directory.")
        return

    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"

    print("Loading Untrained Policy...")
    untrained_policy = PolicyNet()  # Randomly initialized weights
    untrained_player = RLPlayer(
        policy=untrained_policy,
        account_configuration=AccountConfiguration("UntrainedPlayer", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=2,
        team=team,
        accept_open_team_sheet=True,
    )

    if not replays_dir.exists():
        print(f"Error: Replays directory not found at {replays_dir}")
        print("Please ensure you have generated replay files before running behavior cloning.")
        return

    print("--- Starting Behavior Cloning Training ---")
    dataset = ReplayDataset(str(replays_dir))
    if len(dataset) == 0:
        print("Error: No replays found in dataset. Please run replay_gen.py first.")
        return

    bc_policy = train_behavior_cloning(dataset)

    if not bc_policy:
        print("Error: Behavior cloning training failed to return a model.")
        return

    bc_player = RLPlayer(
        policy=bc_policy,
        account_configuration=AccountConfiguration("BCPlayer", None),
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
    untrained_results = await evaluate_against_opponents(untrained_player, opponents)
    bc_results = await evaluate_against_opponents(bc_player, opponents)

    print("\n==================================")
    print("Behaviour Cloning Winrates")
    print("==================================")

    for opp in opponents:
        opp_name = opp.username
        diff = bc_results[opp_name] - untrained_results[opp_name]
        print(f"Against {opp_name}:")
        print(f"  - Untrained Policy Win Rate: {untrained_results[opp_name]:.2%}")
        print(
            f"  - Behavior Cloning Win Rate: {bc_results[opp_name]:.2%} ({'+' if diff >= 0 else ''}{diff:.2%} vs untrained)"
        )

    # Clean up
    for p in [untrained_player, bc_player] + opponents:
        await p.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
