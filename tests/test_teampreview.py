import argparse
import asyncio
import sys
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))

from heuristic import FuzzyHeuristic
from teampreviewhandler import TeamPreviewHandler
from teams import RandomTeamFromPool


class WithTeamPreview:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tp_handler = TeamPreviewHandler()

    def teampreview(self, battle: AbstractBattle) -> str:
        return self._tp_handler.select_team(battle)


class RandomPlayerTP(WithTeamPreview, RandomPlayer):
    pass


class MaxBasePowerPlayerTP(WithTeamPreview, MaxBasePowerPlayer):
    pass


class SimpleHeuristicsPlayerTP(WithTeamPreview, SimpleHeuristicsPlayer):
    pass


class FuzzyHeuristicTP(WithTeamPreview, FuzzyHeuristic):
    pass


MATCHUPS = [
    (RandomPlayer, RandomPlayerTP, "Random"),
    (MaxBasePowerPlayer, MaxBasePowerPlayerTP, "MaxBP"),
    (SimpleHeuristicsPlayer, SimpleHeuristicsPlayerTP, "SimpleH"),
    (FuzzyHeuristic, FuzzyHeuristicTP, "FuzzyH"),
]


async def main(n_battles: int):
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

    def create_player(player_class, name):
        return player_class(
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10,
            team=team,
            accept_open_team_sheet=True,
        )

    players = []
    print(f"Running {len(MATCHUPS)} matchups * {n_battles} battles each...\n")

    for base_cls, tp_cls, name in MATCHUPS:
        p_base = create_player(base_cls, name)
        p_tp = create_player(tp_cls, f"TP{name}")

        print(f"Battling {name} vs {name}+TP ({n_battles} games)...")
        await p_base.battle_against(p_tp, n_battles=n_battles)

        base_wr = p_base.win_rate
        print(f"  {name}:     {base_wr:.2%}")
        print(f"  {name}+TP:  {1 - base_wr:.2%}\n")

        players.extend([p_base, p_tp])
        p_base.reset_battles()
        p_tp.reset_battles()

    for p in players:
        await p.ps_client.stop_listening()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare each heuristic with and without team preview handling."
    )
    parser.add_argument(
        "-b",
        type=int,
        default=100,
        help="Number of battles per matchup (default: 100)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.b))
