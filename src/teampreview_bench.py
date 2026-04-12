from __future__ import annotations

from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from heuristic import FuzzyHeuristic
from teampreviewhandler import TeamPreviewHandler
from teams import RandomTeamFromPool


class WithTeamPreview:

    def __init__(self, *args, tp_model=None, tp_temperature=1.0, tp_deterministic=False, **kwargs):
        if "lsa_model" in kwargs:
            tp_model = kwargs.pop("lsa_model")
        elif "tp_model" in kwargs:
            tp_model = kwargs.pop("tp_model")
        super().__init__(*args, **kwargs)
        self._tp_handler = TeamPreviewHandler(
            model=tp_model,
            temperature=tp_temperature,
            deterministic=tp_deterministic,
        )

    def teampreview(self, battle: AbstractBattle) -> str:
        if self._tp_handler.model is None:
            return super().teampreview(battle)
        return self._tp_handler.select_team(battle)


class RandomPlayerTP(WithTeamPreview, RandomPlayer):
    pass


class MaxBasePowerPlayerTP(WithTeamPreview, MaxBasePowerPlayer):
    pass


class SimpleHeuristicsPlayerTP(WithTeamPreview, SimpleHeuristicsPlayer):
    pass


class FuzzyHeuristicTP(WithTeamPreview, FuzzyHeuristic):
    pass


class FuzzyHeuristicHybridTP(WithTeamPreview, FuzzyHeuristic):

    def __init__(self, *args, tp_fuzzy_hybrid_similarity: float = 0.5, **kwargs):
        self._tp_hybrid_s = float(tp_fuzzy_hybrid_similarity)
        super().__init__(*args, **kwargs)

    def teampreview(self, battle: AbstractBattle) -> str:
        if self._tp_handler.model is None:
            return FuzzyHeuristic.teampreview(self, battle)
        return self._tp_handler.select_team_with_fuzzy_fallback(
            battle,
            lambda b: FuzzyHeuristic.teampreview(self, b),
            min_bring_jaccard=self._tp_hybrid_j,
        )


MATCHUPS: list[tuple[type, type, str]] = [
    (RandomPlayer, RandomPlayerTP, "Random"),
    (MaxBasePowerPlayer, MaxBasePowerPlayerTP, "MaxBP"),
    (SimpleHeuristicsPlayer, SimpleHeuristicsPlayerTP, "SimpleH"),
    (FuzzyHeuristic, FuzzyHeuristicTP, "FuzzyH"),
]


def load_teampreview_model(path: Path) -> Any:
    from teampreview_lsa import LSATeamPreviewModel
    from teampreview_supervised import SupervisedTeamPreviewModel, is_supervised_checkpoint

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if is_supervised_checkpoint(path):
        return SupervisedTeamPreviewModel.load(path)
    return LSATeamPreviewModel.load(path)


async def run_teampreview_benchmark(
    *,
    root_dir: Path,
    checkpoint: Path | None,
    n_battles: int,
    tp_temperature: float = 1.0,
    tp_deterministic: bool = False,
    matchups: list[tuple[type, type, str]] | None = None,
    fuzzy_hybrid: bool = False,
    fuzzy_hybrid_similarity: float = 0.5,
) -> list[dict[str, Any]]:
    teams_dir = root_dir / "teams"
    if not teams_dir.is_dir():
        raise FileNotFoundError(f"Teams directory not found: {teams_dir}")
    team_files = [p.read_text(encoding="utf-8") for p in teams_dir.iterdir() if p.is_file()]
    if not team_files:
        raise RuntimeError(f"No team files in {teams_dir}")

    tp_model = None
    if checkpoint is not None:
        tp_model = load_teampreview_model(checkpoint)

    team = RandomTeamFromPool(team_files)
    fmt = "gen9vgc2025regh"
    rows: list[dict[str, Any]] = []
    mlist = matchups or MATCHUPS

    def create_player(player_class: type, name: str, *, use_tp: bool):
        kw: dict[str, Any] = dict(
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10,
            team=team,
            accept_open_team_sheet=True,
        )
        if use_tp and tp_model is not None:
            kw["tp_model"] = tp_model
            kw["tp_temperature"] = tp_temperature
            kw["tp_deterministic"] = tp_deterministic
        if use_tp and player_class is FuzzyHeuristicHybridTP:
            kw["tp_fuzzy_hybrid_jaccard"] = fuzzy_hybrid_jaccard
        return player_class(**kw)

    players: list[Any] = []
    for base_cls, tp_cls, name in mlist:
        eff_tp = FuzzyHeuristicHybridTP if (fuzzy_hybrid and name == "FuzzyH") else tp_cls
        print(f"Battling {name} vs {name}+TP ({n_battles} games)...")
        p_base = create_player(base_cls, name, use_tp=False)
        p_tp = create_player(eff_tp, f"TP{name}", use_tp=True)

        await p_base.battle_against(p_tp, n_battles=n_battles)
        base_wr = p_base.win_rate
        rows.append({"name": name, "base_wr": base_wr, "tp_wr": 1.0 - base_wr})

        players.extend([p_base, p_tp])
        p_base.reset_battles()
        p_tp.reset_battles()

    for p in players:
        await p.ps_client.stop_listening()

    return rows
