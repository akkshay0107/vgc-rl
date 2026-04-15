import random
from collections.abc import Callable
from typing import Any

from poke_env.battle import AbstractBattle, DoubleBattle
from teampreview_document import team_species_list
from teampreview_hybrid import fuse_teampreview


class TeamPreviewHandler:
    def __init__(
        self,
        model: Any = None,
        *,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.bring = 4
        self.lead = 2
        self.temperature = temperature
        self.deterministic = deterministic

        if self.model is not None and hasattr(self.model, "temperature"):
            self.model.temperature = self.temperature

    def select_team(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)

        if self.model is None:
            members = list(range(1, len(battle.team) + 1))
            random.shuffle(members)
            return "/team " + "".join([str(c) for c in members[: self.bring]])

        decision = self.decide(battle)
        bring_1_based = [i + 1 for i in decision[0]]
        lead_1_based = [i + 1 for i in decision[1]]
        remaining = [i for i in bring_1_based if i not in set(lead_1_based)]
        ordered = lead_1_based + remaining
        return "/team " + "".join(str(i) for i in ordered)

    def select_team_with_fuzzy_fallback(
        self,
        battle: AbstractBattle,
        fuzzy_fn: Callable[[AbstractBattle], str],
        *,
        min_bring_similarity: float = 0.5,
    ) -> str:
        assert isinstance(battle, DoubleBattle)
        fuzzy_cmd = fuzzy_fn(battle)
        if self.model is None:
            return fuzzy_cmd
        model_cmd = self.select_team(battle)
        return fuse_teampreview(fuzzy_cmd, model_cmd, min_bring_similarity=min_bring_similarity)

    def decide(self, battle: DoubleBattle) -> tuple[tuple[int, ...], tuple[int, ...]]:
        team_size = len(battle.team)
        if team_size <= 0:
            raise ValueError("team is empty")

        bring_k = min(self.bring, team_size)
        lead_k = min(self.lead, bring_k)

        if self.model is not None and callable(getattr(self.model, "decide_from_battle", None)):
            return self.model.decide_from_battle(
                battle, bring_k=bring_k, lead_k=lead_k, deterministic=self.deterministic
            )

        if self.model is not None:
            our_species = team_species_list(battle, our_side=True)
            opp_species = team_species_list(battle, our_side=False)
            return self.model.decide_stochastic(
                our_species, opp_species, team_size=team_size, bring_k=bring_k, lead_k=lead_k
            )

        members = list(range(team_size))
        random.shuffle(members)
        bring_tuple = tuple(members[:bring_k])
        lead_tuple = tuple(members[:lead_k])
        return bring_tuple, lead_tuple


def team_species(battle: DoubleBattle, our_side: bool = True) -> list[str]:
    return team_species_list(battle, our_side=our_side)
