import random
from typing import Any

from poke_env.battle import AbstractBattle, DoubleBattle

try:
    from teampreview_lsa import LSATeamPreviewModel
except ImportError:
    LSATeamPreviewModel = None 


def team_species(battle: DoubleBattle, our_side: bool = True) -> list[str]:
    team = battle.team if our_side else battle.opponent_team
    out: list[str] = []
    for mon in team.values():
        name = getattr(mon, "species", None)
        if name is not None and hasattr(name, "name"):
            out.append(str(name.name))
        else:
            out.append(getattr(mon, "base_species", str(mon)))
    return out


class TeamPreviewHandler:
    def __init__(
        self,
        model: Any = None,
        *,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.bring = 4
        self.lead = 2
        self.temperature = temperature
        self._is_lsa = LSATeamPreviewModel is not None and isinstance(model, LSATeamPreviewModel)

        if self.model is not None and self._is_lsa:
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

    def decide(self, battle: DoubleBattle) -> tuple[tuple[int, ...], tuple[int, ...]]:
        team_size = len(battle.team)
        if team_size <= 0:
            raise ValueError("team is empty")

        bring_k = min(self.bring, team_size)
        lead_k = min(self.lead, bring_k)

        if self._is_lsa:
            our_species = team_species(battle, our_side=True)
            opp_species = team_species(battle, our_side=False)
            return self.model.decide_stochastic(  # type: ignore[union-attr]
                our_species, opp_species, team_size=team_size, bring_k=bring_k, lead_k=lead_k
            )

        members = list(range(team_size))
        random.shuffle(members)
        bring_tuple = tuple(members[:bring_k])
        lead_tuple = tuple(members[:lead_k])
        return bring_tuple, lead_tuple
