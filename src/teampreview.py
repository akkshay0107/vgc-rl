import random

from poke_env.battle import AbstractBattle, DoubleBattle


class TeamPreviewHandler:
    """
    Class to handler team preview selection for the RL player
    """

    def __init__(self) -> None:
        pass

    def select_team(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        # TODO: implement this entirely
        # as of rn just copied the random teampreview impl from poke-env
        members = list(range(1, len(battle.team) + 1))
        random.shuffle(members)
        return "/team " + "".join([str(c) for c in members])
