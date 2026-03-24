import numpy as np
import torch
from numpy.typing import NDArray
from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.player import (
    BattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    Player,
)

import observation_builder
from env import Gen9VGCEnv


class FuzzyHeuristic(Player):
    """
    A heuristic player designed for Double Battles.
    Generates all valid pairs of actions, scores them using a set of rules,
    takes the top 5, and samples using softmax.
    """

    # Move + ability classes
    REDIRECTION = {"followme", "ragepowder"}
    SETUP = {
        "nastyplot",
        "dragondance",
        "bulkup",
        "quiverdance",
        "curse",
    }
    SPEED_CONTROL = {"icywind", "drumbeating"}
    SPREAD = {
        "makeitrain",
        "eruption",
        "heatwave",
        "earthquake",
        "rockslide",
        "dazzlinggleam",
        "hypervoice",
        "muddywater",
        "blizzard",
        "snarl",
        "icywind",
        "expandingforce",
    }
    PIVOT = {"uturn", "partingshot"}
    STAT_DROPPING = {"icywind", "partingshot", "drumbeating"}
    PLUS_PRIORITY = {"extremespeed", "grassyglide", "suckerpunch", "aquajet"}
    GAMBLES = {"direclaw", "fissure"}
    WEATHER_ABILITIES = {"FIRE": "drought", "WATER": "drizzle", "ICE": "snowwarning"}
    TERRAIN_ABILITIES = {"GRASS": "grassysurge", "PSYCHIC": "psychicsurge"}
    WEATHER_TYPE = {
        Weather.SUNNYDAY: "FIRE",
        Weather.RAINDANCE: "WATER",
        Weather.SNOW: "ICE",
    }
    TERRAIN_TYPE = {
        Field.GRASSY_TERRAIN: "GRASS",
        Field.PSYCHIC_TERRAIN: "PSYCHIC",
    }
    # some sets omitted since they have one element and are
    # compared as strings below (like protect / trickroom / tailwind)

    def __init__(self, k: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def _get_actions(self, battle: DoubleBattle, poke_no: int) -> NDArray[np.int64]:
        mask = observation_builder.get_action_mask(battle)
        valid_indices = torch.where(mask[poke_no] > 0)[0]
        return valid_indices.cpu().numpy()

    def _is_valid_pair(self, a0: int, a1: int) -> bool:
        # Both switching to same slot
        if (1 <= a0 <= 6) and a0 == a1:
            return False
        # Both terastallizing
        if (27 <= a0 <= 46) and (27 <= a1 <= 46):
            return False
        # Both passing
        if a0 == 0 and a1 == 0:
            return False

        return True

    def _score_pair(self, battle: DoubleBattle, a0: int, a1: int) -> float:
        pass

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        actions0 = self._get_actions(battle, 0)
        actions1 = self._get_actions(battle, 1)

        scores = []
        pairs = []
        for a0 in actions0:
            for a1 in actions1:
                if self._is_valid_pair(a0, a1):
                    scores.append(self._score_pair(battle, a0, a1))
                    pairs.append((a0, a1))

        if not scores:
            o0 = (
                Gen9VGCEnv._action_to_order_individual(actions0[0], battle, fake=False, pos=0)
                if actions0
                else PassBattleOrder()
            )
            o1 = (
                Gen9VGCEnv._action_to_order_individual(actions1[0], battle, fake=False, pos=1)
                if actions0
                else PassBattleOrder()
            )
            return DoubleBattleOrder(o0, o1)  # type: ignore

        scores_t = torch.tensor(scores, dtype=torch.float32)
        kc = min(len(scores), self.k)
        top_scores, top_indices = torch.topk(scores_t, kc)

        temp = 2.0
        if torch.isnan(top_scores / temp).any():
            # fallback if something goes wrong
            idx = 0
        else:
            idx = torch.distributions.Categorical(logits=top_scores / temp).sample().item()

        chosen_pair = pairs[top_indices[idx].item()]  # type: ignore
        return DoubleBattleOrder(chosen_pair[0], chosen_pair[1])

    # Helper checks
    def _is_immune_to_fake_out(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if any(t and "GHOST" in str(t) for t in opp.types):
            return True
        if getattr(opp, "item", "") == "covertcloak":
            return True
        return False

    def _is_immune_to_powder(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if any(t and "GRASS" in str(t) for t in opp.types):
            return True
        if getattr(opp, "item", "") == "safetygoggles":
            return True
        return False

    def _is_immune_to_prankster(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if any(t and "DARK" in str(t) for t in opp.types):
            return True
        return False

    def _can_hit_ghost_with_normal_fighting(self, attacker: Pokemon) -> bool:
        ab = getattr(attacker, "ability", "")
        if ab == "mindseye" or ab == "scrappy":
            return True
        return False

    def _active_weather_type(self, battle: DoubleBattle) -> tuple[str, int]:
        weather_info = getattr(battle, "_weather", {})
        for w, start_turn in weather_info.items():
            if w in self.WEATHER_TYPE:
                turns = max(0, 5 - (battle.turn - start_turn))
                return self.WEATHER_TYPE[w], turns
        return "NONE", 0

    def _active_terrain_type(self, battle: DoubleBattle) -> tuple[str, int]:
        for field, start_turn in battle.fields.items():
            if field in self.TERRAIN_TYPE:
                turns = max(0, 5 - (battle.turn - start_turn))
                return self.TERRAIN_TYPE[field], turns
        return "NONE", 0

    def _trickroom_turns(self, battle: DoubleBattle) -> int:
        start_turn = battle.fields.get(Field.TRICK_ROOM, -1)
        if start_turn >= 0:
            return max(0, 5 - (battle.turn - start_turn))
        return 0

    def _tailwind_turns(self, battle: DoubleBattle) -> int:
        start_turn = battle.side_conditions.get(SideCondition.TAILWIND, -1)
        if start_turn >= 0:
            return max(0, 4 - (battle.turn - start_turn))
        return 0

    def _veil_turns(self, battle: DoubleBattle) -> int:
        start_turn = battle.side_conditions.get(SideCondition.AURORA_VEIL, -1)
        if start_turn >= 0:
            return max(0, 5 - (battle.turn - start_turn))
        return 0


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    from poke_env import AccountConfiguration, LocalhostServerConfiguration

    sys.path.append(str(Path(__file__).parent))
    from replay_gen import TerminalPlayer
    from teams import RandomTeamFromPool

    async def battle_against_bot():
        teams_dir = Path(__file__).parent.parent / "teams"
        team_files = [
            path.read_text(encoding="utf-8") for path in teams_dir.iterdir() if path.is_file()
        ]

        team = RandomTeamFromPool(team_files)
        fmt = "gen9vgc2025regh"

        bot_player = FuzzyHeuristic(
            k=10,
            account_configuration=AccountConfiguration("FuzzyBot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )

        terminal_player = TerminalPlayer(
            save_dir="/tmp/replays",
            account_configuration=AccountConfiguration("TerminalHuman", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )

        print("Starting 1 battle against Gen9DoublesFuzzyHeuristic...")
        await terminal_player.battle_against(bot_player, n_battles=1)
        print("Battle finished!")

    asyncio.run(battle_against_bot())
