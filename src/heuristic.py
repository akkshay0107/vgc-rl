import copy

import numpy as np
import torch
from numpy.typing import NDArray
from poke_env.battle import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
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

    def _defensive_rating(self, battle: DoubleBattle, mon: Pokemon) -> float:
        return 0.0

    def _score_switch(self, battle: DoubleBattle, action: int, pos: int) -> float:
        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return 0.0

        score = 0.0
        # SWITCHING BONUSES
        pref_atk = "atk" if active_mon.base_stats["atk"] >= active_mon.base_stats["spa"] else "spa"
        boost = active_mon.boosts.get(pref_atk, 0)
        if boost > 0:
            mult = (2.0 + boost) / 2.0
        else:
            mult = 2.0 / (2.0 - boost)

        # If mult < 1.0 (drop/bad speed), 1.0 - mult > 0 -> positive score (incentivize switch)
        # If mult > 1.0 (boost/good speed), 1 / mult - 1 < 0 -> negative score (penalize switch)
        if mult < 1.0:
            score += (1.0 - mult) * 5.0  # Weight for negative drops
        else:
            obm = 1.0 / mult
            score += (obm - 1.0) * 2.5  # Weight for positive boosts (penalty to switch)

        # incentivize switching out if affected by negative effects
        if Effect.ENCORE in active_mon.effects and battle.available_moves[pos]:
            locked_move = battle.available_moves[pos][0]
            if locked_move.category == MoveCategory.STATUS:
                score += 4.0
            else:
                type_effectiveness = [
                    opp.damage_multiplier(locked_move)
                    for opp in battle.opponent_active_pokemon
                    if opp is not None
                ]
                score += max(0.0, 4.0 - sum(type_effectiveness) / len(type_effectiveness))

        if Effect.TAUNT in active_mon.effects:
            status_moves = [
                m for m in active_mon.moves.values() if m.category == MoveCategory.STATUS
            ]
            score += float(len(status_moves))

        if Effect.CONFUSION in active_mon.effects:
            score += 1.0

        switch_mon = list(battle.team.values())[action - 1]
        score += self._defensive_rating(battle, switch_mon) - self._defensive_rating(
            battle, active_mon
        )
        return score

    def _score_tera(self, battle: DoubleBattle, action: int, pos: int) -> float:
        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return 0.0

        dummy_mon = copy.deepcopy(active_mon)
        dummy_mon._terastallized = True
        return self._defensive_rating(battle, dummy_mon) - self._defensive_rating(
            battle, active_mon
        )

    def _score_single_action(self, battle: DoubleBattle, action: int, pos: int) -> float:
        if 1 <= action <= 6:
            return self._score_switch(battle, action, pos)

        score = 0.0
        # score normal move here
        if 27 <= action <= 46:
            score += self._score_tera(battle, action, pos)
        return score

    def _score_synergy(self, battle: DoubleBattle, a0: int, a1: int) -> float:
        return 0.0

    def _score_pair(self, battle: DoubleBattle, a0: int, a1: int) -> float:
        score = 0.0
        score += self._score_single_action(battle, a0, pos=0)
        score += self._score_single_action(battle, a1, pos=1)
        score += self._score_synergy(battle, a0, a1)
        return score

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
                if actions1
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
        o0 = Gen9VGCEnv._action_to_order_individual(chosen_pair[0], battle, fake=False, pos=0)
        o1 = Gen9VGCEnv._action_to_order_individual(chosen_pair[1], battle, fake=False, pos=1)
        return DoubleBattleOrder(o0, o1)

    # Helper checks
    def _is_immune_to_fake_out(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if PokemonType.GHOST in opp.types:
            return True
        if getattr(opp, "item", "") == "covertcloak":
            return True
        return False

    def _is_immune_to_powder(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if PokemonType.GRASS in opp.types:
            return True
        if getattr(opp, "item", "") == "safetygoggles":
            return True
        return False

    def _is_immune_to_prankster(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if PokemonType.DARK in opp.types:
            return True
        return False

    def _can_hit_ghost_with_normal_fighting(self, attacker: Pokemon) -> bool:
        ab = getattr(attacker, "ability", "")
        if ab == "mindseye" or ab == "scrappy":
            return True
        return False

    def _active_weather_type(self, battle: DoubleBattle) -> tuple[str, int]:
        for w, start_turn in battle.weather.items():
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

    def _stab_multiplier(self, mon: Pokemon, move: Move) -> float:
        has_adaptability = mon.ability == "adaptability"

        # Get original types
        original_types = mon.original_types

        is_original_stab = move.type in original_types

        if not mon.is_terastallized:
            if is_original_stab:
                return 2.0 if has_adaptability else 1.5
            return 1.0

        # Terastallized
        is_tera_stab = move.type == mon.tera_type
        # no tera stellar thankfully
        if is_tera_stab:
            if is_original_stab:  # Tera matches original type
                return 2.25 if has_adaptability else 2.0
            else:  # Tera does not match original type
                return 2.0 if has_adaptability else 1.5

        if is_original_stab:  # Move matches original type but not Tera type
            return 2.0 if has_adaptability else 1.5

        return 1.0


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
