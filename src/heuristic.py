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

    def _calculate_damage(
        self, move: Move, attacker: Pokemon, defender: Pokemon, battle: DoubleBattle
    ) -> float:
        if move.category == MoveCategory.STATUS:
            return 0.0

        power = move.base_power
        if move.id in ["eruption", "waterspout"]:
            power = 150 * (attacker.current_hp / attacker.max_hp)
        elif power <= 0:
            power = 60

        level = attacker.level
        if move.category == MoveCategory.PHYSICAL:
            a_stat = attacker.stats["atk"]
            d_stat = defender.stats["def"]
            a_boost = attacker.boosts.get("atk", 0)
            d_boost = defender.boosts.get("def", 0)
            if attacker.ability == "guts" and attacker.status:
                a_stat *= 1.5  # type: ignore
        else:  # SPECIAL
            a_stat = attacker.stats["spa"]
            d_stat = defender.stats["spd"]
            a_boost = attacker.boosts.get("spa", 0)
            d_boost = defender.boosts.get("spd", 0)

        a_stat *= self._get_boost_mult(a_boost)  # type: ignore
        d_stat *= self._get_boost_mult(d_boost)  # type: ignore

        # snow defense boost
        if PokemonType.ICE in defender.types:
            if Weather.SNOW in battle.weather:
                if move.category == MoveCategory.PHYSICAL:
                    d_stat *= 1.5

        # damage formula
        damage = (((2 * level / 5) + 2) * power * a_stat / d_stat) / 50 + 2
        if move.id in self.SPREAD:
            damage *= 0.75

        # weather multiplier
        weather_type, _ = self._active_weather_type(battle)
        if weather_type == "FIRE":
            if move.type == PokemonType.FIRE:
                damage *= 1.5
            elif move.type == PokemonType.WATER:
                damage *= 0.5
        elif weather_type == "WATER":
            if move.type == PokemonType.WATER:
                damage *= 1.5
            elif move.type == PokemonType.FIRE:
                damage *= 0.5

        damage *= self._stab_multiplier(attacker, move)
        damage *= defender.damage_multiplier(move)

        # burn multiplier
        if attacker.status == "brn" and move.category == MoveCategory.PHYSICAL:
            if attacker.ability != "guts" and move.id != "facade":
                damage *= 0.5

        max_hp = defender.max_hp if (defender.max_hp and defender.max_hp > 0) else 100
        return damage / max_hp

    def _offensive_rating(
        self, move: Move, attacker: Pokemon, recepient: Pokemon, battle: DoubleBattle
    ) -> float:
        dmg_pct = self._calculate_damage(move, attacker, recepient, battle)
        rating = 5.0 * dmg_pct

        # ko bonus
        current_hp_pct = (
            recepient.current_hp / recepient.max_hp
            if (recepient.max_hp and recepient.max_hp > 0)
            else 1.0
        )
        if dmg_pct >= current_hp_pct:
            rating += 2.0

        # outspeed bonus
        if move.priority > 0:
            rating += 1.0
        else:
            # Speed comparison
            a_is_op = attacker in battle.opponent_active_pokemon
            r_is_op = recepient in battle.opponent_active_pokemon

            a_spe = attacker.stats["spe"] * self._get_boost_mult(attacker.boosts.get("spe", 0))  # type: ignore
            if attacker.ability == "unburden" and not attacker.item:
                a_spe *= 2.0
            if self._tailwind_turns(battle, a_is_op) > 0:
                a_spe *= 2.0

            r_spe = recepient.stats["spe"] * self._get_boost_mult(recepient.boosts.get("spe", 0))  # type: ignore
            if recepient.ability == "unburden" and not recepient.item:
                r_spe *= 2.0
            if self._tailwind_turns(battle, r_is_op) > 0:
                r_spe *= 2.0

            if self._trickroom_turns(battle) > 0:
                if a_spe < r_spe:
                    rating += 0.5
            else:
                if a_spe > r_spe:
                    rating += 0.5

        if move.accuracy is not True:
            rating *= move.accuracy / 100.0
        return rating

    def _get_proba(self, battle: DoubleBattle, mon: Pokemon) -> NDArray[np.float64] | None:
        ors = []
        for op in battle.opponent_active_pokemon:
            if op is None or op.fainted:
                continue
            for move in op.moves.values():
                if move.category == MoveCategory.STATUS:
                    continue
                ors.append(self._offensive_rating(move, op, mon, battle))

        if len(ors) == 0:
            return None

        ors_arr = np.array(ors, dtype=np.float64)
        total = np.sum(ors_arr)
        if total == 0:
            return np.ones(len(ors_arr), dtype=np.float64) / len(ors_arr)
        return ors_arr / total

    def _get_dmg(self, battle: DoubleBattle, mon: Pokemon) -> NDArray[np.float64]:
        dmgs = []
        for op in battle.opponent_active_pokemon:
            if op is None or op.fainted:
                continue
            for move in op.moves.values():
                if move.category == MoveCategory.STATUS:
                    continue
                dmgs.append(self._calculate_damage(move, op, mon, battle))

        return np.array(dmgs, dtype=np.float64)

    def _score_switch(self, battle: DoubleBattle, action: int, pos: int) -> float:
        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return 0.0

        score = 0.0
        # SWITCHING BONUSES
        pref_atk = "atk" if active_mon.base_stats["atk"] >= active_mon.base_stats["spa"] else "spa"
        mult = self._get_boost_mult(active_mon.boosts.get(pref_atk, 0))

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

        proba = self._get_proba(battle, active_mon)
        if proba is None:
            return score

        dmg_active = self._get_dmg(battle, active_mon)
        defensive_rating_active = np.dot(proba, dmg_active)

        switch_mon = list(battle.team.values())[action - 1]
        dmg_switch = self._get_dmg(battle, switch_mon)
        defensive_rating_switch = np.dot(proba, dmg_switch)

        score += defensive_rating_switch - defensive_rating_active
        return score

    def _score_tera(self, battle: DoubleBattle, active_mon: Pokemon, dummy_tera: Pokemon) -> float:
        proba = self._get_proba(battle, active_mon)
        if proba is None:
            return 0.0

        dmg_active = self._get_dmg(battle, active_mon)
        defensive_rating_active = np.dot(proba, dmg_active)

        dmg_tera = self._get_dmg(battle, dummy_tera)
        defensive_rating_tera = np.dot(proba, dmg_tera)
        return defensive_rating_tera - defensive_rating_active

    def _score_move(self, battle: DoubleBattle, action: int, mon: Pokemon, pos: int) -> float:
        # Action mapping from env.py:
        move_idx = (action - 7) % 20 // 5
        target_idx = (action - 7) % 5 - 2
        mvs = (
            battle.available_moves[pos]
            if len(battle.available_moves[pos]) == 1
            and battle.available_moves[pos][0].id in ["struggle", "recharge"]
            else list(mon.moves.values())
        )

        if move_idx >= len(mvs):
            return -10.0

        move = mvs[move_idx]
        if move.category == MoveCategory.STATUS:
            # TODO: add a utility calc for status moves
            return 0.0

        # determine targets
        targets = []
        if target_idx == 1:
            targets = [battle.opponent_active_pokemon[0]]
        elif target_idx == 2:
            targets = [battle.opponent_active_pokemon[1]]
        elif target_idx == 0:
            targets = [battle.active_pokemon[1 - pos]]
        elif target_idx == -1:
            targets = [mon]
        elif target_idx == -2:
            if move.id in self.SPREAD:
                targets = [op for op in battle.opponent_active_pokemon if op and not op.fainted]
            elif move.category != MoveCategory.STATUS:
                # Pick first non-fainted opponent
                for op in battle.opponent_active_pokemon:
                    if op and not op.fainted:
                        targets = [op]
                        break
            else:
                targets = [mon]

        if not targets:
            return -1.0

        total_rating = 0.0
        for t in targets:
            if t is None or t.fainted:
                continue
            total_rating += self._offensive_rating(move, mon, t, battle)

        return total_rating if total_rating > 0 else -1.0

    def _score_single_action(self, battle: DoubleBattle, action: int, pos: int) -> float:
        if 1 <= action <= 6:
            return self._score_switch(battle, action, pos)

        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return 0.0

        if action <= 26:
            return self._score_move(battle, action, active_mon, pos)

        dummy_tera = copy.deepcopy(active_mon)
        dummy_tera._terastallized = True
        return self._score_tera(battle, active_mon, dummy_tera) + self._score_move(
            battle, action, dummy_tera, pos
        )

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

    def _tailwind_turns(self, battle: DoubleBattle, opponent: bool = False) -> int:
        side_conds = battle.opponent_side_conditions if opponent else battle.side_conditions
        start_turn = side_conds.get(SideCondition.TAILWIND, -1)
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

    def _get_boost_mult(self, boost):
        if boost >= 0:
            return (2.0 + boost) / 2.0
        return 2.0 / (2.0 - boost)


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
