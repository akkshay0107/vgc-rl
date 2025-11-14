import torch
from poke_env.battle import AbstractBattle, DoubleBattle, SideCondition, Weather, Field, Effect
from poke_env.battle.pokemon import Pokemon
from lookups import POKEMON

BATTLE_STATE_DIMS = (2, 5, 30)
# Define action space parameters (from gen9vgcenv.py)
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1  # Tera
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)


class Encoder:
    """
    Description of embedded vector

    2 channels - one for each player

    5 rows - 1st row for field conditions on respective players side of the field
                Next 4 rows for the state of each pokemon selected to play

    30 cols for pokemons
        Col 0 - pokemonID
        Col 1 - primary typing
        Col 2 - secondary typing
        Col 3 - tera type (0 if tera not used else tera type)
        Col 4 - item held / consumed or knocked off
        Col 5 - non volatile status condition
        Col [6-8] - taunt, encore, confusion status respectively (turns active)
        Col 9 - current HP stat
        Col [10-15] - base stats
        Col [16-22] - stat stages (all 6 base stars excluding HP + accuracy and evasion)
        Col [23-26] - pp for each of the 4 moves
        Col 27 - protect counter
        Col 28 - boolean that denotes whether the last turn missed or not (for stomping tantrum)
        Col 29 - last move used (1 - 4)

    cols for field effects (first 5 are global, 6 and 7th are local, value of 0 means inactive)
        also stores some team level counters
        Col 0 - trick room turns remaining
        Col 1 - grassy terrain turns remaining
        Col 2 - psy terrain turns remaining
        Col 3 - sun turns remaining
        Col 4 - rain turns remaining
        Col 5 - tailwind turns remaining
        Col 6 - aurora veil turns remaining
        Col 7 - number of fainted pokemon in the team
        Col 8 - rage fist stacks (0 if no annihilape in the team)
        Col 9 - tera burnt or not
        Col [10-29] - padding using 0 (future space to expand ??)
    """

    @staticmethod
    def _get_pokemon_id(pokemon: Pokemon) -> int:
        if not pokemon:
            return -1

        # Get move list in the exact order they appear.
        move_ids = [move.id for move in pokemon.moves.values()]
        move_list_str = ",".join(move_ids)
        return POKEMON.get(move_list_str, -1)

    @staticmethod
    def _encode_pokemon(pokemon: Pokemon | None, pokemon_row: torch.Tensor):
        if pokemon is None:
            pokemon_row[0] = -1  # set ID to -1 to imply unknown
            return

        pokemon_row[0] = Encoder._get_pokemon_id(pokemon)

        pokemon_row[1] = pokemon.type_1.value
        pokemon_row[2] = 0 if pokemon.type_2 is None else pokemon.type_2.value

        pokemon_row[3] = 1 if pokemon.is_terastallized else 0
        pokemon_row[4] = 1 if pokemon.item else 0

        pokemon_row[5] = 0 if pokemon.status is None else pokemon.status.value

        # Volatile statuses: taunt, encore, confusion
        curr_effects = pokemon.effects
        pokemon_row[6] = 1 if Effect.TAUNT in curr_effects else 0
        pokemon_row[7] = 1 if Effect.ENCORE in curr_effects else 0
        pokemon_row[8] = 1 if Effect.CONFUSION in curr_effects else 0

        pokemon_row[9] = pokemon._current_hp if pokemon._current_hp is not None else 0

        stats = ["hp", "atk", "def", "spa", "spd", "spe"]
        for i, stat in enumerate(stats):
            pokemon_row[10 + i] = pokemon.base_stats[stat]

        boosts = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for i, boost in enumerate(boosts):
            pokemon_row[16 + i] = pokemon.boosts[boost]

        for i, move in enumerate(pokemon.moves):
            pokemon_row[23 + i] = pokemon.moves[move].current_pp

        pokemon_row[27] = pokemon.protect_counter
        pokemon_row[28] = prev_move_failed = 0  # TODO: implement this (not tracked in poke-env)
        pokemon_row[29] = last_move_used = 0  # Added this for new BattleState

    @staticmethod
    def encode_battle_state(battle: DoubleBattle, state: torch.Tensor):
        """
        Fills the state tensor with the current state of the battle.

        Args:
            battle (Battle): The battle object from poke-env.
            state (torch.Tensor): The 2x5x30 tensor to be filled.
        """
        state.zero_()  # Reset tensor to zeros

        # Encode global field conditions for both players
        for player_idx in range(2):
            field_row = state[player_idx, 0]

            trick_room_turns = 0
            grassy_terrain_turns = 0
            psychic_terrain_turns = 0
            if battle.fields:
                grassy_terrain_start = battle.fields.get(Field.GRASSY_TERRAIN, -1)
                psychic_terrain_start = battle.fields.get(Field.PSYCHIC_TERRAIN, -1)
                trick_room_start = battle.fields.get(Field.TRICK_ROOM, -1)

                if grassy_terrain_start >= 0:
                    grassy_terrain_turns = 5 - (battle.turn - grassy_terrain_start)
                elif psychic_terrain_start >= 0:
                    psychic_terrain_turns = 5 - (battle.turn - psychic_terrain_start)

                if trick_room_start >= 0:
                    trick_room_turns = 5 - (battle.turn - trick_room_start)

            field_row[0] = trick_room_turns
            field_row[1] = grassy_terrain_turns
            field_row[2] = psychic_terrain_turns

            sun_turns = 0
            rain_turns = 0
            if battle._weather:
                rain_start = battle._weather.get(Weather.RAINDANCE, -1)
                sun_start = battle._weather.get(Weather.SUNNYDAY, -1)
                if rain_start >= 0:
                    rain_turns = 5 - (battle.turn - rain_start)
                elif sun_start >= 0:
                    sun_turns = 5 - (battle.turn - sun_start)

            field_row[3] = sun_turns
            field_row[4] = rain_turns

        # Local effects
        tailwind_turns = 0
        veil_turns = 0
        if battle.side_conditions:
            tailwind_start = battle.side_conditions.get(SideCondition.TAILWIND, -1)
            veil_start = battle.side_conditions.get(SideCondition.AURORA_VEIL, -1)
            if tailwind_start >= 0:
                tailwind_turns = 4 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        state[0, 0, 5] = tailwind_turns
        state[0, 0, 6] = veil_turns

        tailwind_turns = 0
        veil_turns = 0
        if battle.opponent_side_conditions:
            tailwind_start = battle.opponent_side_conditions.get(SideCondition.TAILWIND, -1)
            veil_start = battle.opponent_side_conditions.get(SideCondition.AURORA_VEIL, -1)
            if tailwind_start >= 0:
                tailwind_turns = 4 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        state[1, 0, 5] = tailwind_turns
        state[1, 0, 6] = veil_turns

        p1_fainted_count = 0
        p1_rage_fist_stacks = 0  # TODO: calculate this somehow
        active_slot, bench_slot = 1, 4

        for mon in battle.team.values():
            if mon.active:
                Encoder._encode_pokemon(mon, state[0, active_slot])
                active_slot += 1
            else:
                Encoder._encode_pokemon(mon, state[0, bench_slot])
                bench_slot -= 1
                if mon.fainted:
                    p1_fainted_count += 1

        state[0, 0, 7] = p1_fainted_count
        state[0, 0, 8] = p1_rage_fist_stacks

        p2_fainted_count = 0
        p2_rage_fist_stacks = 0  # TODO: calculate this somehow
        active_slot, bench_slot = 1, 4

        for mon in battle.opponent_team.values():
            if mon.active:
                Encoder._encode_pokemon(mon, state[1, active_slot])
                active_slot += 1
            else:
                Encoder._encode_pokemon(mon, state[1, bench_slot])
                bench_slot -= 1
                if mon.fainted:
                    p2_fainted_count += 1

        state[1, 0, 7] = p2_fainted_count
        state[1, 0, 8] = p2_rage_fist_stacks

    @staticmethod
    def get_action_mask(battle: AbstractBattle):
        """
        Returns a [2, ACT_SIZE] action mask for both active Pokémon.
        Each row is a mask for the legal actions of that Pokémon.
        """
        assert isinstance(battle, DoubleBattle)

        # direct copy of vgc-bench action mask
        def single_action_mask(battle: DoubleBattle, pos: int) -> list[int]:
            switch_space = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if not battle.trapped[pos]
                and pokemon.base_species in [p.base_species for p in battle.available_switches[pos]]
            ]
            active_mon = battle.active_pokemon[pos]
            if battle._wait or (any(battle.force_switch) and not battle.force_switch[pos]):
                actions = [0]
            elif all(battle.force_switch) and len(battle.available_switches[0]) == 1:
                actions = switch_space + [0]
            elif battle.teampreview or active_mon is None:
                actions = switch_space
            else:
                move_spaces = [
                    [7 + 5 * i + j + 2 for j in battle.get_possible_showdown_targets(move, active_mon)]
                    for i, move in enumerate(active_mon.moves.values())
                    if move.id in [m.id for m in battle.available_moves[pos]]
                ]
                move_space = [i for s in move_spaces for i in s]
                tera_space = [i + 20 for i in move_space if battle.can_tera[pos]]
                if (
                    not move_space
                    and len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                ):
                    move_space = [9]
                actions = switch_space + move_space + tera_space
            actions = actions or [0]
            action_mask = [int(i in actions) for i in range(ACT_SIZE)]
            return action_mask

        # Stack for both active Pokémon (positions 0 and 1)
        mask0 = single_action_mask(battle, 0)
        mask1 = single_action_mask(battle, 1)
        # Return as a torch tensor of shape (2, ACT_SIZE)
        return torch.tensor([mask0, mask1], dtype=torch.uint8)
