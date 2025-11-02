import torch
from poke_env.battle import Battle, SideCondition, Weather, Field, Effect
from poke_env.battle.pokemon import Pokemon
from lookups import POKEMON


class Encoder:
    def _get_pokemon_id(self, pokemon: Pokemon) -> int:
        if not pokemon:
            return -1

        # Get move list in the exact order they appear.
        move_ids = [move.id for move in pokemon.moves.values()]
        move_list_str = ",".join(move_ids)
        return POKEMON.get(move_list_str, -1)

    def _encode_pokemon(self, pokemon: Pokemon | None, pokemon_row: torch.Tensor):
        if pokemon is None:
            pokemon_row[0] = -1 # set ID to -1 to imply unknown
            return

        pokemon_row[0] = self._get_pokemon_id(pokemon)

        pokemon_row[1] = pokemon._type_1.value
        pokemon_row[2] = 0 if pokemon._type_2 is None else pokemon._type_2.value

        pokemon_row[3] = 1 if pokemon.is_terastallized() else 0
        pokemon_row[4] = 1 if pokemon.item() else 0

        pokemon_row[5] = 0 if pokemon._status is None else pokemon._status.value

        # Volatile statuses: taunt, encore, confusion
        curr_effects = pokemon._effects
        pokemon_row[6] = 1 if Effect.TAUNT in curr_effects else 0
        pokemon_row[7] = 1 if Effect.ENCORE in curr_effects else 0
        pokemon_row[8] = 1 if Effect.CONFUSION in curr_effects else 0

        pokemon_row[9] = pokemon._current_hp

        stats = ["atk", "def", "spa", "spd", "spe"]
        for i, stat in enumerate(stats):
            pokemon_row[10 + i] = pokemon._base_stats[stat]

        boosts = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for i, boost in enumerate(boosts):
            pokemon_row[15 + i] = pokemon._boosts[boost]

    def encode_battle_state(self, battle: Battle, state: torch.Tensor):
        """
        Fills the state tensor with the current state of the battle.

        Args:
            battle (Battle): The battle object from poke-env.
            state (torch.Tensor): The 2x5x22 tensor to be filled.
        """
        state.zero_()  # Reset tensor to zeros

        # Encode global field conditions for both players
        for player_idx in range(2):
            field_row = state[player_idx, 0]

            trick_room_turns = 0
            grassy_terrain_turns = 0
            psychic_terrain_turns = 0
            if battle._fields:
                grassy_terrain_start = battle._fields.get(Field.GRASSY_TERRAIN, -1)
                psychic_terrain_start = battle._fields.get(Field.PSYCHIC_TERRAIN, -1)
                trick_room_start = battle._fields.get(Field.TRICK_ROOM, -1)

                if grassy_terrain_start >= 0:
                    grassy_terrain_turns = 5 - (battle.turn - grassy_terrain_start)
                elif psychic_terrain_start >= 0:
                    psychic_terrain_turns = 5 - (
                        battle.turn - psychic_terrain_start
                    )

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
        if battle._side_conditions:
            tailwind_start = battle._side_conditions.get(SideCondition.TAILWIND, -1)
            veil_start = battle._side_conditions.get(SideCondition.AURORA_VEIL, -1)
            if tailwind_start >= 0:
                tailwind_turns = 4 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        state[0, 0, 5] = tailwind_turns
        state[0, 0, 6] = veil_turns

        tailwind_turns = 0
        veil_turns = 0
        if battle._opponent_side_conditions:
            tailwind_start = battle._opponent_side_conditions.get(
                SideCondition.TAILWIND, -1
            )
            veil_start = battle._opponent_side_conditions.get(
                SideCondition.AURORA_VEIL, -1
            )
            if tailwind_start >= 0:
                tailwind_turns = 4 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        state[1, 0, 5] = tailwind_turns
        state[1, 0, 6] = veil_turns

        # TODO: fetch the correct pokemon for the player
        # and the opponent (None is opponents pokemon is unknown)
