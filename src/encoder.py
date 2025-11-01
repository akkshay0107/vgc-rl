import torch
from poke_env.battle import Battle, SideCondition, Weather, Field
from poke_env.battle.pokemon import Pokemon
from lookups import POKEMON, NON_VOLATILE_STATUS, TYPES


class Encoder:
    def _get_pokemon_id(self, pokemon: Pokemon) -> int:
        if not pokemon:
            return -1

        # Get move list in the exact order they appear.
        move_ids = [move.id for move in pokemon.moves.values()]
        move_list_str = ",".join(move_ids)
        return POKEMON.get(move_list_str, -1)

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
        if battle._side_conditions:
            tailwind_start = battle._side_conditions.get(SideCondition.TAILWIND, -1)
            veil_start = battle._side_conditions.get(SideCondition.AURORA_VEIL, -1)
            if tailwind_start >= 0:
                tailwind_turns = 5 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        state[0,0,5] = tailwind_turns
        state[0,0,6] = veil_turns

        tailwind_turns = 0
        veil_turns = 0
        if battle._opponent_side_conditions:
            tailwind_start = battle._opponent_side_conditions.get(SideCondition.TAILWIND, -1)
            veil_start = battle._opponent_side_conditions.get(SideCondition.AURORA_VEIL, -1)
            if tailwind_start >= 0:
                tailwind_turns = 5 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        state[1,0,5] = tailwind_turns
        state[1,0,6] = veil_turns


        for player_idx, player in enumerate([battle.player, battle.opponent]):
            for i in range(4):  # 4 pokemon per player on the field in VGC
                if i < len(player.team):
                    pokemon = list(player.team.values())[i]
                    pokemon_row = state[player_idx, i + 1]

                    # Col 0 - pokemonID
                    pokemon_row[0] = self._get_pokemon_id(pokemon)

                    # Col 1-2 - typing
                    if pokemon.types:
                        pokemon_row[1] = TYPES.get(pokemon.types[0].name.capitalize(), 0)
                        if len(pokemon.types) > 1:
                            pokemon_row[2] = TYPES.get(pokemon.types[1].name.capitalize(), 0)

                    # Col 3 - tera burnt
                    pokemon_row[3] = 1 if pokemon.terastallized else 0

                    # Col 4 - item held / consumed or knocked off (1 if item is held, 0 otherwise)
                    pokemon_row[4] = 1 if pokemon.item else 0

                    # Col 5 - non volatile status
                    pokemon_row[5] = (
                        NON_VOLATILE_STATUS.get(pokemon.status.name.lower(), 0)
                        if pokemon.status
                        else 0
                    )

                    # Col 6-8 - volatile statuses
                    pokemon_row[6] = 1 if "taunt" in pokemon.volatile_status else 0
                    pokemon_row[7] = 1 if "encore" in pokemon.volatile_status else 0
                    pokemon_row[8] = 1 if "confusion" in pokemon.volatile_status else 0

                    # Col 9 - current HP stat
                    pokemon_row[9] = pokemon.current_hp

                    # Col 10-14 - base stats (excluding HP)
                    pokemon_row[10] = pokemon.base_stats["atk"]
                    pokemon_row[11] = pokemon.base_stats["def"]
                    pokemon_row[12] = pokemon.base_stats["spa"]
                    pokemon_row[13] = pokemon.base_stats["spd"]
                    pokemon_row[14] = pokemon.base_stats["spe"]

                    # Col 15-21 - stat stages
                    pokemon_row[15] = pokemon.boosts["atk"]
                    pokemon_row[16] = pokemon.boosts["def"]
                    pokemon_row[17] = pokemon.boosts["spa"]
                    pokemon_row[18] = pokemon.boosts["spd"]
                    pokemon_row[19] = pokemon.boosts["spe"]
                    pokemon_row[20] = pokemon.boosts["accuracy"]
                    pokemon_row[21] = pokemon.boosts["evasion"]
