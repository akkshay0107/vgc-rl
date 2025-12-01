import json

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from transformers import BertModel, BertTokenizer

from lookups import ITEM_DESCRIPTION, MOVES, POKEMON, POKEMON_DESCRIPTION

OBS_DIM = (13, 650)  # 624 from TinyBERT and 26 for additional information
# Define action space parameters (from gen9vgcenv.py)
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)


class Encoder:
    """
    Static library class containing methods to
    - Get the set of all valid actions from the current battle state
    - Encode the battle state into an observation for the policy network
    """

    tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    @staticmethod
    def _get_pokemon_as_text(pokemon: Pokemon, status: int) -> str:
        # building the text description for each pokemon.
        if status == -1:
            status_str = "This Pokemon is DROPPED. It is not part of the battle."
        elif status == 0:
            status_str = "This pokemon MAY or MAY NOT be in the back as a switch."
        elif status == 1:
            status_str = "This pokemon IS ACTIVE. It is currently on the field."
        elif status == 2:
            status_str = "This pokemon is IN THE BACK. It is able to switch in."
        elif status == 3:
            status_str = "This pokemon has FAINTED. It no longer participates in the battle."
        elif status == 4:
            status_str = "This pokemon CANNOT BE SWITCHED IN. May or may not be in team."
        else:
            status_str = "Unknown status."

        movelist = list(pokemon.moves.keys())
        if not movelist:
            return status_str + " Moves are unknown."

        joint_movelist = ",".join(movelist)
        id = POKEMON[joint_movelist]

        desc = POKEMON_DESCRIPTION[id]

        if pokemon.item is None:
            item_desc = "Holds no item."
        else:
            item_desc = ITEM_DESCRIPTION[id]

        get_move_desc = lambda move, desc: move + ":" + json.dumps(desc, separators=(",", ":"))  # noqa: E731

        moves_desc = [get_move_desc(move, MOVES[move]) for move in movelist]

        return status_str + desc + item_desc + "Has the following moves: " + ". ".join(moves_desc)

    @staticmethod
    def _encode_pokemon(
        pokemon: Pokemon, battle: DoubleBattle, status: int
    ) -> tuple[str, list[float]]:
        """
        status indicates whether we know if pokemon is active, benched, dropped, fainted or unknown
        -1 = dropped
        0 = unknown
        1 = active
        2 = benched
        3 = fainted
        4 = stuck out (dropped from own team / pokemon inside is trapped)
        """
        # Text input for each pokemon
        pokemon_str = Encoder._get_pokemon_as_text(pokemon, status)

        # Array inputs for each pokemon (roughly normalize to [0,1])
        pokemon_row = [0.0] * 26
        pokemon_row[0] = pokemon.type_1.value / 18
        pokemon_row[1] = 0 if pokemon.type_2 is None else pokemon.type_2.value / 18

        pokemon_row[2] = 0 if not pokemon.is_terastallized else pokemon.tera_type.value / 18  # type: ignore
        pokemon_row[3] = 1 if pokemon.item else 0

        curr_effects = pokemon.effects
        pokemon_row[4] = curr_effects.get(Effect.TAUNT, 0) / 3
        pokemon_row[5] = curr_effects.get(Effect.ENCORE, 0) / 3
        pokemon_row[6] = 1 if Effect.CONFUSION in curr_effects else 0

        pokemon_row[7] = pokemon.current_hp_fraction if pokemon.current_hp is not None else 0

        stats = ["hp", "atk", "def", "spa", "spd", "spe"]
        for i, stat in enumerate(stats):
            pokemon_row[8 + i] = pokemon.base_stats[stat] / 200

        boosts = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for i, boost in enumerate(boosts):
            pokemon_row[14 + i] = pokemon.boosts[boost] / 6

        for i, move in enumerate(pokemon.moves):
            pokemon_row[21 + i] = pokemon.moves[move].current_pp / pokemon.moves[move].max_pp

        pokemon_row[25] = pokemon.protect_counter / 3

        return pokemon_str, pokemon_row

    @staticmethod
    def _get_description(
        battle: DoubleBattle,
    ) -> tuple[list[str], list[float], list[str], list[float]]:
        p1_mon_txt = []
        p1_mon_arr = []
        possible_switches = set()
        for switches in battle.available_switches:
            for mon in switches:
                possible_switches.add(mon)

        for mon in battle.team.values():
            if mon.fainted:
                status = 3
            elif mon in battle.active_pokemon:
                status = 1
            elif mon in possible_switches:
                status = 2
            else:
                status = 4
            mon_txt, mon_arr = Encoder._encode_pokemon(mon, battle, status)
            p1_mon_txt.append(mon_txt)
            p1_mon_arr.append(mon_arr)

        p2_mon_txt = []
        p2_mon_arr = []
        revealed = [mon for mon in battle.opponent_team.values() if mon.revealed]

        for mon in battle.opponent_team.values():
            if mon.fainted:
                status = 3
            elif mon in battle.opponent_active_pokemon:
                status = 1
            elif mon.revealed:
                status = 2
            elif len(revealed) == 4:
                status = -1
            else:
                status = 0
            mon_txt, mon_arr = Encoder._encode_pokemon(mon, battle, status)
            p2_mon_txt.append(mon_txt)
            p2_mon_arr.append(mon_arr)

        return p1_mon_txt, p1_mon_arr, p2_mon_txt, p2_mon_arr

    @staticmethod
    def _get_locals(battle: DoubleBattle):
        """
        For each side,
        0 = trick room turns
        1 = grassy terrain turns
        2 = psychic terrain turns
        3 = sun turns
        4 = rain turns
        5 = tailwind turns
        6 = aurora veil turns
        """
        # Encode global field conditions for both players
        p1_row = [0.0] * 7
        p2_row = [0.0] * 7
        for field_row in [p1_row, p2_row]:
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
        p1_row[5] = tailwind_turns
        p1_row[6] = veil_turns

        tailwind_turns = 0
        veil_turns = 0
        if battle.opponent_side_conditions:
            tailwind_start = battle.opponent_side_conditions.get(SideCondition.TAILWIND, -1)
            veil_start = battle.opponent_side_conditions.get(SideCondition.AURORA_VEIL, -1)
            if tailwind_start >= 0:
                tailwind_turns = 4 - (battle.turn - tailwind_start)
            if veil_start >= 0:
                veil_turns = 5 - (battle.turn - veil_start)
        p2_row[5] = tailwind_turns
        p2_row[6] = veil_turns

        return p1_row, p2_row

    @staticmethod
    def _get_locals_as_text(battle: DoubleBattle) -> str:
        p1_row, opp_row = Encoder._get_locals(battle)
        header = (
            "Effects: Trick Room (reverses speed order), "
            "Grassy Terrain (heals 1/16 of max HP for grounded Pokémon and boosts Grass moves by 30%), "
            "Psychic Terrain (prevents priority moves by grounded Pokémon and boosts Psychic moves by 30%), "
            "Sunny Weather (boosts Fire moves by 50%, weakens Water moves by 50%), "
            "Rainy Weather (boosts Water moves by 50%, weakens Fire moves by 50%), "
            "Tailwind (doubles team speed), "
            "Aurora Veil (reduces damage taken by team by 33%)."
        )

        p1_row, p2_row = Encoder._get_locals(battle)

        effects = [
            "Trick Room",
            "Grassy Terrain",
            "Psychic Terrain",
            "Sunny Weather",
            "Rainy Weather",
            "Tailwind",
            "Aurora Veil",
        ]

        def describe_side(name_suffix, row):
            parts = []
            for name, turns in zip(effects, row):
                if turns > 0:
                    parts.append(f"{name} active for {int(turns)} more turns")
                else:
                    parts.append(f"{name} inactive")
            return f"{name_suffix} has: " + ". ".join(parts)

        p1_desc = describe_side("Player 1", p1_row)
        p2_desc = describe_side("Player 2", p2_row)

        return header + " " + p1_desc + ". " + p2_desc + "."

    @staticmethod
    def encode_battle_state(battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle.teampreview:
            return torch.rand(OBS_DIM, dtype=torch.float32)

        p1_txt, p1_arr, opp_txt, opp_arr = Encoder._get_description(battle)
        field_conditions = Encoder._get_locals_as_text(battle)

        # concatenate the output from the CLS token and the mean of all tokens in the sequence
        def get_cls_mean_concat(text: str, max_len: int = 512) -> torch.Tensor:
            encoded = Encoder.tokenizer(
                text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = Encoder.model(**encoded)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, 312)
            cls_emb = last_hidden[:, 0, :]  # (1, 312)
            # Exclude padding tokens for mean pooling
            mask = encoded["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (1, 312)
            len_nonpad = mask.sum(dim=1).clamp(min=1)  # avoid div by zero
            mean_emb = sum_hidden / len_nonpad  # (1, 312)
            concat_emb = torch.cat([cls_emb, mean_emb], dim=-1)  # (1, 624)
            return concat_emb

        # combine the stuff above into one observation
        # row 0: pad p1 and opp locals into a string and pass it through encoder (1, 624)
        # row 1-6: pokemons for p1 (6, 650)
        # rows 7-12: pokemons for p2 (6, 650)

        all_embeddings = []

        field_emb = get_cls_mean_concat(field_conditions)
        all_embeddings.append(
            torch.cat(
                [
                    field_emb,  # (1, 624)
                    torch.zeros(
                        (1, 26),
                        dtype=field_emb.dtype,
                        device=field_emb.device,
                    ),
                ],
                dim=-1,
            )  # (1, 650)
        )

        for mon_txt, mon_arr in zip(p1_txt, p1_arr):
            emb = get_cls_mean_concat(mon_txt)
            extra = torch.Tensor(mon_arr, device=emb.device).unsqueeze(0)
            all_embeddings.append(torch.cat([emb, extra], dim=1))

        for mon_txt, mon_arr in zip(opp_txt, opp_arr):
            emb = get_cls_mean_concat(mon_txt)
            extra = torch.Tensor(mon_arr, device=emb.device).unsqueeze(0)
            all_embeddings.append(torch.cat([emb, extra], dim=1))

        return torch.cat(all_embeddings, dim=0)  # should be (13, 650)

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
                    [
                        7 + 5 * i + j + 2
                        for j in battle.get_possible_showdown_targets(move, active_mon)
                    ]
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
