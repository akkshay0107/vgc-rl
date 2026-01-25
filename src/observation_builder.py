import json

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from transformers import BertModel, BertTokenizerFast

from lookups import (
    EFFECT_DESCRIPTION,
    ITEM_DESCRIPTION,
    MOVES,
    POKEMON,
    POKEMON_DESCRIPTION,
    STATUS_DESCRIPTION,
)

TINYBERT_SZ = 624
EXTRA_SZ = 28
OBS_DIM = (38, TINYBERT_SZ)  # 1 field row + 1 info row + 3 tokens * 12 pokemon

# Define action space parameters (from gen9vgcenv.py)
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)

tokenizer = BertTokenizerFast.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


def _get_pokemon_text(pokemon: Pokemon, cond: int) -> tuple[str, str]:
    # information about pokemon position
    if cond == -1:
        cond_str = "This Pokemon is DROPPED. It is not part of the battle."
    elif cond == 0:
        cond_str = "This pokemon MAY or MAY NOT be in the back as a switch."
    elif cond == 1:
        cond_str = "This pokemon IS ACTIVE. It is currently on the field."
    elif cond == 2:
        cond_str = "This pokemon is IN THE BACK. It is able to switch in."
    elif cond == 3:
        cond_str = "This pokemon has FAINTED. It no longer participates in the battle."
    elif cond == 4:
        cond_str = "This pokemon CANNOT BE SWITCHED IN. May or may not be in team."
    else:
        cond_str = "We do not know about this pokemon."

    movelist = list(pokemon.moves.keys())
    joint_movelist = ",".join(movelist)
    id = POKEMON[joint_movelist]

    pokemon_desc = POKEMON_DESCRIPTION[id]

    def get_move_desc(move: str, desc) -> str:
        return move + ":" + json.dumps(desc, separators=(",", ":"))

    moves_desc = " ".join([get_move_desc(move, MOVES[move]) for move in movelist])

    item_desc = (
        "Holds no item."
        if (pokemon.item is None or pokemon.item == "")
        else ITEM_DESCRIPTION[pokemon.item]
    )

    status_desc = (
        "No status condition." if pokemon.status is None else STATUS_DESCRIPTION[pokemon.status]
    )

    def describe_effect(effect: Effect, turns: int) -> str:
        effect_desc = EFFECT_DESCRIPTION.get(effect)
        if effect_desc is not None:
            return f"{effect_desc}. Has been active for {turns} turns."
        else:
            return ""

    effect_desc = " ".join(
        [describe_effect(effect, turn) for effect, turn in pokemon.effects.items()]
    )

    first_turn_in = (
        "Can use first turn only moves."
        if pokemon.first_turn
        else "Cannot use first turn only moves."
    )

    first_half = cond_str + pokemon_desc + moves_desc
    second_half = item_desc + status_desc + effect_desc + first_turn_in

    return first_half, second_half


def _get_pokemon_obs(
    pokemon: Pokemon, battle: DoubleBattle, cond: int
) -> tuple[tuple[str, str], list[float]]:
    """
    cond indicates whether we know if pokemon is active, benched, dropped, fainted or unknown
    -1 = dropped
    0 = unknown
    1 = active
    2 = benched
    3 = fainted
    4 = stuck out (dropped from own team / pokemon inside is trapped)
    """
    # Text input for each pokemon
    pokemon_str = _get_pokemon_text(pokemon, cond)

    # Extra inputs for each pokemon (roughly normalized to [0,1])
    pokemon_row = [0.0] * EXTRA_SZ
    pokemon_row[0] = pokemon.type_1.value / 18.0
    pokemon_row[1] = 0.0 if pokemon.type_2 is None else pokemon.type_2.value / 18.0
    pokemon_row[2] = 0.0 if not pokemon.is_terastallized else pokemon.tera_type.value / 18.0  # type: ignore

    pokemon_row[3] = pokemon.current_hp_fraction if pokemon.current_hp is not None else 0.0

    stats = ["hp", "atk", "def", "spa", "spd", "spe"]
    for i, stat in enumerate(stats):
        pokemon_row[4 + i] = pokemon.base_stats[stat] / 200.0

    boosts = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    for i, boost in enumerate(boosts):
        pokemon_row[10 + i] = pokemon.boosts[boost] / 6.0

    for i, move in enumerate(pokemon.moves):
        pokemon_row[17 + i] = pokemon.moves[move].current_pp / pokemon.moves[move].max_pp

    pokemon_row[21] = pokemon.protect_counter / 3.0

    pokemon_row[22] = float(pokemon.first_turn)
    curr_effects = pokemon.effects
    pokemon_row[23] = curr_effects.get(Effect.TAUNT, 0) / 3.0
    pokemon_row[24] = curr_effects.get(Effect.ENCORE, 0) / 3.0
    pokemon_row[25] = 1.0 if Effect.CONFUSION in curr_effects else 0.0
    pokemon_row[26] = curr_effects.get(Effect.YAWN, 0)

    pokemon_row[27] = pokemon.weight / 300.0  # heaviest pokemon is ursa bm at 330

    return pokemon_str, pokemon_row


def _get_team_obs(
    battle: DoubleBattle,
) -> tuple[list[tuple[str, str]], list[float], list[tuple[str, str]], list[float]]:
    p1_mon_txt = []
    p1_mon_arr = []
    possible_switches = {mon for switches in battle.available_switches for mon in switches}

    for mon in battle.team.values():
        if mon.fainted:
            cond = 3
        elif mon in battle.active_pokemon:
            cond = 1
        elif mon in possible_switches:
            cond = 2
        else:
            cond = 4
        mon_txt, mon_arr = _get_pokemon_obs(mon, battle, cond)
        p1_mon_txt.append(mon_txt)
        p1_mon_arr.append(mon_arr)

    p2_mon_txt = []
    p2_mon_arr = []
    revealed = [mon for mon in battle.opponent_team.values() if mon.revealed]

    for mon in battle.opponent_team.values():
        if mon.fainted:
            cond = 3
        elif mon in battle.opponent_active_pokemon:
            cond = 1
        elif mon.revealed:
            cond = 2
        elif len(revealed) == 4:
            cond = -1
        else:
            cond = 0
        mon_txt, mon_arr = _get_pokemon_obs(mon, battle, cond)
        p2_mon_txt.append(mon_txt)
        p2_mon_arr.append(mon_arr)

    return p1_mon_txt, p1_mon_arr, p2_mon_txt, p2_mon_arr


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
    p1_row = [0.0] * 7
    p2_row = [0.0] * 7

    # Global effects
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

    sun_turns = 0
    rain_turns = 0
    if battle._weather:
        rain_start = battle._weather.get(Weather.RAINDANCE, -1)
        sun_start = battle._weather.get(Weather.SUNNYDAY, -1)
        if rain_start >= 0:
            rain_turns = 5 - (battle.turn - rain_start)
        elif sun_start >= 0:
            sun_turns = 5 - (battle.turn - sun_start)

    global_effects = [
        trick_room_turns,
        grassy_terrain_turns,
        psychic_terrain_turns,
        sun_turns,
        rain_turns,
    ]
    for i in range(5):
        p1_row[i] = global_effects[i]
        p2_row[i] = global_effects[i]

    # Player 1 local effects
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

    # Player 2 local effects
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


def _get_field_text(battle: DoubleBattle) -> str:
    header = (
        "Effects: Trick Room (reverses speed order), "
        "Grassy Terrain (heals 1/16 of max HP for grounded Pokémon and boosts Grass moves by 30%), "
        "Psychic Terrain (prevents priority moves by grounded Pokémon and boosts Psychic moves by 30%), "
        "Sunny Weather (boosts Fire moves by 50%, weakens Water moves by 50%), "
        "Rainy Weather (boosts Water moves by 50%, weakens Fire moves by 50%), "
        "Tailwind (doubles team speed), "
        "Aurora Veil (reduces damage taken by team by 33%)."
    )

    p1_row, p2_row = _get_locals(battle)

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


def _get_info_text(p1_tera: Pokemon | None, p2_tera: Pokemon | None) -> str:
    # As of right now this just stores the global tera information
    # realistically this can be extended to store more information
    # like speed order of pokemon so far

    if p1_tera is None:
        p1_str = "You have not terastallized yet."
    else:
        p1_str = f"You have terastallized your {p1_tera.species} into the {p1_tera.tera_type} type. You cannot terastallize any other pokemon."

    if p2_tera is None:
        p2_str = "The opponent has not terastallized yet."
    else:
        p2_str = f"The opponent has terastallized their {p2_tera.species} into the {p2_tera.tera_type} type. They cannot terastallize any other pokemon."

    return p1_str + p2_str


def from_battle(battle: AbstractBattle):
    assert isinstance(battle, DoubleBattle)
    if battle.teampreview:
        return torch.rand(OBS_DIM, dtype=torch.float32)

    p1_txt, p1_arr, opp_txt, opp_arr = _get_team_obs(battle)
    field_txt = _get_field_text(battle)

    p1_tera = None
    for mon in battle.team.values():
        if mon.is_terastallized:
            p1_tera = mon
            break

    opp_tera = None
    for mon in battle.opponent_team.values():
        if mon.is_terastallized:
            opp_tera = mon
            break

    info_txt = _get_info_text(p1_tera, opp_tera)

    # combine the stuff above into one observation
    # row 0: pad p1 and opp locals into a string
    # row 1: extra information (for now tera usage)
    # row 2-25: 2 rows per pokemon in players team and opponents team (text)
    # row 26-37: 1 rows per pokemon in players team and opponents team (numeric)
    p1_flat = [text for pair in p1_txt for text in pair]
    opp_flat = [text for pair in opp_txt for text in pair]
    text_inputs = [field_txt, info_txt, *p1_flat, *opp_flat]

    # cls mean concat
    encoded = tokenizer(
        text_inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state  # (26, seq_len, 312)
    cls_emb = last_hidden[:, 0, :]  # (26, 312)
    # Exclude padding tokens for mean pooling
    mask = encoded["attention_mask"].unsqueeze(-1)  # (26, seq_len, 1)
    masked_hidden = last_hidden * mask
    sum_hidden = masked_hidden.sum(dim=1)
    len_nonpad = mask.sum(dim=1).clamp(min=1)  # avoid div by zero
    mean_emb = sum_hidden / len_nonpad  # (26, 312)
    text_emb = torch.cat([cls_emb, mean_emb], dim=-1)  # (26, 624)

    num_emb = torch.tensor(p1_arr + opp_arr)
    num_emb = torch.cat([num_emb, torch.zeros((12, TINYBERT_SZ - EXTRA_SZ))], dim=1)

    return torch.cat([text_emb, num_emb], dim=0)


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
