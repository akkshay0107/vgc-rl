import re
from functools import lru_cache

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from transformers import BertModel, BertTokenizerFast

from lookups import (
    ACT_SIZE,
    EFFECT_DESCRIPTION,
    EXTRA_SZ,
    ITEM_DESCRIPTION,
    MOVES,
    POKEMON,
    POKEMON_DESCRIPTION,
    STATUS_DESCRIPTION,
    TINYBERT_SZ,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)  # type: ignore
model.eval()
# Pre-compiled constants and regexes for optimization
CONDITION_DESC = {
    -1: "This Pokemon is DROPPED. It is not part of the battle.",
    0: "This pokemon MAY or MAY NOT be in the back as a switch.",
    1: "This pokemon IS ACTIVE. It is currently on the field.",
    2: "This pokemon is IN THE BACK. It is able to switch in.",
    3: "This pokemon has FAINTED. It no longer participates in the battle.",
    4: "This pokemon CANNOT BE SWITCHED IN. May or may not be in team.",
}
DEFAULT_CONDITION_DESC = "We do not know about this pokemon."

# Regex patterns for _get_turn_summary
MOVE_RE = re.compile(r"\|move\|([^|]+)\|([^|]+)")
DAMAGE_RE = re.compile(r"\|-damage\|([^|]+)\|([^|]+)")
FAINT_RE = re.compile(r"\|faint\|([^|]+)")
STATUS_RE = re.compile(r"\|-status\|([^|]+)\|([^|]+)")
BOOST_RE = re.compile(r"\|-(boost|unboost)\|([^|]+)\|([^|]+)\|([^|]+)")
ABILITY_RE = re.compile(r"\|-ability\|([^|]+)\|([^|]+)")
TERA_RE = re.compile(r"\|-terastallize\|([^|]+)\|([^|]+)")
CLEAN_ID_RE = re.compile(r"[^a-z0-9]")


def _to_id_str(s: str) -> str:
    return CLEAN_ID_RE.sub("", s.lower())


def _get_last_move(battle: DoubleBattle, pokemon: Pokemon) -> str | None:
    # Check current turn's events first
    for event in reversed(battle.current_observation.events):
        if event[1] == "move":
            try:
                event_mon = battle.get_pokemon(event[2])
                if event_mon == pokemon:
                    move_name = event[3]
                    return _to_id_str(move_name)
            except Exception:
                continue

    # Check observations from previous turns
    for turn in range(battle.turn, 0, -1):
        if turn not in battle.observations:
            continue
        obs = battle.observations[turn]
        for event in reversed(obs.events):
            if event[1] == "move":
                try:
                    event_mon = battle.get_pokemon(event[2])
                    if event_mon == pokemon:
                        move_name = event[3]
                        return _to_id_str(move_name)
                except Exception:
                    continue
    return None


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    # normalized turns left
    if start_turn < 0:
        return 0
    val = max(0, duration - (battle.turn - start_turn))
    return val / float(duration)


def _get_pokemon_text(
    pokemon: Pokemon | None, cond: int, last_move_id: str | None
) -> tuple[str, str]:
    if pokemon is None:
        return "This slot is empty.", "No information available."

    # information about pokemon position
    cond_str = CONDITION_DESC.get(cond, DEFAULT_CONDITION_DESC)

    movelist = list(pokemon.moves.keys())
    joint_movelist = ",".join(movelist)
    id = POKEMON.get(joint_movelist, 0)

    pokemon_desc = POKEMON_DESCRIPTION.get(id, "Unknown Pokemon description.")

    moves_desc = " ".join([f"{m}:{MOVES.get(m, 'No details.')}" for m in movelist])

    item_desc = (
        ITEM_DESCRIPTION.get(pokemon.item, "Holds no item.")
        if pokemon.item and pokemon.item in ITEM_DESCRIPTION
        else "Holds no item."
    )

    status_desc = (
        STATUS_DESCRIPTION.get(pokemon.status, "No status condition.")
        if pokemon.status
        else "No status condition."
    )

    def describe_effect(effect: Effect, turns: int) -> str:
        desc = EFFECT_DESCRIPTION.get(effect)
        return f"{desc}. Has been active for {turns} turns." if desc else ""

    effect_desc = " ".join(
        [describe_effect(effect, turn) for effect, turn in pokemon.effects.items()]
    )

    first_turn_in = (
        "Can use first turn only moves."
        if pokemon.first_turn
        else "Cannot use first turn only moves."
    )

    last_move_desc = (
        f" The last move this Pokemon used was {last_move_id}."
        if last_move_id
        else " This Pokemon has not used a move yet."
    )

    first_half = f"{cond_str}{pokemon_desc}{item_desc}{status_desc}{effect_desc}{first_turn_in}"
    second_half = f"{moves_desc}{last_move_desc}"

    return first_half, second_half


def _get_pokemon_obs(
    pokemon: Pokemon | None, battle: DoubleBattle, cond: int, orig_idx: int
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
    last_move_id = _get_last_move(battle, pokemon) if pokemon and cond == 1 else None

    # Text input for each pokemon
    pokemon_str = _get_pokemon_text(pokemon, cond, last_move_id)

    # Extra inputs for each pokemon (roughly normalized to [0,1])
    pokemon_row = [0.0] * EXTRA_SZ
    if pokemon is None:
        return pokemon_str, pokemon_row

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

    # 0.1% to get a protect counter of 4 (anything above is nearly impossible)
    pokemon_row[21] = pokemon.protect_counter / 4.0

    pokemon_row[22] = float(pokemon.first_turn)
    curr_effects = pokemon.effects
    pokemon_row[23] = curr_effects.get(Effect.TAUNT, 0) / 3.0
    pokemon_row[24] = curr_effects.get(Effect.ENCORE, 0) / 3.0
    pokemon_row[25] = 1.0 if Effect.CONFUSION in curr_effects else 0.0
    pokemon_row[26] = curr_effects.get(Effect.YAWN, 0) / 2.0

    pokemon_row[27] = pokemon.weight / 300.0  # heaviest pokemon is ursa bm at 330
    pokemon_row[28] = (orig_idx + 1) / 6.0  # Original team index (1-6) or 0 if unknown/opp

    # one hot of last move used
    if last_move_id and pokemon:
        move_ids = list(pokemon.moves.keys())
        if last_move_id in move_ids:
            move_idx = move_ids.index(last_move_id)
            if move_idx < 4:
                pokemon_row[29 + move_idx] = 1.0
            else:
                pokemon_row[33] = 1.0
        else:
            pokemon_row[33] = 1.0
    else:
        pokemon_row[33] = 1.0
    # index 33 is 1 if last move is not known (flinch, switch, etc)

    return pokemon_str, pokemon_row


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool
) -> list[tuple[Pokemon | None, int]]:
    """
    Returns a list of 6 (Pokemon, original_index) tuples.
    Order: Active Slot 0, Active Slot 1, Bench 0, Bench 1, Dropped 0, Dropped 1
    """
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    slot0 = active[0] if active else None
    slot1 = active[1] if len(active) > 1 else None
    assigned = {slot0, slot1} - {None}

    def get_orig_idx(mon):
        if mon is None or is_opponent:
            return -1
        # Map back to original team index
        for i, m in enumerate(battle.team.values()):
            if m == mon:
                return i
        return -1

    others = [(m, get_orig_idx(m)) for m in team.values() if m not in assigned]

    if battle.teampreview:
        # During team preview, show all 6 in a flat list
        res = others + [(None, -1)] * 6
        return res[:6]

    bench, dropped = [], []
    if not is_opponent:
        possible_switches = {mon for switches in battle.available_switches for mon in switches}
        for mon, idx in others:
            if mon.fainted or mon in possible_switches:
                bench.append((mon, idx))
            else:
                dropped.append((mon, idx))
    else:
        # For opponent during battle, we only ever see at most 2 benched pokemon in VGC
        bench = others

    # Pad to exact sizes
    bench = (bench + [(None, -1)] * 2)[:2]
    dropped = (dropped + [(None, -1)] * 2)[:2]

    return [(slot0, get_orig_idx(slot0)), (slot1, get_orig_idx(slot1))] + bench + dropped


def _get_team_obs(battle: DoubleBattle):
    def process_mons(mons):
        txt, arr = [], []
        for i, (mon, idx) in enumerate(mons):
            if mon is None:
                cond = 0
            elif battle.teampreview:
                cond = 2
            elif i < 2:
                cond = 1
            elif mon.fainted:
                cond = 3
            else:
                cond = 2 if i < 4 else -1
            t, a = _get_pokemon_obs(mon, battle, cond, idx)
            txt.append(t)
            arr.append(a)
        return txt, arr

    p1_txt, p1_arr = process_mons(_get_ordered_pokemon(battle, is_opponent=False))
    p2_txt, p2_arr = process_mons(_get_ordered_pokemon(battle, is_opponent=True))

    return p1_txt, p1_arr, p2_txt, p2_arr


def _get_locals(battle: DoubleBattle):
    """
    Returns turn remain counts for various field and side effects.
    """
    # Global effects
    trick_room_turns = _get_turns_left(battle, battle.fields.get(Field.TRICK_ROOM, -1))
    grassy_terrain_turns = _get_turns_left(battle, battle.fields.get(Field.GRASSY_TERRAIN, -1))
    psychic_terrain_turns = _get_turns_left(battle, battle.fields.get(Field.PSYCHIC_TERRAIN, -1))

    rain_turns = _get_turns_left(battle, battle._weather.get(Weather.RAINDANCE, -1))
    sun_turns = _get_turns_left(battle, battle._weather.get(Weather.SUNNYDAY, -1))
    snow_turns = _get_turns_left(battle, battle._weather.get(Weather.SNOW, -1))

    global_effects = [
        trick_room_turns,
        grassy_terrain_turns,
        psychic_terrain_turns,
        sun_turns,
        rain_turns,
        snow_turns,
    ]

    p1_row = global_effects + [
        _get_turns_left(battle, battle.side_conditions.get(SideCondition.TAILWIND, -1), duration=4),
        _get_turns_left(battle, battle.side_conditions.get(SideCondition.AURORA_VEIL, -1)),
    ]

    p2_row = global_effects + [
        _get_turns_left(
            battle, battle.opponent_side_conditions.get(SideCondition.TAILWIND, -1), duration=4
        ),
        _get_turns_left(battle, battle.opponent_side_conditions.get(SideCondition.AURORA_VEIL, -1)),
    ]

    return p1_row, p2_row


def _get_field_text(battle: DoubleBattle) -> str:
    header = (
        "Effects: Trick Room (reverses speed order), "
        "Grassy Terrain (heals 1/16 of max HP for grounded Pokémon and boosts Grass moves by 30%), "
        "Psychic Terrain (prevents priority moves by grounded Pokémon and boosts Psychic moves by 30%), "
        "Sunny Weather (boosts Fire moves by 50%, weakens Water moves by 50%), "
        "Rainy Weather (boosts Water moves by 50%, weakens Fire moves by 50%), "
        "Snowy Weather (boosts Defense of Ice types by 50%), "
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
        "Snowy Weather",
        "Tailwind",
        "Aurora Veil",
    ]

    def describe_side(name_suffix, row):
        parts = [
            f"{name} active for {int(turns)} more turns" if turns > 0 else f"{name} inactive"
            for name, turns in zip(effects, row)
        ]
        return f"{name_suffix} has: {'. '.join(parts)}"

    return f"{header} {describe_side('Player 1', p1_row)}. {describe_side('Player 2', p2_row)}."


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


@lru_cache(maxsize=50_000)
def _encode_one(text: str):
    enc = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    # cls mean concat
    with torch.inference_mode():
        out = model(**enc).last_hidden_state[0]
        cls = out[0]
        mask = enc["attention_mask"][0].unsqueeze(-1)
        mean = (out * mask).sum(0) / mask.sum(0).clamp(min=1)
        emb = torch.cat([cls, mean], dim=-1)
    return emb.float().cpu()


def encode_texts(texts: list[str]):
    return torch.stack([_encode_one(t) for t in texts], dim=0)


def _get_turn_summary(battle: DoubleBattle, turn: int) -> str:
    if turn <= 0:
        return f"Turn {turn}: Battle has not started yet."

    if turn not in battle.observations:
        return f"Turn {turn}: No information available."

    obs = battle.observations[turn]
    # events is List[List[str]]
    messages = ["|".join(e) for e in obs.events]

    summary_parts = []
    for msg in messages:
        if MOVE_RE.search(msg):
            match = MOVE_RE.search(msg)
            if match:
                pkm, move = match.groups()
                summary_parts.append(f"{pkm} used {move}")
        elif DAMAGE_RE.search(msg):
            match = DAMAGE_RE.search(msg)
            if match:
                pkm, health = match.groups()
                summary_parts.append(f"{pkm} took damage ({health})")
        elif FAINT_RE.search(msg):
            match = FAINT_RE.search(msg)
            if match:
                summary_parts.append(f"{match.group(1)} fainted")
        elif STATUS_RE.search(msg):
            match = STATUS_RE.search(msg)
            if match:
                summary_parts.append(f"{match.group(1)} was {match.group(2)}ed")
        elif BOOST_RE.search(msg):
            match = BOOST_RE.search(msg)
            if match:
                verb, pkm, stat, amount = match.groups()
                summary_parts.append(f"{pkm}'s {stat} {verb}ed by {amount}")
        elif ABILITY_RE.search(msg):
            match = ABILITY_RE.search(msg)
            if match:
                summary_parts.append(f"{match.group(1)}'s {match.group(2)} activated")
        elif TERA_RE.search(msg):
            match = TERA_RE.search(msg)
            if match:
                summary_parts.append(f"{match.group(1)} terastallized to {match.group(2)}")

    if not summary_parts:
        return f"Turn {turn}: Ongoing actions."

    return f"Turn {turn}: {'; '.join(summary_parts[:10])}."


def from_battle(battle: AbstractBattle):
    assert isinstance(battle, DoubleBattle)

    p1_txt_pairs, p1_arr, opp_txt_pairs, opp_arr = _get_team_obs(battle)
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

    # Summaries for the last 3 turns
    hist_summaries = [_get_turn_summary(battle, battle.turn - i) for i in [1, 2, 3]]

    p1_field_nums, p2_field_nums = _get_locals(battle)
    tera_flags = [0 if p1_tera is None else 1, 0 if opp_tera is None else 1]

    # 8 (p1 side) + 1 (p1 tera) + 8 (p2 side) + 1 (p2 tera) + 2 (teampreview one-hot) + 1 (turn) = 21 values
    tp_one_hot = [1.0, 0.0] if battle.teampreview else [0.0, 1.0]
    field_num_row_raw = torch.tensor(
        [
            *p1_field_nums,
            tera_flags[0],
            *p2_field_nums,
            tera_flags[1],
            *tp_one_hot,
            battle.turn / 16.0,  # normalized turn count
        ]
    )
    field_num_row = torch.cat(
        [field_num_row_raw, torch.zeros(TINYBERT_SZ - len(field_num_row_raw))]
    )

    p1_flat = [text for pair in p1_txt_pairs for text in pair]  # 12
    opp_flat = [text for pair in opp_txt_pairs for text in pair]  # 12

    text_emb = encode_texts(
        [*hist_summaries, field_txt, info_txt, *p1_flat, *opp_flat]
    )  # 3 + 1 + 1 + 12 + 12 = 29 tokens

    num_rows = torch.tensor(p1_arr + opp_arr)  # 6 + 6 = 12 rows
    num_emb = torch.cat([num_rows, torch.zeros((12, TINYBERT_SZ - EXTRA_SZ))], dim=1)  # 12 rows

    return torch.cat([text_emb, num_emb, field_num_row.unsqueeze(0)], dim=0)


def get_action_mask(battle: AbstractBattle):
    assert isinstance(battle, DoubleBattle)
    if battle.teampreview:
        # Only allow distinct combinations (p1 < p2) to avoid redundant permutations
        # action = (p1-1)*6 + (p2-1)
        mask = [0] * ACT_SIZE
        for a in range(36):
            p1 = a // 6 + 1
            p2 = a % 6 + 1
            if p1 < p2 and p1 <= len(battle.team) and p2 <= len(battle.team):
                mask[a] = 1
        return torch.tensor([mask, mask], dtype=torch.uint8)

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
        elif all(battle.force_switch) and len(battle.available_switches[pos]) == 1:
            actions = switch_space + [0]
        elif active_mon is None:
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

    mask0 = single_action_mask(battle, 0)
    mask1 = single_action_mask(battle, 1)
    return torch.tensor([mask0, mask1], dtype=torch.uint8)
