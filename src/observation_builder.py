import re
from functools import lru_cache
from pathlib import Path

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
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


def get_tinybert_model_and_tokenizer(device):
    """
    Loads TinyBERT model and tokenizer from local 'models/' directory.
    If not found, it downloads them from Hugging Face and saves them locally.
    """
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model_dir = Path(__file__).parent.parent / "models" / "TinyBERT_General_4L_312D"

    if not model_dir.exists():
        print(f"Downloading TinyBERT model to {model_dir}...")
        model_dir.mkdir(parents=True, exist_ok=True)
        _tokenizer = BertTokenizerFast.from_pretrained(model_name)
        _model = BertModel.from_pretrained(model_name)
        _tokenizer.save_pretrained(model_dir)
        _model.save_pretrained(model_dir)

    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = get_tinybert_model_and_tokenizer(device)


# Pre-compiled constants and regexes for optimization
SLOT_STATUS_DESC = {
    -1: "This Pokemon is DROPPED. It is not part of the battle.",
    0: "This pokemon MAY or MAY NOT be in the back as a switch.",
    1: "This pokemon IS ACTIVE. It is currently on the field.",
    2: "This pokemon is IN THE BACK. It is able to switch in.",
    3: "This pokemon has FAINTED. It no longer participates in the battle.",
    4: "This pokemon CANNOT BE SWITCHED IN. May or may not be in team.",
}
DEFAULT_SLOT_STATUS_DESC = "We do not know about this pokemon."

SLOT_STATUS_IDX = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

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
    cond_str = SLOT_STATUS_DESC.get(cond, DEFAULT_SLOT_STATUS_DESC)

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
        # If no pokemon, we still set the slot status to unknown (index 1)
        role_idx = SLOT_STATUS_IDX.get(cond, 1)
        pokemon_row[55 + role_idx] = 1.0
        return pokemon_str, pokemon_row

    # Types One-Hot (0-53)
    # Type 1 (0-17)
    if pokemon.type_1:
        pokemon_row[pokemon.type_1.value - 1] = 1.0
    # Type 2 (18-35)
    if pokemon.type_2:
        pokemon_row[18 + pokemon.type_2.value - 1] = 1.0
    # Tera Type (36-53)
    if pokemon.is_terastallized:
        pokemon_row[36 + pokemon.tera_type.value - 1] = 1.0

    # Tera Flag (54)
    pokemon_row[54] = 1.0 if pokemon.is_terastallized else 0.0

    # Slot Status One-Hot (55-60)
    role_idx = SLOT_STATUS_IDX.get(cond, 1)
    pokemon_row[55 + role_idx] = 1.0

    # Numerical Stats (61-82)
    # HP (61)
    pokemon_row[61] = pokemon.current_hp_fraction if pokemon.current_hp is not None else 0.0

    # Base Stats (62-67)
    stats = ["hp", "atk", "def", "spa", "spd", "spe"]
    for i, stat in enumerate(stats):
        pokemon_row[62 + i] = pokemon.base_stats[stat] / 200.0

    # Boosts (68-74)
    boosts = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    for i, boost in enumerate(boosts):
        pokemon_row[68 + i] = pokemon.boosts[boost] / 6.0

    # PP (75-78)
    for i, move in enumerate(pokemon.moves):
        if i < 4:
            pokemon_row[75 + i] = pokemon.moves[move].current_pp / pokemon.moves[move].max_pp

    # Misc Stats (79-82)
    pokemon_row[79] = min(pokemon.protect_counter, 4) / 4.0
    pokemon_row[80] = float(pokemon.first_turn)
    pokemon_row[81] = pokemon.weight / 300.0
    pokemon_row[82] = (orig_idx + 1) / 6.0

    # 5. Last Move One-Hot (83-87)
    if last_move_id and pokemon:
        move_ids = list(pokemon.moves.keys())
        if last_move_id in move_ids:
            move_idx = move_ids.index(last_move_id)
            if move_idx < 4:
                pokemon_row[83 + move_idx] = 1.0
            else:
                pokemon_row[87] = 1.0
        else:
            pokemon_row[87] = 1.0
    else:
        pokemon_row[87] = 1.0

    # 6. Status one-hot and counter (88-97)
    statuses = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP]
    for i, s in enumerate(statuses):
        if pokemon.status == s:
            pokemon_row[88 + i] = 1.0
            pokemon_row[93 + i] = min(getattr(pokemon, "status_counter", 0), 5) / 5.0

    # 7. Effects one-hot and counter (98-103)
    curr_effects = pokemon.effects
    effects = [Effect.CONFUSION, Effect.TAUNT, Effect.ENCORE]
    for i, e in enumerate(effects):
        if e in curr_effects:
            pokemon_row[98 + i] = 1.0
            pokemon_row[101 + i] = curr_effects[e] / 5.0

    return pokemon_str, pokemon_row


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool
) -> list[tuple[Pokemon | None, int]]:
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    def get_orig_idx(mon):
        if mon is None or is_opponent:
            return -1
        for i, m in enumerate(battle.team.values()):
            if m == mon:
                return i
        return -1

    if battle.teampreview:
        res = [(m, get_orig_idx(m)) for m in team.values()]
        return (res + [(None, -1)] * 6)[:6]

    # Pack actives first, then the rest of the team to avoid None slots if mon exists
    res = []
    for m in active:
        if m is not None:
            res.append((m, get_orig_idx(m)))

    assigned = {m for m, i in res}
    others_list = [m for m in team.values() if m not in assigned]

    if is_opponent:
        res += [(m, -1) for m in others_list]
    else:
        # My team: prioritize bench (fainted or switchable) over dropped
        possible_switches = {mon for switches in battle.available_switches for mon in switches}
        bench, dropped = [], []
        for mon in others_list:
            idx = get_orig_idx(mon)
            if mon.fainted or mon in possible_switches:
                bench.append((mon, idx))
            else:
                dropped.append((mon, idx))
        res += bench + dropped

    return (res + [(None, -1)] * 6)[:6]


def _get_team_obs(battle: DoubleBattle):
    possible_switches = {mon for switches in battle.available_switches for mon in switches}

    def process_mons(mons, is_opponent: bool):
        txt, arr = [], []
        active_list = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
        # For opponent, count how many unique pokemon have been revealed in battle
        revealed_count = sum(1 for m in battle.opponent_team.values() if m.revealed)

        for i, (mon, idx) in enumerate(mons):
            if mon is None:
                cond = 0
            elif battle.teampreview:
                cond = 2
            elif mon in active_list:
                cond = 1
            elif mon.fainted:
                cond = 3
            elif is_opponent:
                # In VGC, if 4 unique opponents were seen, the other 2 are DROPPED
                if not mon.revealed and revealed_count >= 4:
                    cond = -1
                else:
                    cond = 2
            elif mon in possible_switches:
                cond = 2
            else:
                cond = -1  # Dropped

            t, a = _get_pokemon_obs(mon, battle, cond, idx)
            txt.append(t)
            arr.append(a)
        return txt, arr

    p1_txt, p1_arr = process_mons(
        _get_ordered_pokemon(battle, is_opponent=False), is_opponent=False
    )
    p2_txt, p2_arr = process_mons(_get_ordered_pokemon(battle, is_opponent=True), is_opponent=True)

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
            battle,
            battle.opponent_side_conditions.get(SideCondition.TAILWIND, -1),
            duration=4,
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
    hist_summaries = [_get_turn_summary(battle, battle.turn - i) for i in [1, 2, 3]]

    p1_flat = [text for pair in p1_txt_pairs for text in pair]
    opp_flat = [text for pair in opp_txt_pairs for text in pair]

    texts = [*hist_summaries, field_txt, info_txt, *p1_flat, *opp_flat]

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

    text_emb = encode_texts(texts)  # 3 + 1 + 1 + 12 + 12 = 29 tokens

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
