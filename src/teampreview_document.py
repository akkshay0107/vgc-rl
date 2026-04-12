
from __future__ import annotations

from typing import Any

from poke_env.battle import DoubleBattle

from teampreview_lsa import _normalize_species


def team_species_list(battle: DoubleBattle, our_side: bool = True) -> list[str]:
    team = battle.team if our_side else battle.opponent_team
    out: list[str] = []
    for mon in team.values():
        name = getattr(mon, "species", None)
        if name is not None and hasattr(name, "name"):
            out.append(str(name.name))
        else:
            out.append(getattr(mon, "base_species", str(mon)))
    return out


def _tok(s: str) -> str:
    t = _normalize_species(s)
    return t[:48] if t else "unknown"


def document_species_tagged(our_species: list[str], opp_species: list[str]) -> str:
    """TF-IDF-friendly layout (works when replays have no items/moves)."""
    ou = " ".join(f"our{i}:{_tok(s)}" for i, s in enumerate(our_species))
    op = " ".join(f"opp{i}:{_tok(s)}" for i, s in enumerate(opp_species))
    return f"{ou} || {op}"


def _mon_tokens(mon: Any) -> str:
    sp = _tok(str(getattr(mon.species, "name", mon.species)))
    item = getattr(mon, "item", None)
    item_s = _tok(str(getattr(item, "name", item))) if item else "noitem"
    ab = getattr(mon, "ability", None)
    ab_s = _tok(str(getattr(ab, "name", ab))) if ab else "noability"
    moves = []
    for mv in getattr(mon, "moves", {}).values():
        if mv is not None:
            moves.append(_tok(str(getattr(mv, "id", mv))))
    while len(moves) < 4:
        moves.append("nomove")
    move_s = ",".join(moves[:4])
    return f"{sp} item:{item_s} ab:{ab_s} mv:{move_s}"


def rich_document_from_battle(battle: DoubleBattle) -> str:
    assert isinstance(battle, DoubleBattle)
    our = " ".join(
        f"our{i}:{_mon_tokens(m)}" for i, m in enumerate(battle.team.values())
    )
    opp = " ".join(
        f"opp{i}:{_mon_tokens(m)}" for i, m in enumerate(battle.opponent_team.values())
    )
    return f"{our} || {opp}"


def document_for_sample(
    our_species: list[str],
    opp_species: list[str],
    *,
    our_party: list[dict[str, Any]] | None = None,
    opp_party: list[dict[str, Any]] | None = None,
) -> str:
    if not our_party and not opp_party:
        return document_species_tagged(our_species, opp_species)

    def side(prefix: str, species: list[str], party: list[dict[str, Any]] | None) -> str:
        parts: list[str] = []
        for i in range(6):
            sp = _tok(species[i]) if i < len(species) else "unknown"
            if party and i < len(party):
                d = party[i]
                item_s = _tok(str(d.get("item", "noitem")))
                ab_s = _tok(str(d.get("ability", "noability")))
                mvs = d.get("moves") or []
                ml = [_tok(str(x)) for x in list(mvs)[:4]]
                while len(ml) < 4:
                    ml.append("nomove")
                parts.append(
                    f"{prefix}{i}:{sp} item:{item_s} ab:{ab_s} mv:{','.join(ml)}"
                )
            else:
                parts.append(f"{prefix}{i}:{sp} item:noitem ab:noability mv:nomove,nomove,nomove,nomove")
        return " ".join(parts)

    return (
        f"{side('our', our_species, our_party)} || "
        f"{side('opp', opp_species, opp_party)}"
    )


def parse_party_from_poke_lines(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from parse_showdown_logs import parse_party_from_poke_lines as _parse_party_from_poke_lines

    return _parse_party_from_poke_lines(lines)
