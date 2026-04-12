from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator


def _tok_species_token(s: str) -> str:
    t = _normalize_species(s)
    return t[:48] if t else "unknown"


def parse_party_from_poke_lines(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p1: list[dict[str, Any]] = []
    p2: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("|poke|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        player = parts[2].strip()
        details = parts[3].strip()
        segs = [s.strip() for s in details.split(",")]
        species = _tok_species_token(segs[0]) if segs else "unknown"
        item = "noitem"
        ability = "noability"
        if len(segs) >= 4:
            cand = segs[-1].lower().replace(" ", "")
            if cand and cand not in ("m", "f", "n"):
                item = _tok_species_token(segs[-1])
        rec = {"species": species, "item": item, "ability": ability, "moves": []}
        if player == "p1":
            p1.append(rec)
        elif player == "p2":
            p2.append(rec)
    return p1, p2


def _extract_battle_log(html: str) -> str | None:
    m = re.search(
        r'<script[^>]*class="battle-log-data"[^>]*>([\s\S]*?)</script>',
        html,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _normalize_species(details: str) -> str:
    part = details.split(",")[0].strip()
    base = part.split("-")[0].strip().lower()
    return base.replace(" ", "").replace("'", "")


def build_log_from_teampreview_sample(
    p1_species: list[str],
    p2_species: list[str],
    p1_bring: tuple[int, ...],
    p1_lead: tuple[int, ...],
    p2_bring: tuple[int, ...],
    p2_lead: tuple[int, ...],
    *,
    p1_won: bool = True,
    expert_side: str | None = None,
    opponent_heuristic: str | None = None,
) -> str:
    meta: list[str] = []
    if expert_side in ("p1", "p2"):
        meta.append(f"# expert_side={expert_side}")
    if opponent_heuristic:
        meta.append(f"# opponent_heuristic={opponent_heuristic}")
    prefix = ("\n".join(meta) + "\n") if meta else ""
    lines: list[str] = []
    for player, species_list in (("p1", p1_species), ("p2", p2_species)):
        for spec in species_list:
            details = f"{spec}, L50" if spec else "Unknown, L50"
            lines.append(f"|poke|{player}|{details}|")
    lines.append("|teampreview|")
    lines.append("|start|")

    p1_order = list(p1_lead[:2]) + [b for b in p1_bring if b not in p1_lead][:2]
    p2_order = list(p2_lead[:2]) + [b for b in p2_bring if b not in p2_lead][:2]
    for slot, species_list, order in (
        ("p1a", p1_species, p1_order),
        ("p1b", p1_species, p1_order),
        ("p1c", p1_species, p1_order),
        ("p1d", p1_species, p1_order),
        ("p2a", p2_species, p2_order),
        ("p2b", p2_species, p2_order),
        ("p2c", p2_species, p2_order),
        ("p2d", p2_species, p2_order),
    ):
        idx = (ord(slot[-1]) - ord("a")) % 4
        if idx < len(order) and order[idx] < len(species_list):
            spec = species_list[order[idx]]
            lines.append(f"|switch|{slot}|{spec}, L50|")
    lines.append("|win|p1|" if p1_won else "|win|p2|")
    return prefix + "\n".join(lines)


def _parse_battle_log(
    lines: list[str],
    expert_side: str | None = None,
    *,
    opponent_heuristic: str | None = None,
) -> Iterator[dict]:
    p1_pokes: list[str] = []
    p2_pokes: list[str] = []
    poke_line_buffer: list[str] = []
    seen_teampreview = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "|clearpoke":
            p1_pokes.clear()
            p2_pokes.clear()
            poke_line_buffer.clear()
            i += 1
            continue
        if line.startswith("|poke|"):
            parts = line.split("|")
            if len(parts) >= 4:
                player = parts[2].strip()
                details = parts[3].strip()
                species = _normalize_species(details)
                if player == "p1":
                    p1_pokes.append(species)
                elif player == "p2":
                    p2_pokes.append(species)
            poke_line_buffer.append(line)
            i += 1
            continue
        if line.startswith("|teampreview"):
            seen_teampreview = True
            i += 1
            continue
        if (line == "|start|" or line == "|start") and seen_teampreview and len(p1_pokes) == 6 and len(p2_pokes) == 6:
            p1_seen_order: list[str] = []
            p2_seen_order: list[str] = []
            p1_seen_set: set[str] = set()
            p2_seen_set: set[str] = set()
            winner = "p1"
            j = i + 1
            while j < len(lines):
                ln = lines[j]
                if ln.startswith("|win|"):
                    parts = ln.split("|")
                    if len(parts) >= 3:
                        winner = parts[2].strip().lower()
                    break
                if ln.startswith("|switch|"):
                    parts = ln.split("|")
                    if len(parts) >= 4:
                        pos_raw = parts[2].strip()
                        pos = pos_raw.split(":")[0].strip().lower()
                        species = _normalize_species(parts[3].strip())
                        if pos in ("p1a", "p1b", "p1c", "p1d") and species not in p1_seen_set:
                            p1_seen_set.add(species)
                            p1_seen_order.append(species)
                        elif pos in ("p2a", "p2b", "p2c", "p2d") and species not in p2_seen_set:
                            p2_seen_set.add(species)
                            p2_seen_order.append(species)
                j += 1

            def build_bring_lead(
                species_order: list[str],
                team: list[str],
            ) -> tuple[tuple[int, ...], tuple[int, ...]]:
                def species_to_party_index(species: str) -> int:
                    for idx, s in enumerate(team):
                        if s == species or species in s or s in species:
                            return idx
                    return 0
                bring_indices: list[int] = []
                for spec in species_order[:4]:
                    idx = species_to_party_index(spec)
                    if idx not in bring_indices and 0 <= idx < 6:
                        bring_indices.append(idx)
                while len(bring_indices) < 4:
                    for k in range(6):
                        if k not in bring_indices:
                            bring_indices.append(k)
                            break
                    else:
                        break
                bring = tuple(bring_indices[:4])
                lead = tuple(bring[:2])
                return bring, lead

            if len(p1_seen_order) >= 2 and len(p2_seen_order) >= 2:
                p1_bring, p1_lead = build_bring_lead(p1_seen_order, p1_pokes)
                p2_bring, p2_lead = build_bring_lead(p2_seen_order, p2_pokes)
                p1_won = winner in ("p1", "player1")
                p1_party, p2_party = parse_party_from_poke_lines(poke_line_buffer)
                def _row(
                    our_s: list[str],
                    opp_s: list[str],
                    br: tuple[int, ...],
                    ld: tuple[int, ...],
                    won: bool,
                    our_p: list,
                    opp_p: list,
                ) -> dict[str, Any]:
                    return {
                        "our_species": our_s,
                        "opp_species": opp_s,
                        "bring": br,
                        "lead": ld,
                        "won": won,
                        "our_party": our_p,
                        "opp_party": opp_p,
                        "opponent_heuristic": opponent_heuristic,
                    }

                if expert_side is None or expert_side not in ("p1", "p2"):
                    yield _row(
                        list(p1_pokes),
                        list(p2_pokes),
                        p1_bring,
                        p1_lead,
                        p1_won,
                        p1_party,
                        p2_party,
                    )
                    yield _row(
                        list(p2_pokes),
                        list(p1_pokes),
                        p2_bring,
                        p2_lead,
                        not p1_won,
                        p2_party,
                        p1_party,
                    )
                elif expert_side == "p1":
                    yield _row(
                        list(p1_pokes),
                        list(p2_pokes),
                        p1_bring,
                        p1_lead,
                        p1_won,
                        p1_party,
                        p2_party,
                    )
                else:
                    assert expert_side == "p2"
                    yield _row(
                        list(p2_pokes),
                        list(p1_pokes),
                        p2_bring,
                        p2_lead,
                        not p1_won,
                        p2_party,
                        p1_party,
                    )
            break
        i += 1


def parse_log_lines(lines: list[str]) -> Iterator[dict]:
    stripped = [line.strip() for line in lines if line.strip()]
    stripped, expert_side, opponent_heuristic = _strip_meta_header_lines(stripped)
    yield from _parse_battle_log(
        stripped,
        expert_side=expert_side,
        opponent_heuristic=opponent_heuristic,
    )


def _strip_meta_header_lines(lines: list[str]) -> tuple[list[str], str | None, str | None]:
    expert_side: str | None = None
    opponent_heuristic: str | None = None
    rest = lines
    while rest and rest[0].startswith("#"):
        line = rest[0]
        if line.startswith("# expert_side="):
            v = line.split("=", 1)[1].strip().lower()
            if v in ("p1", "p2"):
                expert_side = v
        elif line.startswith("# opponent_heuristic="):
            opponent_heuristic = line.split("=", 1)[1].strip().lower()
        rest = rest[1:]
    return rest, expert_side, opponent_heuristic


def parse_html_replay(path: str | Path) -> Iterator[dict]:

    path = Path(path)
    if path.suffix.lower() not in (".html", ".htm"):
        return
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return
    log_text = _extract_battle_log(html)
    if not log_text:
        return
    lines = [line.strip() for line in log_text.split("\n") if line.strip()]
    lines, expert_side, opp_h = _strip_meta_header_lines(lines)
    yield from _parse_battle_log(lines, expert_side=expert_side, opponent_heuristic=opp_h)


def parse_raw_log_replay(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.suffix.lower() not in (".log", ".txt"):
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines, expert_side, opponent_heuristic = _strip_meta_header_lines(lines)
    yield from _parse_battle_log(
        lines,
        expert_side=expert_side,
        opponent_heuristic=opponent_heuristic,
    )


def parse_replays_dir(replays_dir: str | Path, recursive: bool = True) -> Iterator[dict]:
    replays_dir = Path(replays_dir)
    if not replays_dir.is_dir():
        return
    if recursive:
        html_paths = sorted(replays_dir.rglob("*.html")) + sorted(replays_dir.rglob("*.htm"))
        raw_paths = sorted(replays_dir.rglob("*.log")) + sorted(replays_dir.rglob("*.txt"))
    else:
        html_paths = sorted(replays_dir.glob("*.html")) + sorted(replays_dir.glob("*.htm"))
        raw_paths = sorted(replays_dir.glob("*.log")) + sorted(replays_dir.glob("*.txt"))
    for path in html_paths:
        yield from parse_html_replay(path)
    for path in raw_paths:
        yield from parse_raw_log_replay(path)


def _filter_wins(samples: list[dict], use_wins_only: bool) -> list[dict]:
    if use_wins_only:
        return [s for s in samples if s.get("won", True)]
    return samples


def documents_and_labels_lsa(
    samples: list[dict],
    use_wins_only: bool = False,
) -> tuple[list[str], list[tuple[int, ...]], list[tuple[int, ...]], list[bool]]:
    from teampreview_lsa import matchup_string

    samples = _filter_wins(samples, use_wins_only)
    documents: list[str] = []
    bring_labels: list[tuple[int, ...]] = []
    lead_labels: list[tuple[int, ...]] = []
    wins: list[bool] = []
    for s in samples:
        doc = matchup_string(s["our_species"], s["opp_species"])
        documents.append(doc)
        bring_labels.append(s["bring"])
        lead_labels.append(s["lead"])
        wins.append(s.get("won", True))
    return documents, bring_labels, lead_labels, wins


def documents_and_labels_supervised_rich(
    samples: list[dict],
    use_wins_only: bool = False,
) -> tuple[list[str], list[tuple[int, ...]], list[tuple[int, ...]], list[bool]]:
    from teampreview_document import document_for_sample

    samples = _filter_wins(samples, use_wins_only)
    documents: list[str] = []
    bring_labels: list[tuple[int, ...]] = []
    lead_labels: list[tuple[int, ...]] = []
    wins: list[bool] = []
    for s in samples:
        doc = document_for_sample(
            s["our_species"],
            s["opp_species"],
            our_party=s.get("our_party"),
            opp_party=s.get("opp_party"),
        )
        documents.append(doc)
        bring_labels.append(s["bring"])
        lead_labels.append(s["lead"])
        wins.append(s.get("won", True))
    return documents, bring_labels, lead_labels, wins


if __name__ == "__main__":
    import sys
    replays_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("replays")
    samples = list(parse_replays_dir(replays_dir))
    n_wins = sum(1 for s in samples if s.get("won", True))
    print(f"Parsed {len(samples)} teampreview samples from {replays_dir} ({n_wins} wins)")
    for i, s in enumerate(samples):
        w = "W" if s.get("won", True) else "L"
        print(f"  {i+1}. [{w}] our={s['our_species']}, bring={s['bring']}, lead={s['lead']}")
    if not samples:
        print("No samples found.")
