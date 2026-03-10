from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator


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


def _parse_battle_log(lines: list[str]) -> Iterator[dict]:
    p1_pokes: list[str] = []
    p2_pokes: list[str] = []
    seen_teampreview = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "|clearpoke":
            p1_pokes.clear()
            p2_pokes.clear()
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
            i += 1
            continue
        if line.startswith("|teampreview"):
            seen_teampreview = True
            i += 1
            continue
        # |start or |start| (log may omit trailing pipe)
        if (line == "|start|" or line == "|start") and seen_teampreview and len(p1_pokes) == 6 and len(p2_pokes) == 6:
            # In doubles we only have p1a, p1b (and p2a, p2b). The 4 brought are the 4 unique
            # species that appear in any |switch| for that side, in order of first appearance.
            p1_seen_order: list[str] = []
            p2_seen_order: list[str] = []
            p1_seen_set: set[str] = set()
            p2_seen_set: set[str] = set()
            j = i + 1
            while j < len(lines):
                ln = lines[j]
                if ln.startswith("|win|"):
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

            # Need at least 2 unique species per side (the leads); we fill to 4 from team if needed
            if len(p1_seen_order) >= 2 and len(p2_seen_order) >= 2:
                p1_bring, p1_lead = build_bring_lead(p1_seen_order, p1_pokes)
                p2_bring, p2_lead = build_bring_lead(p2_seen_order, p2_pokes)
                yield {
                    "our_species": list(p1_pokes),
                    "opp_species": list(p2_pokes),
                    "bring": p1_bring,
                    "lead": p1_lead,
                }
                yield {
                    "our_species": list(p2_pokes),
                    "opp_species": list(p1_pokes),
                    "bring": p2_bring,
                    "lead": p2_lead,
                }
            break
        i += 1


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
    yield from _parse_battle_log(lines)


def parse_replays_dir(replays_dir: str | Path) -> Iterator[dict]:

    replays_dir = Path(replays_dir)
    for path in sorted(replays_dir.glob("*.html")) + sorted(replays_dir.glob("*.htm")):
        yield from parse_html_replay(path)


def documents_and_labels(
    samples: list[dict],
) -> tuple[list[str], list[tuple[int, ...]], list[tuple[int, ...]]]:
    from teampreview_lsa import matchup_string

    documents: list[str] = []
    bring_labels: list[tuple[int, ...]] = []
    lead_labels: list[tuple[int, ...]] = []
    for s in samples:
        documents.append(matchup_string(s["our_species"], s["opp_species"]))
        bring_labels.append(s["bring"])
        lead_labels.append(s["lead"])
    return documents, bring_labels, lead_labels


if __name__ == "__main__":
    import sys
    replays_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("replays")
    samples = list(parse_replays_dir(replays_dir))
    print(f"Parsed {len(samples)} teampreview samples from {replays_dir}")
    for i, s in enumerate(samples):
        print(f"  {i+1}. our={s['our_species']}, bring={s['bring']}, lead={s['lead']}")
    if not samples:
        print("No samples found.")
