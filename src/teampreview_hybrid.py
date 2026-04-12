from __future__ import annotations


def parse_team_command(cmd: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if "/team" in cmd:
        rest = cmd.split("/team", 1)[1]
    else:
        rest = cmd
    digits = [int(c) for c in rest if c.isdigit()]
    if len(digits) < 4:
        return (), ()
    order4 = digits[:4]
    bring = tuple(sorted(x - 1 for x in order4))
    lead = tuple(x - 1 for x in digits[:2])
    return bring, lead


def bring_set(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    if u == 0:
        return 1.0
    return len(sa & sb) / u


def fuse_teampreview(
    fuzzy_cmd: str,
    model_cmd: str,
    *,
    min_bring_similarity: float = 0.5,
) -> str:
    fb, _ = parse_team_command(fuzzy_cmd)
    mb, _ = parse_team_command(model_cmd)
    if len(fb) < 4 or len(mb) < 4:
        return fuzzy_cmd
    if bring_set(fb, mb) >= min_bring_similarity:
        return model_cmd
    return fuzzy_cmd
