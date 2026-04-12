from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))

from teampreview_bench import MATCHUPS, load_teampreview_model, run_teampreview_benchmark


async def main(
    n_battles: int,
    lsa_path: str | None,
    *,
    tp_temperature: float,
    tp_deterministic: bool,
    fuzzy_hybrid: bool,
    fuzzy_hybrid_similarity: float,
):
    root_dir = Path(__file__).resolve().parent.parent
    checkpoint = Path(lsa_path) if lsa_path else None
    if checkpoint is not None and not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}")
        return

    if checkpoint is not None:
        m = load_teampreview_model(checkpoint)
        from teampreview_supervised import SupervisedTeamPreviewModel

        kind = "supervised" if isinstance(m, SupervisedTeamPreviewModel) else "LSA"
        print(f"Loaded {kind} teampreview model from {checkpoint.resolve()}\n")

    print(f"Running {len(MATCHUPS)} matchups * {n_battles} battles each...\n")
    if tp_deterministic:
        print("(+TP uses deterministic supervised argmax when checkpoint is supervised)\n")
    if tp_temperature != 1.0:
        print(f"(+TP temperature={tp_temperature})\n")

    if checkpoint is None:
        print(
            "No --lsa: +TP uses the same teampreview as the base bot"
        )

    if fuzzy_hybrid:
        print(
            "(FuzzyH+TP: hybrid — use model when bring similarity > threshold)\n"
        )

    rows = await run_teampreview_benchmark(
        root_dir=root_dir,
        checkpoint=checkpoint,
        n_battles=n_battles,
        tp_temperature=tp_temperature,
        tp_deterministic=tp_deterministic,
        fuzzy_hybrid=fuzzy_hybrid,
        fuzzy_hybrid_similarity=fuzzy_hybrid_similarity,
    )

    for row in rows:
        name = row["name"]
        print(f"  {name}:     {row['base_wr']:.2%}")
        print(f"  {name}+TP:  {row['tp_wr']:.2%}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare each heuristic with and without a trained teampreview model"
    )
    parser.add_argument(
        "-b",
        type=int,
        default=100,
        help="Number of battles per matchup (default: 100)",
    )
    parser.add_argument(
        "--lsa",
        type=str,
        default=None,
        metavar="PATH",
        help="Trained .pt from train_teampreview_lsa.py",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="TeamPreviewHandler / model temperature for +TP (default: 1.0)",
    )

    parser.add_argument(
        "--fuzzy-hybrid",
        action="store_true",
        help="For FuzzyH+TP only: fuse fuzzy expert with model",
    )
    parser.add_argument(
        "--fuzzy-hybrid-similarity",
        type=float,
        default=0.5,
        help="Minimum bring-set similarity to use model /team in fuzzy hybrid (default: 0.5)",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.b,
            args.lsa,
            tp_temperature=args.temperature,
            tp_deterministic=False,
            fuzzy_hybrid=args.fuzzy_hybrid,
            fuzzy_hybrid_similarity=args.fuzzy_hybrid_similarity,
        )
    )
