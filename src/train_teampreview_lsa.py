from __future__ import annotations

import argparse
from pathlib import Path

from parse_showdown_logs import documents_and_labels, parse_replays_dir
from teampreview_lsa import LSATeamPreviewModel


def check_checkpoint(path):
    path = Path(path)
    if not path.exists():
        print("File not found:", path)
        return False
    if path.stat().st_size == 0:
        print("File is empty:", path)
        return False
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("Failed to load:", e)
        return False
    for key in ("vectorizer", "svd", "doc_vectors", "bring_labels", "lead_labels"):
        if key not in state:
            print("Checkpoint missing key:", key)
            return False
    n = len(state["bring_labels"])
    if n == 0:
        print("Checkpoint has no training data")
        return False
    try:
        model = LSATeamPreviewModel.load(path)
    except Exception as e:
        print("Model.load failed:", e)
        return False
    bring, lead = model.decide_stochastic(
        ["a", "b", "c", "d", "e", "f"], ["g", "h", "i", "j", "k", "l"],
        team_size=6, bring_k=4, lead_k=2)
    if len(bring) != 4 or len(lead) != 2:
        print("Bad inference")
        return False
    print("OK:", path, n, "samples")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays_dir", type=str, default=None, help="Directory with replay files")
    ap.add_argument("--out", type=str, default="teampreview_lsa.pt", help="Output path for model")
    ap.add_argument("--n_components", type=int, default=50, help="dimensions")
    ap.add_argument("--max_preds", type=int, default=10, help="Max documents for inference")
    ap.add_argument("--temperature", type=float, default=1.0, help="temperature")
    ap.add_argument("--min_similarity", type=float, default=0.0, help="min cosine similarity")
    ap.add_argument("--limit", type=int, default=0, help="Max number of samples to load, 0 = all)")
    ap.add_argument("--check", type=str, default=None, metavar="PATH", help="Validate existing .pt checkpoint")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(0 if check_checkpoint(args.check) else 1)

    if not args.replays_dir:
        ap.error("Provide --replays_dir")

    samples = []
    if args.replays_dir:
        replays_dir = Path(args.replays_dir)
        if not replays_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {replays_dir}")
        for i, sample in enumerate(parse_replays_dir(replays_dir)):
            samples.append(sample)
            if args.limit and len(samples) >= args.limit:
                break

    if not samples:
        raise RuntimeError("No valid team-preview samples found")

    documents, bring_labels, lead_labels = documents_and_labels(samples)
    print(f"Loaded {len(documents)} training samples")

    model = LSATeamPreviewModel(
        n_components=args.n_components,
        max_preds=args.max_preds,
        temperature=args.temperature,
        min_similarity=args.min_similarity,
    )
    model.fit(documents, bring_labels, lead_labels, team_size=6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved LSA model to {out_path}")


if __name__ == "__main__":
    main()
