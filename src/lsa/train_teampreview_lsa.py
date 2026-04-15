from __future__ import annotations

import argparse
import json
from pathlib import Path

from teampreview_lsa import LSATeamPreviewModel
from teampreview_supervised import SupervisedTeamPreviewModel

from parse_showdown_logs import (
    documents_and_labels_lsa,
    documents_and_labels_supervised_rich,
    parse_replays_dir,
)


def check_checkpoint(path: str | Path) -> bool:
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
    if state.get("model_type") == SupervisedTeamPreviewModel.MODEL_TYPE:
        for key in ("vectorizer", "svd", "bring_estimators", "lead_clf"):
            if key not in state:
                print("Checkpoint missing key:", key)
                return False
        try:
            model = SupervisedTeamPreviewModel.load(path)
        except Exception as e:
            print("Model.load failed:", e)
            return False
    else:
        for key in ("vectorizer", "svd", "doc_vectors", "bring_labels", "lead_labels"):
            if key not in state:
                print("Checkpoint missing key:", key)
                return False
        n_s = len(state["bring_labels"])
        if n_s == 0:
            print("Checkpoint has no training data")
            return False
        try:
            model = LSATeamPreviewModel.load(path)
        except Exception as e:
            print("Model.load failed:", e)
            return False
    bring, lead = model.decide_stochastic(
        ["a", "b", "c", "d", "e", "f"],
        ["g", "h", "i", "j", "k", "l"],
        team_size=6,
        bring_k=4,
        lead_k=2,
    )
    if len(bring) != 4 or len(lead) != 2:
        print("Bad inference")
        return False
    print("OK:", path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LSA or supervised teampreview models")
    ap.add_argument("--replays_dir", type=str, default=None, help="Directory with replay files")
    ap.add_argument("--out", type=str, default="teampreview_lsa.pt", help="Output path")
    ap.add_argument(
        "--algorithm",
        choices=("lsa", "supervised"),
        default="lsa",
        help="lsa=retrieval LSA; supervised=logistic on TF-IDF+SVD (default: lsa)",
    )
    ap.add_argument(
        "--params_json",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON from tune_teampreview_lsa (best.params); keys depend on --algorithm",
    )
    ap.add_argument("--n_components", type=int, default=50, help="(LSA) SVD dimensions")
    ap.add_argument("--max_preds", type=int, default=10, help="(LSA) retrieval pool")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--min_similarity", type=float, default=0.0, help="(LSA) similarity floor")
    ap.add_argument(
        "--max_features", type=int, default=30_000, help="(supervised) TF-IDF vocab cap"
    )
    ap.add_argument("--C_bring", type=float, default=1.0, help="(supervised) bring logreg C")
    ap.add_argument("--C_lead", type=float, default=1.0, help="(supervised) lead logreg C")
    ap.add_argument(
        "--win_weight",
        type=float,
        default=2.0,
        help="(supervised) sample_weight multiplier for winning games (1.0=no weighting)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max number of samples to load, 0 = all)")
    ap.add_argument(
        "--no-recursive", action="store_true", help="Do not search replays_dir recursively"
    )
    ap.add_argument(
        "--wins-only", action="store_true", help="Only use samples where the recorded side won"
    )
    ap.add_argument(
        "--filter-opponent",
        type=str,
        default=None,
        metavar="TAG",
        help="Only keep samples with this opponent_heuristic tag (from replay_gen logs), e.g. fuzzy, random",
    )
    ap.add_argument(
        "--check", type=str, default=None, metavar="PATH", help="Validate existing .pt checkpoint"
    )
    args = ap.parse_args()

    if args.check:
        raise SystemExit(0 if check_checkpoint(args.check) else 1)

    if not args.replays_dir:
        ap.error("Provide --replays_dir")

    if args.params_json:
        payload = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        pr = payload["best"]["params"]
        if args.algorithm == "lsa":
            args.n_components = int(pr["n_components"])
            args.max_preds = int(pr["max_preds"])
            args.temperature = float(pr["temperature"])
            args.min_similarity = float(pr["min_similarity"])
            print(
                "Hyperparameters from --params_json:",
                f"n_components={args.n_components}",
                f"max_preds={args.max_preds}",
                f"temperature={args.temperature}",
                f"min_similarity={args.min_similarity}",
            )
        else:
            args.n_components = int(pr.get("n_components", args.n_components))
            args.C_bring = float(pr.get("C_bring", args.C_bring))
            args.C_lead = float(pr.get("C_lead", args.C_lead))
            args.win_weight = float(pr.get("win_weight", args.win_weight))
            args.max_features = int(pr.get("max_features", args.max_features))
            if "temperature" in pr:
                args.temperature = float(pr["temperature"])
            print(
                "Hyperparameters from --params_json:",
                f"n_components={args.n_components}",
                f"C_bring={args.C_bring}",
                f"C_lead={args.C_lead}",
                f"win_weight={args.win_weight}",
                f"max_features={args.max_features}",
            )

    samples = []
    if args.replays_dir:
        replays_dir = Path(args.replays_dir)
        if not replays_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {replays_dir}")
        recursive = not args.no_recursive
        f_opp = (args.filter_opponent or "").strip().lower() or None
        for sample in parse_replays_dir(replays_dir, recursive=recursive):
            if f_opp and (sample.get("opponent_heuristic") or "").lower() != f_opp:
                continue
            samples.append(sample)
            if args.limit and len(samples) >= args.limit:
                break

    if not samples:
        raise RuntimeError("No valid team-preview samples found")

    n_wins_before = sum(1 for s in samples if s.get("won", True))
    if args.algorithm == "lsa":
        documents, bring_labels, lead_labels, wins = documents_and_labels_lsa(
            samples, use_wins_only=args.wins_only
        )
        doc_desc = "lsa_flat"
    else:
        documents, bring_labels, lead_labels, wins = documents_and_labels_supervised_rich(
            samples, use_wins_only=args.wins_only
        )
        doc_desc = "supervised_rich"
    n_wins = sum(wins)
    print(
        f"Loaded {len(documents)} training samples ({n_wins} wins); "
        f"algorithm={args.algorithm} documents={doc_desc}"
    )
    if not documents:
        msg = (
            "No training samples after --wins-only / document pipeline. "
            f"Before filtering: {len(samples)} samples, {n_wins_before} with won=True."
        )
        if args.wins_only and n_wins_before == 0:
            msg += (
                " Every sample is a loss (won=False). Replay logs from replay_gen use "
                "expert_side=p1: only games your bot won count. Record more battles, "
                "remove --wins-only, or use --win_weight instead of dropping losses."
            )
        elif args.wins_only:
            msg += " Try dropping --filter-opponent or --wins-only to widen the pool."
        raise RuntimeError(msg)

    if args.algorithm == "lsa":
        model = LSATeamPreviewModel(
            n_components=args.n_components,
            max_preds=args.max_preds,
            temperature=args.temperature,
            min_similarity=args.min_similarity,
        )
        model.fit(documents, bring_labels, lead_labels, team_size=6, win_labels=wins)
    else:
        model = SupervisedTeamPreviewModel(
            n_components=args.n_components,
            max_features=args.max_features,
            C_bring=args.C_bring,
            C_lead=args.C_lead,
            temperature=args.temperature,
        )
        model.fit(
            documents,
            bring_labels,
            lead_labels,
            team_size=6,
            win_labels=wins,
            win_weight=args.win_weight,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
