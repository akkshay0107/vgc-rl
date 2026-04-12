"""
Reference: Carli, B.L. (2025) Predicting competitive Pokémon VGC leads using
Latent Semantic Analysis. Journal of Geek Studies 12(2): 75-83.
https://doi.org/10.5281/zenodo.15808330

Future work
- Extension: add moves/item metadata to document string in fit()
- Extension: supervised learning for finetune
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _normalize_species(name: str) -> str:
    return name.split("-")[0].split(",")[0].strip().lower().replace(" ", "").replace("'", "")


def matchup_string(our_species: list[str], opp_species: list[str]) -> str:
    our = " ".join(_normalize_species(s) for s in our_species)
    opp = " ".join(_normalize_species(s) for s in opp_species)
    return f"{our} VS {opp}"


class LSATeamPreviewModel:

    def __init__(
        self,
        n_components: int = 50,
        max_preds: int = 10,
        temperature: float = 1.0,
        min_similarity: float = 0.0,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.max_preds = max_preds
        self.temperature = temperature
        self.min_similarity = min_similarity
        self.rng = np.random.default_rng(random_state)

        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.doc_vectors: np.ndarray | None = None 
        self.bring_labels: list[tuple[int, ...]] = []  
        self.lead_labels: list[tuple[int, ...]] = []   # subset of bring currently
        self.win_labels: list[bool] = []   # whether our side won for weighting at inference 
        self.team_size: int = 6

    def fit(
        self,
        documents: list[str],
        bring_labels: list[tuple[int, ...]],
        lead_labels: list[tuple[int, ...]],
        team_size: int = 6,
        win_labels: list[bool] | None = None,
    ) -> LSATeamPreviewModel:
        if not documents or len(documents) != len(bring_labels) or len(documents) != len(lead_labels):
            raise ValueError("documents, bring_labels, and lead_labels must have same length")
        self.team_size = team_size
        if win_labels is not None and len(win_labels) != len(documents):
            raise ValueError("win_labels must have same length as documents")
        self.win_labels = list(win_labels) if win_labels is not None else []

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=10_000,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(documents)
        self.svd = TruncatedSVD(n_components=min(self.n_components, X.shape[1], X.shape[0] - 1))
        latent = self.svd.fit_transform(X)
        self.doc_vectors = normalize(latent, norm="l2", axis=1)
        self.bring_labels = list(bring_labels)
        self.lead_labels = list(lead_labels)
        return self

    def query_vector(self, matchup: str) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("Model not fitted")
        X = self.vectorizer.transform([matchup])
        latent = self.svd.transform(X)
        return normalize(latent, norm="l2", axis=1)[0]

    def similarities(self, query_vec: np.ndarray) -> np.ndarray:
        if self.doc_vectors is None:
            raise RuntimeError("Model not fitted")
        return self.doc_vectors @ query_vec

    def sample_bring_lead(
        self,
        matchup: str,
        team_size: int,
        bring_k: int,
        lead_k: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        
        query_vec = self.query_vector(matchup)
        sims = self.similarities(query_vec)
        sims = np.clip(sims, self.min_similarity, None)

        top_n = min(len(sims), max(20, self.max_preds * 2))
        top_idx = np.argsort(sims)[::-1][:top_n]
        weights = sims[top_idx].astype(np.float64)
        if self.win_labels:
            win_bonus = 1.5
            for k, i in enumerate(top_idx):
                if i < len(self.win_labels) and self.win_labels[i]:
                    weights[k] *= win_bonus
        if np.sum(weights) <= 0:

            indices = self.rng.choice(self.team_size, size=min(bring_k, self.team_size), replace=False)
            bring = tuple(int(x) for x in indices[:bring_k])
            lead = tuple(int(x) for x in bring[:lead_k])
            return bring, lead

        logits = np.log(weights + 1e-12) / max(1e-6, self.temperature)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        chosen = self.rng.choice(len(top_idx), p=probs)
        i = top_idx[chosen]
        bring = tuple(self.bring_labels[i][:bring_k])
        lead = tuple(self.lead_labels[i][:lead_k])

        bring = tuple(b for b in bring if 0 <= b < team_size)[:bring_k]
        lead = tuple(l for l in lead if l in bring and 0 <= l < team_size)[:lead_k]
        if len(lead) < lead_k:
            for b in bring:
                if b not in lead:
                    lead = lead + (b,)
                if len(lead) >= lead_k:
                    break
            lead = lead[:lead_k]
        return bring, lead

    def decide_stochastic(
        self,
        our_species: list[str],
        opp_species: list[str],
        team_size: int = 6,
        bring_k: int = 4,
        lead_k: int = 2,
        *,
        document: str | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not self.bring_labels or not self.lead_labels:
            indices = list(range(min(team_size, 6)))
            self.rng.shuffle(indices)
            bring = tuple(indices[:bring_k])
            lead = tuple(indices[:lead_k])
            return bring, lead
        doc = document if document is not None else matchup_string(our_species, opp_species)
        return self.sample_bring_lead(doc, team_size, bring_k, lead_k)

    def decide_from_battle(
        self,
        battle: Any,
        bring_k: int = 4,
        lead_k: int = 2,
        *,
        deterministic: bool = False,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        from poke_env.battle import DoubleBattle

        from teampreview_document import team_species_list

        _ = deterministic  # retrieval policy is always stochastic; flag kept for API parity
        assert isinstance(battle, DoubleBattle)
        our = team_species_list(battle, True)
        opp = team_species_list(battle, False)
        ts = len(battle.team)
        doc = matchup_string(our, opp)
        return self.sample_bring_lead(doc, ts, bring_k, lead_k)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "n_components": self.n_components,
            "max_preds": self.max_preds,
            "temperature": self.temperature,
            "min_similarity": self.min_similarity,
            "team_size": self.team_size,
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "doc_vectors": self.doc_vectors,
            "bring_labels": self.bring_labels,
            "lead_labels": self.lead_labels,
            "win_labels": getattr(self, "win_labels", []),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> LSATeamPreviewModel:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            n_components=state.get("n_components", 50),
            max_preds=state.get("max_preds", 10),
            temperature=state.get("temperature", 1.0),
            min_similarity=state.get("min_similarity", 0.0),
        )
        model.team_size = state.get("team_size", 6)
        model.vectorizer = state["vectorizer"]
        model.svd = state["svd"]
        model.doc_vectors = state["doc_vectors"]
        model.bring_labels = state["bring_labels"]
        model.lead_labels = state["lead_labels"]
        model.win_labels = state.get("win_labels", [])
        for k, v in kwargs.items():
            if hasattr(model, k):
                setattr(model, k, v)
        return model
