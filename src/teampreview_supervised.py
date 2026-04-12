from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from teampreview_document import document_for_sample, rich_document_from_battle

# (position in sorted-bring list) ordered lead slots — 12 permutations P(4,2)
_ORDERED_LEAD_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(4) for j in range(4) if i != j
]


def _encode_ordered_lead(bring: tuple[int, ...], lead: tuple[int, ...]) -> int:
    bs = sorted(bring)
    ia = bs.index(lead[0])
    ib = bs.index(lead[1])
    return _ORDERED_LEAD_PAIRS.index((ia, ib))


def _decode_ordered_lead(bring: tuple[int, ...], cls: int) -> tuple[int, int]:
    ia, ib = _ORDERED_LEAD_PAIRS[cls]
    bs = sorted(bring)
    return bs[ia], bs[ib]


class SupervisedTeamPreviewModel:
    MODEL_TYPE = "supervised_v1"

    def __init__(
        self,
        n_components: int = 96,
        max_features: int = 30_000,
        C_bring: float = 1.0,
        C_lead: float = 1.0,
        temperature: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.C_bring = C_bring
        self.C_lead = C_lead
        self.temperature = temperature
        self._rs = random_state
        self.rng = np.random.default_rng(random_state)
        self.team_size: int = 6

        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self._bring_estimators: list[LogisticRegression | DummyClassifier] = []
        self._lead_clf: LogisticRegression | DummyClassifier | None = None

    def fit(
        self,
        documents: list[str],
        bring_labels: list[tuple[int, ...]],
        lead_labels: list[tuple[int, ...]],
        team_size: int = 6,
        win_labels: list[bool] | None = None,
        *,
        win_weight: float = 2.0,
    ) -> SupervisedTeamPreviewModel:
        if len(documents) != len(bring_labels) or len(documents) != len(lead_labels):
            raise ValueError("documents, bring_labels, and lead_labels must have same length")
        if not documents:
            raise ValueError("need at least one training document (empty list after filtering)")
        self.team_size = team_size
        n = len(documents)
        y_bring = np.zeros((n, team_size), dtype=np.int32)
        for i, b in enumerate(bring_labels):
            for j in b:
                if 0 <= j < team_size:
                    y_bring[i, j] = 1
        y_lead = np.array(
            [_encode_ordered_lead(b, lead_labels[i]) for i, b in enumerate(bring_labels)],
            dtype=np.int64,
        )

        sw: np.ndarray | None = None
        if win_labels is not None and len(win_labels) == n:
            sw = np.array([1.0 + (win_weight - 1.0) * float(w) for w in win_labels], dtype=np.float64)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=self.max_features,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(documents)
        n_comp = min(self.n_components, max(1, X.shape[1] - 1), max(1, X.shape[0] - 1))
        rs = 0 if self._rs is None else int(self._rs)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=rs)
        Z = self.svd.fit_transform(X)
        Z = normalize(Z, norm="l2", axis=1)

        self._bring_estimators = []
        for j in range(team_size):
            yj = y_bring[:, j]
            if np.unique(yj).size < 2:
                dc = DummyClassifier(strategy="constant", constant=int(yj[0]))
                dc.fit(Z, yj)
                self._bring_estimators.append(dc)
            else:
                lr = LogisticRegression(
                    C=self.C_bring,
                    max_iter=400,
                    random_state=self._rs,
                    solver="lbfgs",
                )
                lr.fit(Z, yj, sample_weight=sw)
                self._bring_estimators.append(lr)

        Zl = np.hstack([Z, y_bring.astype(np.float64)])
        if np.unique(y_lead).size < 2:
            self._lead_clf = DummyClassifier(strategy="constant", constant=int(y_lead[0]))
            self._lead_clf.fit(Zl, y_lead)
        else:
            self._lead_clf = LogisticRegression(
                C=self.C_lead,
                max_iter=400,
                random_state=self._rs,
                solver="lbfgs",
            )
            self._lead_clf.fit(Zl, y_lead, sample_weight=sw)
        return self

    def _transform(self, documents: list[str]) -> np.ndarray:
        assert self.vectorizer is not None and self.svd is not None
        X = self.vectorizer.transform(documents)
        Z = self.svd.transform(X)
        return normalize(Z, norm="l2", axis=1)

    def _bring_probs(self, Z: np.ndarray) -> np.ndarray:
        cols: list[np.ndarray] = []
        for est in self._bring_estimators:
            pr = est.predict_proba(Z)
            if pr.shape[1] == 1:
                cols.append(np.zeros(Z.shape[0], dtype=np.float64))
            else:
                cols.append(pr[:, 1].astype(np.float64))
        return np.column_stack(cols)

    def predict_bring_lead_deterministic(
        self, document: str, bring_k: int = 4, lead_k: int = 2
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        Z = self._transform([document])
        p = self._bring_probs(Z)[0]
        n_take = min(4, self.team_size)
        top = np.argsort(-p)[:n_take]
        bring = tuple(sorted(int(x) for x in top))
        bh = np.zeros(self.team_size, dtype=np.float64)
        for i in bring:
            bh[i] = 1.0
        Zl = np.hstack([Z, bh.reshape(1, -1)])
        assert self._lead_clf is not None
        cls = int(self._lead_clf.predict(Zl)[0])
        lead_full = _decode_ordered_lead(bring, cls)
        lead = lead_full[:lead_k]
        return bring, lead

    def sample_bring_lead(
        self,
        document: str,
        team_size: int,
        bring_k: int,
        lead_k: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        Z = self._transform([document])
        p_b = np.clip(self._bring_probs(Z)[0], 1e-6, 1.0)
        p_b = p_b / p_b.sum()
        try:
            picked = self.rng.choice(team_size, size=bring_k, replace=False, p=p_b)
        except ValueError:
            picked = np.arange(team_size)[:bring_k]
        bring = tuple(sorted(int(x) for x in picked))[:bring_k]

        bh = np.zeros(self.team_size, dtype=np.float64)
        for i in bring:
            bh[i] = 1.0
        Zl = np.hstack([Z, bh.reshape(1, -1)])
        assert self._lead_clf is not None
        logp = self._lead_clf.predict_log_proba(Zl)[0]
        if logp.size <= 1:
            cls = int(self._lead_clf.predict(Zl)[0])
        else:
            t = max(self.temperature, 1e-6)
            logits = logp / t
            ex = np.exp(logits - logits.max())
            pr = ex / ex.sum()
            cls = int(self.rng.choice(len(pr), p=pr))
        lead_full = _decode_ordered_lead(bring, cls)
        lead = lead_full[:lead_k]
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
        doc = document if document is not None else document_for_sample(
            our_species, opp_species, our_party=None, opp_party=None
        )
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

        assert isinstance(battle, DoubleBattle)
        ts = len(battle.team)
        doc = rich_document_from_battle(battle)
        if deterministic:
            return self.predict_bring_lead_deterministic(doc, bring_k=bring_k, lead_k=lead_k)
        return self.sample_bring_lead(doc, ts, bring_k, lead_k)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": self.MODEL_TYPE,
            "n_components": self.n_components,
            "max_features": self.max_features,
            "C_bring": self.C_bring,
            "C_lead": self.C_lead,
            "temperature": self.temperature,
            "team_size": self.team_size,
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "bring_estimators": self._bring_estimators,
            "lead_clf": self._lead_clf,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> SupervisedTeamPreviewModel:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("model_type") != cls.MODEL_TYPE:
            raise ValueError(f"Expected model_type {cls.MODEL_TYPE}, got {state.get('model_type')}")
        m = cls(
            n_components=state.get("n_components", 96),
            max_features=state.get("max_features", 30_000),
            C_bring=state.get("C_bring", 1.0),
            C_lead=state.get("C_lead", 1.0),
            temperature=state.get("temperature", 1.0),
        )
        m.team_size = state.get("team_size", 6)
        m.vectorizer = state["vectorizer"]
        m.svd = state["svd"]
        m._bring_estimators = state["bring_estimators"]
        m._lead_clf = state["lead_clf"]
        for k, v in kwargs.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m


def is_supervised_checkpoint(path: str | Path) -> bool:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        return state.get("model_type") == SupervisedTeamPreviewModel.MODEL_TYPE
    except OSError:
        return False
