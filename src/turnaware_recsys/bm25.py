from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from math import log

import numpy as np

from .features import Track


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25TrackRetriever:
    """Okapi BM25 with corpus statistics fit on the entire catalog."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.documents: dict[str, Counter[str]] = {}
        self.idf: dict[str, float] = {}
        self.average_length = 1.0

    def fit(self, catalog: Mapping[str, Track]) -> BM25TrackRetriever:
        if not catalog:
            raise ValueError("Catalog must not be empty")
        self.documents = {
            tid: Counter(tokenize(catalog[tid].search_text)) for tid in sorted(catalog)
        }
        frequencies: Counter[str] = Counter()
        for document in self.documents.values():
            frequencies.update(document.keys())
        n = len(self.documents)
        self.idf = {term: log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequencies.items()}
        self.average_length = max(sum(sum(d.values()) for d in self.documents.values()) / n, 1)
        return self

    def rank_candidates(
        self, query: str, candidate_ids: Sequence[str]
    ) -> tuple[list[str], list[float]]:
        if not self.documents:
            raise RuntimeError("fit must be called before ranking")
        scores = []
        for tid in candidate_ids:
            document = self.documents[tid]
            norm = self.k1 * (1 - self.b + self.b * sum(document.values()) / self.average_length)
            score = sum(
                self.idf.get(term, 0) * document[term] * (self.k1 + 1) / (document[term] + norm)
                for term in tokenize(query)
            )
            scores.append(score)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [candidate_ids[i] for i in order], [float(scores[i]) for i in order]

    def search(
        self, query: str, *, top_k: int = 100, exclude: Sequence[str] = ()
    ) -> tuple[list[str], list[float]]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        excluded = set(exclude)
        ids, scores = self.rank_candidates(query, [t for t in self.documents if t not in excluded])
        return ids[:top_k], scores[:top_k]
