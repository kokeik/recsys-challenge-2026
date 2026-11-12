from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .features import Track


class TfidfTrackRetriever:
    """Small lexical baseline used by the reproducible demo."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.track_ids: list[str] = []
        self.matrix = None

    def fit(self, catalog: Mapping[str, Track]) -> TfidfTrackRetriever:
        self.track_ids = sorted(catalog)
        documents = [catalog[track_id].search_text for track_id in self.track_ids]
        self.matrix = self.vectorizer.fit_transform(documents)
        return self

    def search(
        self, query: str, *, top_k: int = 100, exclude: Sequence[str] = ()
    ) -> tuple[list[str], list[float]]:
        if self.matrix is None:
            raise RuntimeError("fit must be called before search")
        query_vector = self.vectorizer.transform([query])
        similarities = (self.matrix @ query_vector.T).toarray().ravel()
        excluded = set(exclude)
        order = np.argsort(-similarities, kind="stable")
        selected = [index for index in order if self.track_ids[index] not in excluded][:top_k]
        return (
            [self.track_ids[index] for index in selected],
            [float(similarities[index]) for index in selected],
        )

    def rank_candidates(
        self, query: str, candidate_ids: Sequence[str]
    ) -> tuple[list[str], list[float]]:
        """Score only the supplied pool, retaining zero-score candidates."""
        if self.matrix is None:
            raise RuntimeError("fit must be called before rank_candidates")
        index = {track_id: i for i, track_id in enumerate(self.track_ids)}
        indices = [index[track_id] for track_id in candidate_ids]
        scores = (self.matrix[indices] @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        return [candidate_ids[i] for i in order], [float(scores[i]) for i in order]
