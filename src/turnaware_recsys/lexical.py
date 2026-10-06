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

