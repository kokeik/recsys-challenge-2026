from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class DenseTrackRetriever:
    """Cosine ranking of caller-supplied embeddings; no API calls or bundled vectors."""

    def __init__(self, track_vectors: Mapping[str, Sequence[float]]) -> None:
        if not track_vectors:
            raise ValueError("Track embeddings must not be empty")
        self.track_ids = sorted(track_vectors)
        self.matrix = np.asarray([track_vectors[t] for t in self.track_ids], dtype=np.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[1] == 0:
            raise ValueError("Embeddings must be nonempty vectors of equal length")
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        if not np.isfinite(self.matrix).all() or np.any(norms == 0):
            raise ValueError("Embeddings must be finite and nonzero")
        self.matrix /= norms
        self.index = {tid: i for i, tid in enumerate(self.track_ids)}

    def rank_candidates(
        self, query_vector: Sequence[float], candidate_ids: Sequence[str]
    ) -> tuple[list[str], list[float]]:
        query = np.asarray(query_vector, dtype=np.float64)
        if query.shape != (self.matrix.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Query embedding has invalid dimension or values")
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("Query embedding must be nonzero")
        scores = self.matrix[[self.index[tid] for tid in candidate_ids]] @ (query / norm)
        order = np.argsort(-scores, kind="stable")
        return [candidate_ids[i] for i in order], [float(scores[i]) for i in order]

    def search(
        self, query_vector: Sequence[float], *, top_k: int = 100, exclude: Sequence[str] = ()
    ) -> tuple[list[str], list[float]]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        excluded = set(exclude)
        ids, scores = self.rank_candidates(
            query_vector, [tid for tid in self.track_ids if tid not in excluded]
        )
        return ids[:top_k], scores[:top_k]
