from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .candidate_pool import Candidate
from .features import FeatureExtractor, QueryContext


@dataclass(frozen=True)
class TrainingQuery:
    context: QueryContext
    candidates: Sequence[Candidate]
    target_track_id: str


class TurnAwareReranker:
    """Group-aware reranker with a LightGBM LambdaRank backend when available."""

    def __init__(self, extractor: FeatureExtractor, random_state: int = 42) -> None:
        self.extractor = extractor
        self.random_state = random_state
        self.model = None
        self.backend = "unfitted"
        self.feature_names_: list[str] = []

    def fit(self, queries: Sequence[TrainingQuery]) -> TurnAwareReranker:
        matrices: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        groups: list[int] = []

        for query in queries:
            if query.context.turn_number < 2:
                continue
            if query.target_track_id not in {item.track_id for item in query.candidates}:
                continue
            matrix, names = self.extractor.transform(query.candidates, query.context)
            matrices.append(matrix)
            labels.append(
                np.asarray(
                    [item.track_id == query.target_track_id for item in query.candidates],
                    dtype=np.int8,
                )
            )
            groups.append(len(query.candidates))
            self.feature_names_ = names

        if not matrices:
            raise ValueError("No trainable turn>=2 query has its target in the candidate pool")
        x = np.vstack(matrices)
        y = np.concatenate(labels)

        try:
            from lightgbm import LGBMRanker

            self.model = LGBMRanker(
                objective="lambdarank",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                random_state=self.random_state,
                verbosity=-1,
            )
            self.model.fit(x, y, group=groups)
            self.backend = "lightgbm-lambdarank"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            positive_weight = max(float((y == 0).sum() / max((y == 1).sum(), 1)), 1.0)
            weights = np.where(y == 1, positive_weight, 1.0)
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=self.random_state,
            )
            self.model.fit(x, y, sample_weight=weights)
            self.backend = "sklearn-classifier-fallback"
        return self

    def rank(self, candidates: Sequence[Candidate], context: QueryContext) -> list[str]:
        if self.model is None:
            raise RuntimeError("fit must be called before rank")
        matrix, names = self.extractor.transform(candidates, context)
        if names != self.feature_names_:
            raise RuntimeError("Feature schema changed between training and inference")
        if self.backend == "lightgbm-lambdarank":
            scores = self.model.predict(matrix)
        else:
            scores = self.model.predict_proba(matrix)[:, 1]
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [candidates[index].track_id for index in order]

