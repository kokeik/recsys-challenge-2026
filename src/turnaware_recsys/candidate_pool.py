from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log1p


@dataclass
class Candidate:
    """One track in the union of multiple ranked retrieval sources."""

    track_id: str
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)

    def rank_features(
        self, source_names: Sequence[str], *, include_scores: bool = True
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        for source in source_names:
            rank = self.source_ranks.get(source)
            features[f"{source}_present"] = float(rank is not None)
            features[f"{source}_inv_rank"] = 0.0 if rank is None else 1.0 / rank
            features[f"{source}_log_rank"] = 0.0 if rank is None else 1.0 / log1p(rank)
            if include_scores:
                features[f"{source}_score"] = self.source_scores.get(source, 0.0)
        return features


class CandidatePoolBuilder:
    """Build a deterministic union while retaining per-source ranks and scores."""

    def __init__(self, source_limits: Mapping[str, int] | None = None) -> None:
        self.source_limits = dict(source_limits or {})
        if any(limit < 0 for limit in self.source_limits.values()):
            raise ValueError("Source limits must not be negative")

    def build(
        self,
        rankings: Mapping[str, Sequence[str]],
        scores: Mapping[str, Sequence[float]] | None = None,
        *,
        exclude: Iterable[str] = (),
    ) -> list[Candidate]:
        excluded = set(exclude)
        candidates: dict[str, Candidate] = {}
        scores = scores or {}

        for source, track_ids in rankings.items():
            limit = self.source_limits.get(source, len(track_ids))
            source_scores = scores.get(source, ())
            for rank, track_id in enumerate(track_ids[:limit], start=1):
                if track_id in excluded:
                    continue
                candidate = candidates.setdefault(track_id, Candidate(track_id=track_id))
                if source in candidate.source_ranks:
                    continue
                candidate.source_ranks[source] = rank
                if rank <= len(source_scores):
                    candidate.source_scores[source] = float(source_scores[rank - 1])

        # A stable initial order helps reproducibility before learned reranking.
        return sorted(
            candidates.values(),
            key=lambda item: (
                min(item.source_ranks.values(), default=10**9),
                -len(item.source_ranks),
                item.track_id,
            ),
        )


def attach_fixed_pool_rankings(
    candidates: Sequence[Candidate],
    rankings: Mapping[str, tuple[Sequence[str], Sequence[float]]],
) -> list[Candidate]:
    """Attach independent rankings without adding, dropping, or reordering candidates."""
    result = [
        Candidate(c.track_id, dict(c.source_ranks), dict(c.source_scores)) for c in candidates
    ]
    by_id = {candidate.track_id: candidate for candidate in result}
    if len(by_id) != len(result):
        raise ValueError("Candidate pool contains duplicate IDs")
    for source, (track_ids, scores) in rankings.items():
        if len(track_ids) != len(by_id) or set(track_ids) != set(by_id):
            raise ValueError(f"{source} must rank exactly the fixed candidate pool")
        if len(scores) != len(track_ids):
            raise ValueError(f"{source} scores and IDs must have the same length")
        if any(source in candidate.source_ranks for candidate in result):
            raise ValueError(f"Source already exists: {source}")
        for rank, (track_id, score) in enumerate(zip(track_ids, scores), 1):
            by_id[track_id].source_ranks[source] = rank
            by_id[track_id].source_scores[source] = float(score)
    return result
