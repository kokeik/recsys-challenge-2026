from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from .bm25 import BM25TrackRetriever
from .candidate_pool import Candidate, CandidatePoolBuilder, attach_fixed_pool_rankings
from .dense import DenseTrackRetriever
from .features import QueryContext, Track
from .item2vec import RelationItem2Vec
from .lexical import TfidfTrackRetriever


def artist_history_ranking(history: Sequence[str], catalog: Mapping[str, Track]) -> list[str]:
    """Retrieve unseen tracks by artists in history; ties break by track ID."""
    counts = Counter(
        catalog[tid].artist for tid in history if tid in catalog and catalog[tid].artist
    )
    seen = set(history)
    return sorted(
        (tid for tid, track in catalog.items() if tid not in seen and track.artist in counts),
        key=lambda tid: (-counts[catalog[tid].artist], tid),
    )


class CandidateGenerator:
    """Item2Vec + artist pool, followed by independent rankings of that fixed pool."""

    def __init__(
        self,
        catalog: Mapping[str, Track],
        item2vec: RelationItem2Vec,
        *,
        item_topn: int = 100,
        artist_topn: int = 50,
        dense: DenseTrackRetriever | None = None,
        query_vectors: Mapping[tuple[str, int], Sequence[float]] | None = None,
    ) -> None:
        if item_topn < 0 or artist_topn < 0:
            raise ValueError("Candidate limits must not be negative")
        self.catalog, self.item2vec = catalog, item2vec
        self.item_topn, self.artist_topn = item_topn, artist_topn
        self.lexical = TfidfTrackRetriever().fit(catalog)
        self.bm25 = BM25TrackRetriever().fit(catalog)
        self.dense = dense
        self.query_vectors = query_vectors or {}
        self.source_names = (
            "item2vec",
            "artist_history",
            "tfidf",
            "bm25",
            "tfidf_history",
            "bm25_history",
        )
        if dense is not None:
            self.source_names += ("dense",)

    def generate(self, context: QueryContext) -> list[Candidate]:
        if context.turn_number < 2:
            return []
        ids, scores = self.item2vec.recommend(context.history_track_ids, top_k=self.item_topn)
        artists = artist_history_ranking(context.history_track_ids, self.catalog)
        candidates = CandidatePoolBuilder(
            {"item2vec": self.item_topn, "artist_history": self.artist_topn}
        ).build(
            {"item2vec": ids, "artist_history": artists},
            {"item2vec": scores},
            exclude=context.history_track_ids,
        )
        pool = [c.track_id for c in candidates]
        history = [self.catalog[tid] for tid in context.history_track_ids if tid in self.catalog]
        history_tags = " ".join(tag for track in history for tag in track.tags)
        history_artist_album = " ".join(f"{track.artist} {track.album}" for track in history)
        rankings = {
            "tfidf": self.lexical.rank_candidates(context.query_text, pool),
            "bm25": self.bm25.rank_candidates(context.query_text, pool),
            "tfidf_history": self.lexical.rank_candidates(history_tags, pool),
            "bm25_history": self.bm25.rank_candidates(history_artist_album, pool),
        }
        if self.dense is not None:
            vector = self.query_vectors[(context.session_id, context.turn_number)]
            rankings["dense"] = self.dense.rank_candidates(vector, pool)
        return attach_fixed_pool_rankings(candidates, rankings)

    def first_turn(self, context: QueryContext, top_k: int = 20) -> list[str]:
        """History-free route; it never invokes the sequential ranker."""
        if self.dense is not None:
            vector = self.query_vectors[(context.session_id, context.turn_number)]
            return self.dense.search(vector, top_k=top_k)[0]
        return self.lexical.search(context.query_text, top_k=top_k)[0]
