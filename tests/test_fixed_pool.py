import pytest

from turnaware_recsys.bm25 import BM25TrackRetriever
from turnaware_recsys.candidate_pool import Candidate, attach_fixed_pool_rankings
from turnaware_recsys.dense import DenseTrackRetriever
from turnaware_recsys.features import QueryContext, Track
from turnaware_recsys.pipeline import CandidateGenerator


def test_rescoring_cannot_add_or_drop_tracks_and_preserves_original_ranks():
    pool = [Candidate("a", {"item2vec": 1}), Candidate("b", {"artist_history": 1})]
    result = attach_fixed_pool_rankings(pool, {"text": (["b", "a"], [0.9, 0.1])})
    assert [c.track_id for c in result] == ["a", "b"]
    assert result[0].source_ranks == {"item2vec": 1, "text": 2}
    assert pool[0].source_ranks == {"item2vec": 1}
    with pytest.raises(ValueError, match="exactly"):
        attach_fixed_pool_rankings(pool, {"text": (["b", "outside"], [1, 0])})
    with pytest.raises(ValueError, match="exactly"):
        attach_fixed_pool_rankings(pool, {"text": (["a", "a"], [1, 0])})


def test_text_match_outside_pool_does_not_become_a_candidate():
    class FixedItem2Vec:
        def recommend(self, history, *, top_k):
            return ["candidate"][:top_k], [0.5][:top_k]

    catalog = {
        "seen": Track("seen", "old", "A"),
        "candidate": Track("candidate", "unrelated", "B"),
        "outside": Track("outside", "exact query jazz", "C"),
    }
    generator = CandidateGenerator(catalog, FixedItem2Vec(), item_topn=1, artist_topn=0)
    context = QueryContext("s", 2, "exact query jazz", ("seen",))
    pool = generator.generate(context)
    assert [c.track_id for c in pool] == ["candidate"]
    assert pool[0].source_ranks["tfidf"] == 1
    assert pool[0].source_ranks["bm25"] == 1
    assert generator.generate(QueryContext("s", 1, "jazz")) == []
    assert generator.first_turn(QueryContext("s", 1, "exact query jazz"), top_k=1) == ["outside"]


def test_artist_history_adds_unseen_tracks_and_retains_source_rank():
    class FixedItem2Vec:
        def recommend(self, history, *, top_k):
            return ["b"], [0.5]

    catalog = {
        "seen": Track("seen", "old", "A"),
        "a": Track("a", "other", "A"),
        "b": Track("b", "similar", "B"),
    }
    pool = CandidateGenerator(catalog, FixedItem2Vec(), artist_topn=1).generate(
        QueryContext("s", 2, "other", ("seen",))
    )
    assert {c.track_id for c in pool} == {"a", "b"}
    assert next(c for c in pool if c.track_id == "a").source_ranks["artist_history"] == 1


def test_bm25_and_dense_rank_the_supplied_pool_including_zero_matches():
    catalog = {"a": Track("a", "guitar rock", "A"), "b": Track("b", "piano jazz", "B")}
    bm25 = BM25TrackRetriever().fit(catalog)
    assert bm25.rank_candidates("jazz", ["a", "b"])[0] == ["b", "a"]
    assert bm25.rank_candidates("unknown", ["b", "a"])[0] == ["b", "a"]
    dense = DenseTrackRetriever({"a": [10, 0], "b": [0, 2], "outside": [1, 1]})
    assert dense.rank_candidates([0, 1], ["a", "b"])[0] == ["b", "a"]
    with pytest.raises(ValueError, match="nonzero"):
        dense.rank_candidates([0, 0], ["a"])
    with pytest.raises(ValueError, match="dimension"):
        dense.rank_candidates([1], ["a"])
