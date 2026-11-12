from turnaware_recsys.candidate_pool import Candidate
from turnaware_recsys.features import FeatureExtractor, QueryContext, Track


def test_history_and_rank_features():
    catalog = {
        "old": Track("old", "Old Song", "Artist A", "Album A", ("rock",), 2020),
        "new": Track("new", "New Guitar", "Artist A", "Album B", ("rock",), 2021),
    }
    candidate = Candidate("new", {"item2vec": 2}, {"item2vec": 0.75})
    context = QueryContext("s1", 2, "more rock guitar", ("old",))
    values = FeatureExtractor(catalog, ("item2vec",)).transform_one(candidate, context)
    assert values["item2vec_present"] == 1.0
    assert values["item2vec_inv_rank"] == 0.5
    assert values["same_artist_ratio"] == 1.0
    assert values["same_album_ratio"] == 0.0
    assert values["query_track_overlap"] > 0.0


def test_empty_album_is_not_a_history_match_and_scores_are_optional():
    catalog = {"a": Track("a", "rock", "Artist"), "b": Track("b", "rock", "Artist")}
    candidate = Candidate("b", {"item2vec": 2}, {"item2vec": 99.0})
    extractor = FeatureExtractor(catalog, ("item2vec",))
    values = extractor.transform_one(candidate, QueryContext("s", 2, "rock", ("a",)))
    assert values["same_album_ratio"] == 0
    assert values["item2vec_inv_rank_x_artist_ratio"] == 0.5
    assert "item2vec_score" not in values
    extractor.include_source_scores = True
    assert extractor.transform_one(candidate, QueryContext("s", 2, "rock"))["item2vec_score"] == 99
