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

