from turnaware_recsys.candidate_pool import CandidatePoolBuilder


def test_union_preserves_source_rank_and_excludes_seen_items():
    pool = CandidatePoolBuilder({"a": 3, "b": 2}).build(
        {"a": ["seen", "x", "y"], "b": ["y", "z", "x"]},
        {"a": [0.9, 0.8, 0.7], "b": [0.6, 0.5, 0.4]},
        exclude=["seen"],
    )
    by_id = {candidate.track_id: candidate for candidate in pool}
    assert set(by_id) == {"x", "y", "z"}
    assert by_id["x"].source_ranks == {"a": 2}
    assert by_id["y"].source_ranks == {"a": 3, "b": 1}
    assert by_id["z"].source_scores["b"] == 0.5

