import pytest

from turnaware_recsys.features import QueryContext, Track
from turnaware_recsys.training import (
    LabeledQuery,
    TrainingSession,
    generate_oof_queries,
    session_folds,
)


def test_oof_excludes_entire_validation_sessions_and_their_text_relations():
    catalog = {f"t{i}": Track(f"t{i}", f"Track {i}", f"Artist {i}") for i in range(6)}
    sessions = [
        TrainingSession(
            f"s{i}", (f"t{2 * i}", f"t{2 * i + 1}"), {f"t{2 * i + 1}": [f"private_text_s{i}"]}
        )
        for i in range(3)
    ]
    queries = [
        LabeledQuery(QueryContext(s.session_id, t, "query", (s.tracks[0],)), s.tracks[1])
        for s in sessions
        for t in (2, 3)
    ]
    fitted = []

    class RecordingRetriever:
        def fit(self, sequences, shared_catalog, text_relations):
            fitted.append((sequences, shared_catalog, text_relations))
            return self

        def recommend(self, history, *, top_k):
            ids = [tid for tid in sorted(catalog) if tid not in history][:top_k]
            return ids, [1.0] * len(ids)

    result = generate_oof_queries(
        sessions, queries, catalog, folds=3, retriever_factory=RecordingRetriever
    )
    assert len(result.queries) == len(queries)
    for audit, (sequences, shared, texts) in zip(result.folds, fitted):
        assert not set(audit.train_session_ids) & set(audit.validation_session_ids)
        assert shared is catalog
        training_tracks = {tid for sequence in sequences for tid in sequence}
        training_tokens = {token for tokens in texts.values() for token in tokens}
        for session in sessions:
            if session.session_id in audit.validation_session_ids:
                assert not set(session.tracks) & training_tracks
                assert f"private_text_{session.session_id}" not in training_tokens
    assert session_folds(["c", "a", "b"]) == session_folds(["b", "c", "a"])


def test_oof_rejects_unmatched_query_session():
    sessions = [TrainingSession("a", ("x", "y")), TrainingSession("b", ("x", "y"))]
    with pytest.raises(ValueError, match="matching training session"):
        generate_oof_queries(
            sessions,
            [LabeledQuery(QueryContext("missing", 2, "text"), "x")],
            {"x": Track("x", "x", "A")},
            folds=2,
        )
