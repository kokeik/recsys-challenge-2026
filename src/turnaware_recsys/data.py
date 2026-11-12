from __future__ import annotations

import json
from pathlib import Path

from .features import QueryContext, Track
from .training import LabeledQuery, TrainingSession


def load_example(
    path: Path,
) -> tuple[dict[str, Track], list[TrainingSession], list[LabeledQuery], list[LabeledQuery]]:
    """Read the documented portable JSON format (no challenge download is performed)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    tracks = [
        Track(
            r["track_id"],
            r["title"],
            r["artist"],
            r.get("album", ""),
            tuple(r.get("tags", ())),
            r.get("year"),
        )
        for r in raw["catalog"]
    ]
    catalog = {t.track_id: t for t in tracks}
    if not catalog or len(catalog) != len(tracks):
        raise ValueError("Catalog must contain unique track IDs")
    sessions = [
        TrainingSession(r["session_id"], tuple(r["tracks"]), r.get("text_relations", {}))
        for r in raw["train_sessions"]
    ]

    def parse_query(row: dict) -> LabeledQuery:
        context = QueryContext(
            row["session_id"],
            row["turn_number"],
            row["query"],
            tuple(row["history"]),
            row.get("category", "unknown"),
            row.get("specificity", "unknown"),
        )
        if context.turn_number < 1:
            raise ValueError("Turn numbers start at 1")
        if context.turn_number == 1 and context.history_track_ids:
            raise ValueError("Turn 1 must have no item history")
        if row["target"] in context.history_track_ids:
            raise ValueError("Target must not appear in the observed history")
        return LabeledQuery(context, row["target"])

    train = [parse_query(r) for r in raw["train_queries"]]
    evaluation = [parse_query(r) for r in raw["evaluation_queries"]]
    train_ids = {s.session_id for s in sessions}
    if train_ids & {q.context.session_id for q in evaluation}:
        raise ValueError("Training and evaluation sessions must be disjoint")
    if len(train_ids) != len(sessions):
        raise ValueError("Training session IDs must be unique")
    for session in sessions:
        if not set(session.tracks) <= set(catalog):
            raise ValueError("Session contains an unknown track")
    for query in train + evaluation:
        if not {*query.context.history_track_ids, query.target_track_id} <= set(catalog):
            raise ValueError("Query contains an unknown track")
    return catalog, sessions, train, evaluation
