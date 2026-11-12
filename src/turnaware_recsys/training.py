from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256

from .dense import DenseTrackRetriever
from .features import QueryContext, Track
from .item2vec import Item2VecConfig, RelationItem2Vec
from .pipeline import CandidateGenerator
from .reranker import TrainingQuery


@dataclass(frozen=True)
class TrainingSession:
    session_id: str
    tracks: tuple[str, ...]
    text_relations: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class LabeledQuery:
    context: QueryContext
    target_track_id: str


@dataclass(frozen=True)
class FoldAudit:
    fold: int
    train_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]


@dataclass
class OOFResult:
    queries: list[TrainingQuery]
    folds: list[FoldAudit]


def session_folds(
    session_ids: Sequence[str], folds: int = 3, seed: int = 42
) -> list[tuple[str, ...]]:
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Training session IDs must be unique")
    if not 2 <= folds <= len(session_ids):
        raise ValueError("Require 2 <= folds <= number of sessions")
    ordered = sorted(session_ids, key=lambda sid: sha256(f"{seed}:{sid}".encode()).digest())
    return [tuple(ordered[i::folds]) for i in range(folds)]


def fit_item2vec(
    sessions: Sequence[TrainingSession],
    catalog: Mapping[str, Track],
    config: Item2VecConfig | None = None,
    *,
    retriever: RelationItem2Vec | None = None,
) -> RelationItem2Vec:
    """Combine only the selected sessions' item sequences and text relations."""
    texts: dict[str, list[str]] = defaultdict(list)
    for session in sessions:
        for track_id, tokens in session.text_relations.items():
            if track_id not in session.tracks:
                raise ValueError("Text relation must refer to a track in its own session")
            texts[track_id].extend(tokens)
    model = retriever if retriever is not None else RelationItem2Vec(config)
    return model.fit([s.tracks for s in sessions], catalog, texts)


def generate_oof_queries(
    sessions: Sequence[TrainingSession],
    queries: Sequence[LabeledQuery],
    catalog: Mapping[str, Track],
    *,
    folds: int = 3,
    seed: int = 42,
    config: Item2VecConfig | None = None,
    item_topn: int = 100,
    artist_topn: int = 50,
    retriever_factory: Callable[[], RelationItem2Vec] | None = None,
    dense: DenseTrackRetriever | None = None,
    query_vectors: Mapping[tuple[str, int], Sequence[float]] | None = None,
) -> OOFResult:
    """Exclude every held-out session's sequences AND text relations before fitting.

    Catalog metadata is shared; no validation target is injected into candidate pools.
    A frozen, externally supplied dense encoder must itself be independent of held-out labels.
    """
    ids = [s.session_id for s in sessions]
    assignments = session_folds(ids, folds, seed)
    if any(q.context.session_id not in set(ids) for q in queries):
        raise ValueError("Every training query must have a matching training session")
    keys = [(q.context.session_id, q.context.turn_number) for q in queries]
    if len(set(keys)) != len(keys):
        raise ValueError("Training query session/turn keys must be unique")
    output: dict[tuple[str, int], TrainingQuery] = {}
    audits = []
    for fold, valid_ids in enumerate(assignments):
        valid = set(valid_ids)
        train = [s for s in sessions if s.session_id not in valid]
        retriever = (
            retriever_factory() if retriever_factory is not None else RelationItem2Vec(config)
        )
        retriever = fit_item2vec(train, catalog, retriever=retriever)
        generator = CandidateGenerator(
            catalog,
            retriever,
            item_topn=item_topn,
            artist_topn=artist_topn,
            dense=dense,
            query_vectors=query_vectors,
        )
        for query in queries:
            context = query.context
            if context.session_id in valid and context.turn_number >= 2:
                output[(context.session_id, context.turn_number)] = TrainingQuery(
                    context, generator.generate(context), query.target_track_id
                )
        audits.append(
            FoldAudit(fold, tuple(sorted(s.session_id for s in train)), tuple(sorted(valid)))
        )
    return OOFResult([output[key] for key in sorted(output)], audits)
