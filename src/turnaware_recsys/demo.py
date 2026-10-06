from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_pool import CandidatePoolBuilder
from .evaluation import evaluate_rankings
from .features import FeatureExtractor, QueryContext, Track
from .item2vec import Item2VecConfig, RelationItem2Vec
from .lexical import TfidfTrackRetriever
from .reranker import TrainingQuery, TurnAwareReranker


def _artist_history_ranking(history: tuple[str, ...], catalog: dict[str, Track]) -> list[str]:
    artists = {catalog[item].artist for item in history if item in catalog}
    return [
        track_id
        for track_id, track in sorted(catalog.items())
        if track.artist in artists and track_id not in history
    ]


def _load(path: Path):
    raw = json.loads(path.read_text())
    catalog = {
        row["track_id"]: Track(
            track_id=row["track_id"],
            title=row["title"],
            artist=row["artist"],
            album=row.get("album", ""),
            tags=tuple(row.get("tags", [])),
            year=row.get("year"),
        )
        for row in raw["catalog"]
    }
    return raw, catalog


def _candidate_query(
    row: dict,
    catalog: dict[str, Track],
    item2vec: RelationItem2Vec,
    lexical: TfidfTrackRetriever,
    pool_builder: CandidatePoolBuilder,
):
    context = QueryContext(
        session_id=row["session_id"],
        turn_number=row["turn_number"],
        query_text=row["query"],
        history_track_ids=tuple(row["history"]),
        category=row.get("category", "unknown"),
        specificity=row.get("specificity", "unknown"),
    )
    item_ids, item_scores = item2vec.recommend(context.history_track_ids, top_k=8)
    text_ids, text_scores = lexical.search(
        context.query_text, top_k=8, exclude=context.history_track_ids
    )
    artist_ids = _artist_history_ranking(context.history_track_ids, catalog)
    candidates = pool_builder.build(
        {"item2vec": item_ids, "tfidf": text_ids, "artist_history": artist_ids},
        {"item2vec": item_scores, "tfidf": text_scores},
        exclude=context.history_track_ids,
    )
    return context, candidates


def run_demo(data_path: Path) -> dict:
    raw, catalog = _load(data_path)
    sequences = [row["tracks"] for row in raw["train_sessions"]]
    item2vec = RelationItem2Vec(Item2VecConfig(vector_size=32, epochs=40)).fit(
        sequences, catalog
    )
    lexical = TfidfTrackRetriever().fit(catalog)
    pool_builder = CandidatePoolBuilder(
        {"item2vec": 8, "tfidf": 8, "artist_history": 5}
    )
    extractor = FeatureExtractor(
        catalog=catalog, source_names=("item2vec", "tfidf", "artist_history")
    )
    reranker = TurnAwareReranker(extractor)

    training_queries = []
    for row in raw["train_queries"]:
        context, candidates = _candidate_query(row, catalog, item2vec, lexical, pool_builder)
        training_queries.append(
            TrainingQuery(context=context, candidates=candidates, target_track_id=row["target"])
        )
    reranker.fit(training_queries)

    evaluation_rows = []
    examples = []
    for row in raw["evaluation_queries"]:
        context, candidates = _candidate_query(row, catalog, item2vec, lexical, pool_builder)
        ranking = reranker.rank(candidates, context)
        evaluation_rows.append((context.turn_number, ranking, row["target"]))
        examples.append(
            {
                "session_id": context.session_id,
                "turn_number": context.turn_number,
                "target": row["target"],
                "top_5": ranking[:5],
            }
        )
    return {
        "backend": reranker.backend,
        "metrics": evaluate_rankings(evaluation_rows, ks=(5, 10)),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic end-to-end demo")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parents[2] / "examples" / "synthetic_data.json",
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.data), indent=2))


if __name__ == "__main__":
    main()

