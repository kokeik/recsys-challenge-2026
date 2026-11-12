from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_example
from .dense import DenseTrackRetriever
from .evaluation import evaluate_rankings
from .features import FeatureExtractor
from .item2vec import Item2VecConfig
from .pipeline import CandidateGenerator
from .reranker import TurnAwareReranker
from .training import fit_item2vec, generate_oof_queries


def run_demo(
    data_path: Path,
    *,
    folds: int = 3,
    item_topn: int = 8,
    artist_topn: int = 5,
    embeddings_path: Path | None = None,
) -> dict:
    catalog, sessions, train, evaluation = load_example(data_path)
    dense, query_vectors = None, None
    if embeddings_path is not None:
        vectors = json.loads(embeddings_path.read_text(encoding="utf-8"))
        dense = DenseTrackRetriever(vectors["tracks"])
        query_vectors = {
            (r["session_id"], r["turn_number"]): r["vector"] for r in vectors["queries"]
        }
    config = Item2VecConfig(vector_size=32, epochs=40)
    oof = generate_oof_queries(
        sessions,
        train,
        catalog,
        folds=folds,
        config=config,
        item_topn=item_topn,
        artist_topn=artist_topn,
        dense=dense,
        query_vectors=query_vectors,
    )
    # Evaluation uses a model refit on ALL training sessions, never evaluation sessions.
    item2vec = fit_item2vec(sessions, catalog, config)
    generator = CandidateGenerator(
        catalog,
        item2vec,
        item_topn=item_topn,
        artist_topn=artist_topn,
        dense=dense,
        query_vectors=query_vectors,
    )
    reranker = TurnAwareReranker(FeatureExtractor(catalog, generator.source_names)).fit(oof.queries)
    rows, examples = [], []
    for query in evaluation:
        context = query.context
        if context.turn_number == 1:
            ranking = generator.first_turn(context)
        else:
            candidates = generator.generate(context)
            ranking = reranker.rank(candidates, context)
        rows.append((context.turn_number, ranking, query.target_track_id))
        examples.append(
            {
                "session_id": context.session_id,
                "turn_number": context.turn_number,
                "target": query.target_track_id,
                "top_5": ranking[:5],
            }
        )
    result = {
        "data_kind": "synthetic_demo",
        "backend": reranker.backend,
        "candidate_sources": ["item2vec", "artist_history"],
        "ranking_sources": list(generator.source_names),
        "oof_training_queries": len(oof.queries),
        "metrics": evaluate_rankings(rows, ks=(1, 5, 10)),
        "examples": examples,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic OOF + fixed-pool ranking demo")
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parents[2] / "examples" / "synthetic_data.json"
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--item-topn", type=int, default=8)
    parser.add_argument("--artist-topn", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, help="Optional local track/query embedding JSON")
    args = parser.parse_args()
    result = run_demo(
        args.data,
        folds=args.folds,
        item_topn=args.item_topn,
        artist_topn=args.artist_topn,
        embeddings_path=args.embeddings,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
