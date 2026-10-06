from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import log2


def _metrics_at_k(ranking: Sequence[str], target: str, k: int) -> tuple[float, float]:
    try:
        rank = ranking[:k].index(target) + 1
    except ValueError:
        return 0.0, 0.0
    return 1.0, 1.0 / log2(rank + 1)


def evaluate_rankings(
    rows: Iterable[tuple[int, Sequence[str], str]], ks: Sequence[int] = (5, 10, 20)
) -> dict[str, dict[str, float]]:
    """Compute hit rate and nDCG overall and by turn.

    Each row is ``(turn_number, ranked_track_ids, target_track_id)``.
    """

    buckets: dict[str, list[tuple[Sequence[str], str]]] = defaultdict(list)
    for turn, ranking, target in rows:
        buckets["all"].append((ranking, target))
        buckets[f"turn_{turn}"].append((ranking, target))

    result: dict[str, dict[str, float]] = {}
    for bucket, examples in sorted(buckets.items()):
        metrics: dict[str, float] = {"count": float(len(examples))}
        for k in ks:
            values = [_metrics_at_k(ranking, target, k) for ranking, target in examples]
            metrics[f"hit@{k}"] = sum(value[0] for value in values) / len(values)
            metrics[f"ndcg@{k}"] = sum(value[1] for value in values) / len(values)
        result[bucket] = metrics
    return result

