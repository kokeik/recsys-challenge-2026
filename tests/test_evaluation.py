import pytest

from turnaware_recsys.evaluation import evaluate_rankings


def test_metrics_are_reported_overall_and_by_turn():
    result = evaluate_rankings(
        [
            (2, ["target", "x"], "target"),
            (2, ["x", "target"], "target"),
            (3, ["x", "y"], "target"),
        ],
        ks=(1, 2),
    )
    assert result["all"]["hit@1"] == pytest.approx(1 / 3)
    assert result["turn_2"]["hit@2"] == 1.0
    assert result["turn_3"]["ndcg@2"] == 0.0

