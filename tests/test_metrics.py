import math

from epicatanexus.metrics import hit_at_k, ndcg_at_k, regression_metrics


def test_regression_metrics_for_perfect_prediction():
    metrics = regression_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert metrics["r2"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert math.isclose(metrics["pcc"], 1.0)


def test_ranking_metrics():
    assert hit_at_k([False, True, False], 2) == 1.0
    assert hit_at_k([False, True, False], 1) == 0.0
    assert ndcg_at_k([1.0, 0.0, 0.0], 3) == 1.0

