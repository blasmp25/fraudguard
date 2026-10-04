import pytest

from fraudguard.models.metrics import evaluate, recall_at_precision


def test_perfect_ranking() -> None:
    result = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result.pr_auc == pytest.approx(1.0)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.threshold == pytest.approx(0.8)
    assert result.fnr == 0.0


def test_recall_at_precision_by_hand() -> None:
    # Sorted by score: 0.9 fraud, 0.8 legit, 0.7 fraud, 0.1 legit
    # Cut at 0.7 -> 2 TP, 1 FP: precision 0.67, recall 1.0
    # Cut at 0.9 -> 1 TP, 0 FP: precision 1.0, recall 0.5
    y, p = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    assert recall_at_precision(y, p, 0.6) == pytest.approx(1.0)
    assert recall_at_precision(y, p, 0.9) == pytest.approx(0.5)


def test_fixed_threshold_counts() -> None:
    # Threshold 0.5 -> predictions [1, 1, 0, 0]: TP=1, FP=1, FN=1
    result = evaluate([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1], threshold=0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.fnr == pytest.approx(0.5)
