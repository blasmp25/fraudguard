"""Evaluation metrics for imbalanced binary classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class EvaluationResult:
    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    fnr: float
    g_mean: float
    recall_at_p50: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def best_f1_threshold(y_true: ArrayLike, proba: ArrayLike) -> float:
    """Probability threshold that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # the last PR point has no threshold
    return float(thresholds[best])


def recall_at_precision(y_true: ArrayLike, proba: ArrayLike, min_precision: float) -> float:
    """Highest recall reachable while keeping precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(
    y_true: ArrayLike, proba: ArrayLike, threshold: float | None = None
) -> EvaluationResult:
    """Threshold-free metrics plus metrics at `threshold` (best-F1 threshold if None)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(proba, dtype=float)
    t = best_f1_threshold(y, p) if threshold is None else threshold
    pred = p >= t

    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return EvaluationResult(
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        threshold=t,
        precision=precision,
        recall=recall,
        f1=f1,
        fnr=1 - recall,
        g_mean=float(np.sqrt(recall * specificity)),
        recall_at_p50=recall_at_precision(y, p, 0.5),
    )
