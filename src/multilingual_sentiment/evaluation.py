"""Small dependency-free evaluation utilities for binary sentiment models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int]
) -> BinaryMetrics:
    """Calculate confusion counts and standard binary metrics."""

    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        raise ValueError("at least one label is required")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length")
    if any(label not in (0, 1) for label in actual + predicted):
        raise ValueError("binary labels must be 0 or 1")

    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)

    return BinaryMetrics(
        accuracy=_ratio(tp + tn, len(actual)),
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )

