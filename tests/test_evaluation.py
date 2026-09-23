import pytest

from multilingual_sentiment.evaluation import binary_classification_metrics


def test_binary_metrics_from_known_confusion_matrix():
    metrics = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert (metrics.true_positive, metrics.true_negative) == (1, 1)
    assert (metrics.false_positive, metrics.false_negative) == (1, 1)


def test_binary_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        binary_classification_metrics([0, 1], [0])


def test_binary_metrics_reject_non_binary_labels():
    with pytest.raises(ValueError, match="0 or 1"):
        binary_classification_metrics([0, 2], [0, 1])


def test_binary_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        binary_classification_metrics([], [])
