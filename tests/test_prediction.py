import pytest

from multilingual_sentiment.prediction import prediction_from_probability


def test_prediction_normalizes_and_labels_persian_text():
    prediction = prediction_from_probability("  اين فيلم عالي بود ", 0.91)
    assert prediction.text == "این فیلم عالی بود"
    assert prediction.language == "fa"
    assert prediction.label == "positive"


def test_prediction_respects_custom_threshold():
    prediction = prediction_from_probability("Not bad", 0.6, threshold=0.7)
    assert prediction.label == "negative"


def test_prediction_rejects_empty_text():
    with pytest.raises(ValueError, match="must not be empty"):
        prediction_from_probability("   ", 0.5)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_prediction_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="between 0 and 1"):
        prediction_from_probability("text", probability)
