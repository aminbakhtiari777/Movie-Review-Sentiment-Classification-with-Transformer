import pytest

from multilingual_sentiment.text import detect_script_language, normalize_text


def test_normalize_persian_variants_and_whitespace():
    assert normalize_text("  كِتاب\n  يک  ") == "کتاب یک"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("این فیلم عالی بود", "fa"),
        ("This movie was great", "en"),
        ("این movie عالی بود", "mixed"),
        ("123 !!!", "unknown"),
    ],
)
def test_detect_script_language(text, expected):
    assert detect_script_language(text) == expected


def test_normalize_requires_string():
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]

