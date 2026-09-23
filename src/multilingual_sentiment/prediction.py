"""A stable, serializable sentiment prediction contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .text import Language, detect_script_language, normalize_text


@dataclass(frozen=True)
class SentimentPrediction:
    text: str
    language: Language
    label: Literal["negative", "positive"]
    positive_probability: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


def prediction_from_probability(
    text: str, positive_probability: float, threshold: float = 0.5
) -> SentimentPrediction:
    if not 0.0 <= positive_probability <= 1.0:
        raise ValueError("positive_probability must be between 0 and 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("text must not be empty")

    return SentimentPrediction(
        text=normalized,
        language=detect_script_language(normalized),
        label="positive" if positive_probability >= threshold else "negative",
        positive_probability=positive_probability,
    )

