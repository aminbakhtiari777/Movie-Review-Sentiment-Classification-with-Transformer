"""Unicode-safe text utilities for Persian and English input."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Language = Literal["fa", "en", "mixed", "unknown"]

_CHARACTER_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
    }
)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_WHITESPACE = re.compile(r"\s+")
_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Normalize presentation variants without translating or deleting content."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_CHARACTER_MAP)
    normalized = normalized.replace("ـ", "")
    normalized = _DIACRITICS.sub("", normalized)
    normalized = normalized.lower()
    return _WHITESPACE.sub(" ", normalized).strip()


def detect_script_language(text: str) -> Language:
    """Detect Persian/Arabic script, Latin script, mixed text, or no letters.

    This is intentionally script detection, not a claim of full linguistic
    identification. It is deterministic and suitable for routing and telemetry.
    """

    normalized = normalize_text(text)
    has_persian = bool(_PERSIAN.search(normalized))
    has_latin = bool(_LATIN.search(normalized))

    if has_persian and has_latin:
        return "mixed"
    if has_persian:
        return "fa"
    if has_latin:
        return "en"
    return "unknown"

