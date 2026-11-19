"""Normalização e tokenização de texto com suporte Unicode."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_accents: bool = False,
) -> str:
    """Normalize text without discarding semantic information by default.

    NFC unifies canonically equivalent Unicode representations. Accents are
    preserved unless ``strip_accents=True`` is explicitly requested.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text)
    if lowercase:
        normalized = normalized.lower()
    if strip_accents:
        decomposed = unicodedata.normalize("NFD", normalized)
        normalized = "".join(
            char for char in decomposed
            if unicodedata.category(char) != "Mn"
        )
        normalized = unicodedata.normalize("NFC", normalized)
    return " ".join(normalized.split())


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    strip_accents: bool = False,
) -> list[str]:
    """Split text into Unicode alphanumeric tokens.

    Punctuation and underscores are separators. Letters with accents and
    numbers are preserved. This baseline tokenizer does not perform stemming,
    lemmatization, subword segmentation or language-specific morphology.
    """
    normalized = normalize_text(
        text,
        lowercase=lowercase,
        strip_accents=strip_accents,
    )
    return _TOKEN_PATTERN.findall(normalized)
