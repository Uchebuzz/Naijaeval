"""Utility subpackage."""

from naijaeval.utils.text import (
    tokenize,
    normalize_text,
    extract_ngrams,
    extract_capitalized_ngrams,
    segment_by_language,
)

__all__ = [
    "tokenize",
    "normalize_text",
    "extract_ngrams",
    "extract_capitalized_ngrams",
    "segment_by_language",
]
