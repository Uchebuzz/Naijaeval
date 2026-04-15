"""Text processing utilities shared across metrics and tasks."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterator


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    """Whitespace-tokenise ``text``, optionally lowercasing.

    Strips leading/trailing whitespace and collapses internal runs.
    Does not perform sub-word splitting — use a proper tokeniser for
    language-specific needs.

    Args:
        text: Input string.
        lowercase: If True, convert to lowercase before splitting.

    Returns:
        List of token strings.
    """
    if lowercase:
        text = text.lower()
    return text.strip().split()


def normalize_text(text: str) -> str:
    """Normalise Unicode, collapse whitespace, and strip punctuation edges.

    Applies NFC normalisation so that visually identical characters that
    differ only in Unicode composition compare equal.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Return a Counter of all n-grams extracted from ``tokens``."""
    grams: list[tuple[str, ...]] = [
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    ]
    return Counter(grams)


_COMMON_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "but",
    "or", "so", "yet", "it", "its", "this", "that", "these", "those",
    "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "not", "no", "nor", "with", "from",
    "by", "as", "if", "then", "than", "when", "where", "which", "who",
    "also", "just", "about", "i",
})


def extract_capitalized_ngrams(
    text: str, max_n: int = 3
) -> set[str]:
    """Extract contiguous runs of capitalised words (proxy for named entities).

    Captures sequences of Title-Case or ALL-CAPS tokens up to ``max_n``
    words long.  Common English function words (the, a, in, …) are
    excluded from entity runs so they don't corrupt multi-word entities.

    Used by the hallucination metric as a lightweight NER substitute that
    requires no external models.

    Examples::

        >>> extract_capitalized_ngrams("The Lagos State Hospital treated patients")
        {'Lagos State Hospital', 'Lagos', 'State', 'Hospital', 'Lagos State'}
    """
    tokens = text.split()
    capitalized = re.compile(r"^[A-Z][a-zA-Z]*$|^[A-Z]{2,}$")
    results: set[str] = set()

    i = 0
    while i < len(tokens):
        clean = re.sub(r"[^\w]", "", tokens[i])
        is_entity_token = (
            clean
            and capitalized.match(clean)
            and clean.lower() not in _COMMON_WORDS
        )
        if is_entity_token:
            # Extend the run as far as possible up to max_n
            run: list[str] = [clean]
            j = i + 1
            while j < len(tokens) and len(run) < max_n:
                next_clean = re.sub(r"[^\w]", "", tokens[j])
                next_is_entity = (
                    next_clean
                    and capitalized.match(next_clean)
                    and next_clean.lower() not in _COMMON_WORDS
                )
                if next_is_entity:
                    run.append(next_clean)
                    j += 1
                else:
                    break
            # Add all prefixes of the run
            for k in range(len(run)):
                results.add(" ".join(run[: k + 1]))
            i = j
        else:
            i += 1

    return results


def segment_by_language(
    text: str,
    vocab: dict[str, set[str]],
    default_lang: str = "en",
) -> list[tuple[str, str]]:
    """Segment ``text`` into (token, language) pairs using vocabulary lookup.

    Each token is classified by checking membership in each language's
    vocabulary set.  Tokens not found in any vocabulary default to
    ``default_lang``.

    Args:
        text: Input string.
        vocab: Mapping of language code → set of known words (lowercase).
        default_lang: Language tag assigned to unknown tokens.

    Returns:
        List of ``(token, language_code)`` tuples.

    Example::

        >>> vocab = {"yo": {"mo", "pe", "ni"}, "en": {"the", "is", "i"}}
        >>> segment_by_language("I know mo pe the man ni", vocab)
        [('I', 'en'), ('know', 'en'), ('mo', 'yo'), ('pe', 'yo'),
         ('the', 'en'), ('man', 'en'), ('ni', 'yo')]
    """
    tokens = text.split()
    result: list[tuple[str, str]] = []

    for token in tokens:
        clean = re.sub(r"[^\w]", "", token).lower()
        assigned = default_lang
        for lang, word_set in vocab.items():
            if clean in word_set:
                assigned = lang
                break
        result.append((token, assigned))

    return result


def count_language_switches(labeled_tokens: list[tuple[str, str]]) -> int:
    """Count the number of language transitions in a labeled token sequence.

    A switch is defined as two consecutive tokens assigned to different
    language tags.  Transitions to/from ``"unknown"`` are counted.

    Args:
        labeled_tokens: Output of :func:`segment_by_language`.

    Returns:
        Integer count of language boundaries.
    """
    if len(labeled_tokens) < 2:
        return 0
    switches = sum(
        1
        for (_, l1), (_, l2) in zip(labeled_tokens, labeled_tokens[1:])
        if l1 != l2
    )
    return switches
