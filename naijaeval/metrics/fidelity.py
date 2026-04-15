"""Fidelity metrics: terminology preservation, BLEU, and chrF.

These metrics evaluate how faithfully a model's output preserves
content from the source — particularly domain-critical terminology
that has no acceptable substitute in healthcare, legal, or financial
contexts.

Metrics
-------
- :class:`TerminologyPreservationMetric` — what fraction of
  domain-critical terms survive through translation or summarisation.
- :class:`BLEUMetric` — standard BLEU via sacrebleu.
- :class:`chrFMetric` — character F-score via sacrebleu; more
  appropriate than BLEU for morphologically rich African languages.
"""

from __future__ import annotations

import re
from typing import Any

from naijaeval.data.domain_terms import DOMAIN_TERMS
from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import register_metric


@register_metric("terminology_preservation_rate")
class TerminologyPreservationMetric(BaseMetric):
    """Measure what fraction of domain-critical terms appear in model outputs.

    **Motivation**

    Standard n-gram metrics (BLEU, chrF) do not weight domain terminology
    differently from common words.  In medical translation, preserving the
    word "hypertension" matters more than preserving "the".  This metric
    directly evaluates term survival.

    **Algorithm**

    For each prediction:

    1. Normalise to lowercase.
    2. For each canonical term in ``terms``, check whether any of its
       acceptable surface forms appear in the prediction (substring match).
    3. Preservation rate = ``n_preserved / n_total_terms``.

    The batch score is the mean over all predictions.

    **Providing term lists**

    Option A — built-in domain::

        metric = TerminologyPreservationMetric(domain="medical")

    Option B — custom dict::

        metric = TerminologyPreservationMetric(terms={
            "hypertension": ["hypertension", "high blood pressure"],
            "malaria": ["malaria"],
        })

    Option C — pass at compute time::

        metric.compute(predictions, references, domain="legal")

    Args:
        terms: Custom term dictionary ``{canonical: [surface_forms]}``.
        domain: One of ``"medical"``, ``"legal"``, ``"financial"``,
            ``"customer_support"``.  Used if ``terms`` is ``None``.

    Raises:
        ValueError: If neither ``terms`` nor ``domain`` is provided and
            none are passed to :meth:`compute`.
    """

    name = "terminology_preservation_rate"
    description = (
        "Fraction of domain-critical terms that appear in model outputs. "
        "Range: [0, 1]; 1.0 = all terms preserved."
    )
    higher_is_better = True

    def __init__(
        self,
        terms: dict[str, list[str]] | None = None,
        domain: str | None = None,
    ) -> None:
        self.terms = terms
        self.domain = domain

    def _resolve_terms(
        self,
        terms: dict[str, list[str]] | None,
        domain: str | None,
    ) -> dict[str, list[str]]:
        if terms is not None:
            return terms
        if domain is not None:
            if domain not in DOMAIN_TERMS:
                raise ValueError(
                    f"Unknown domain '{domain}'. "
                    f"Available: {sorted(DOMAIN_TERMS.keys())}"
                )
            return DOMAIN_TERMS[domain]
        if self.terms is not None:
            return self.terms
        if self.domain is not None:
            if self.domain not in DOMAIN_TERMS:
                raise ValueError(
                    f"Unknown domain '{self.domain}'. "
                    f"Available: {sorted(DOMAIN_TERMS.keys())}"
                )
            return DOMAIN_TERMS[self.domain]
        raise ValueError(
            "No term list provided. Pass 'terms' or 'domain' to the "
            "constructor or to compute()."
        )

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        *,
        terms: dict[str, list[str]] | None = None,
        domain: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute terminology preservation rate.

        Args:
            predictions: Model outputs.
            references: Accepted for API consistency but not used.
            terms: Override term dictionary for this call.
            domain: Override domain for this call.

        Returns:
            MetricResult with:
            - ``score``: mean preservation rate across predictions
            - ``details["per_sample_rates"]``: preservation rate per sample
            - ``details["per_term_recall"]``: fraction of samples where each
              term was preserved
        """
        resolved_terms = self._resolve_terms(terms, domain)
        if not predictions:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        per_sample_rates: list[float] = []
        term_hit_counts: dict[str, int] = {t: 0 for t in resolved_terms}

        for pred in predictions:
            pred_norm = pred.lower()
            hits = 0
            for canonical, forms in resolved_terms.items():
                matched = any(form.lower() in pred_norm for form in forms)
                if matched:
                    hits += 1
                    term_hit_counts[canonical] += 1
            rate = hits / len(resolved_terms) if resolved_terms else 1.0
            per_sample_rates.append(rate)

        mean_rate = sum(per_sample_rates) / len(per_sample_rates)
        n = len(predictions)

        return MetricResult(
            name=self.name,
            score=round(mean_rate, 4),
            details={
                "per_sample_rates": [round(r, 4) for r in per_sample_rates],
                "per_term_recall": {
                    term: round(count / n, 4)
                    for term, count in term_hit_counts.items()
                },
                "missing_terms": [
                    term
                    for term, count in term_hit_counts.items()
                    if count / n < 0.5
                ],
            },
            metadata={
                "n_samples": n,
                "n_terms": len(resolved_terms),
                "domain": domain or self.domain or "custom",
            },
        )


@register_metric("bleu")
class BLEUMetric(BaseMetric):
    """BLEU score via sacrebleu.

    Uses sacrebleu's corpus-level BLEU with default tokenisation.
    Sentence-level scores are also reported in ``details``.

    Args:
        tokenize: sacrebleu tokenisation scheme. Default ``"13a"``
            (standard). Use ``"intl"`` for non-Latin scripts or
            ``"none"`` if pre-tokenised.

    References:
        Papineni et al. (2002). BLEU: a Method for Automatic Evaluation
        of Machine Translation. ACL.
    """

    name = "bleu"
    description = "Corpus-level BLEU score (sacrebleu). Range: [0, 100]."
    higher_is_better = True

    def __init__(self, tokenize: str = "13a") -> None:
        self.tokenize = tokenize

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required: pip install sacrebleu")

        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) must have the same length."
            )

        corpus_score = sacrebleu.corpus_bleu(
            predictions,
            [references],
            tokenize=self.tokenize,
        )
        sentence_scores = [
            sacrebleu.sentence_bleu(pred, [ref]).score
            for pred, ref in zip(predictions, references)
        ]

        return MetricResult(
            name=self.name,
            score=round(corpus_score.score, 2),
            details={
                "sentence_scores": [round(s, 2) for s in sentence_scores],
                "bp": round(corpus_score.bp, 4),
                "precisions": [round(p, 2) for p in corpus_score.precisions],
            },
            metadata={
                "n_samples": len(predictions),
                "tokenize": self.tokenize,
                "sacrebleu_signature": str(corpus_score.get_signature()),
            },
        )


@register_metric("chrf")
class chrFMetric(BaseMetric):
    """chrF (character F-score) via sacrebleu.

    chrF is often more appropriate than BLEU for morphologically rich
    languages (Yoruba, Swahili, Amharic) because it operates at the
    character level and does not penalise morphological variants as
    heavily as word-level n-gram metrics.

    Args:
        char_order: Character n-gram order (default 6).
        word_order: Word n-gram order (default 0 for chrF,
            set to 2 for chrF++).
        beta: F-score beta parameter (default 2; recall-weighted).

    References:
        Popović (2015). chrF: character n-gram F-score for automatic MT
        evaluation. WMT.
    """

    name = "chrf"
    description = "chrF character F-score (sacrebleu). Range: [0, 100]."
    higher_is_better = True

    def __init__(
        self,
        char_order: int = 6,
        word_order: int = 0,
        beta: float = 2.0,
    ) -> None:
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required: pip install sacrebleu")

        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) must have the same length."
            )

        corpus_score = sacrebleu.corpus_chrf(
            predictions,
            [references],
            char_order=self.char_order,
            word_order=self.word_order,
            beta=self.beta,
        )
        sentence_scores = [
            sacrebleu.sentence_chrf(pred, [ref]).score
            for pred, ref in zip(predictions, references)
        ]

        return MetricResult(
            name=self.name,
            score=round(corpus_score.score, 2),
            details={
                "sentence_scores": [round(s, 2) for s in sentence_scores],
            },
            metadata={
                "n_samples": len(predictions),
                "char_order": self.char_order,
                "word_order": self.word_order,
                "beta": self.beta,
            },
        )
