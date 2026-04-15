"""Consistency metrics: hallucination detection and factual consistency.

Hallucination is a critical failure mode in low-resource NLP: when a
model has seen little training data for a language pair or domain, it
tends to generate plausible-sounding but unsupported content.  These
metrics provide lightweight, model-free proxies that can be applied
without a dedicated NLI or fact-checking model.

Metrics
-------
- :class:`HallucinationRateMetric` — entity-based hallucination
  detection: flags content in the output that has no support in the source.
- :class:`ConsistencyScoreMetric` — n-gram overlap-based consistency
  between prediction and source (suitable as a quick faithfulness proxy
  before deploying heavier NLI-based methods).

**Design note on v1 approach**

Both metrics use surface-form heuristics rather than neural models.
This is intentional for v1: zero external model dependencies, instant
inference, and interpretable scores.  The documentation and roadmap
explicitly describe a v2 upgrade path to NLI-based verification.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import register_metric
from naijaeval.utils.text import (
    extract_capitalized_ngrams,
    extract_ngrams,
    normalize_text,
    tokenize,
)


@register_metric("hallucination_rate")
class HallucinationRateMetric(BaseMetric):
    """Entity-based hallucination rate for summarisation and translation.

    **Algorithm**

    For each (prediction, source) pair:

    1. Extract capitalised n-grams from the prediction (proxy for proper
       nouns / named entities — things that are factually specific and
       most likely to be hallucinated).
    2. Check whether each extracted n-gram appears anywhere in the source.
    3. Hallucination rate = fraction of prediction n-grams **not** found
       in the source.

    This is a precision-oriented metric: it measures how much of what
    the model asserted is unsupported, rather than how much of the source
    the model covered (that would be recall).

    **Limitations**

    - Capitalisation heuristics work well for English-script text but
      may miss hallucinations expressed in lowercase (e.g. dates,
      numbers, medical terms).
    - For languages with different capitalisation norms the ``max_ngram``
      parameter should be set to 1 and a custom entity extractor provided.

    A v2 upgrade using ``spaCy`` NER or a dedicated NLI model is planned.

    Args:
        max_ngram: Maximum n-gram length to extract from predictions
            as candidate entities (default 3 — captures "Lagos State
            Hospital" as a single entity).

    Example::

        from naijaeval.metrics import HallucinationRateMetric

        metric = HallucinationRateMetric()
        result = metric.compute(
            predictions=["The Lagos General Hospital treated 500 patients in Kano."],
            references=["The hospital in Lagos treated patients."],
        )
        print(result.score)                      # e.g. 0.5
        print(result.details["hallucinated"])    # ['Kano', '500']
    """

    name = "hallucination_rate"
    description = (
        "Fraction of named entity n-grams in the prediction that are "
        "unsupported by the source. Range: [0, 1]; 0.0 = no hallucination."
    )
    higher_is_better = False

    def __init__(self, max_ngram: int = 3) -> None:
        self.max_ngram = max_ngram

    def _is_supported(self, ngram: str, source: str) -> bool:
        """Check if ``ngram`` appears (case-insensitively) in ``source``."""
        return ngram.lower() in source.lower()

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute hallucination rate.

        Args:
            predictions: Model outputs (summaries or translations).
            references: Source texts the predictions were derived from.
                **Note**: ``references`` here means the *source*, not a
                human reference translation.  For summarisation, this is
                the original document.  For translation, this is the
                source-language sentence.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references/sources "
                f"({len(references)}) must have the same length."
            )
        if not predictions:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        per_sample_rates: list[float] = []
        all_hallucinated: list[str] = []
        per_sample_details: list[dict[str, Any]] = []

        for pred, source in zip(predictions, references):
            pred_entities = extract_capitalized_ngrams(pred, max_n=self.max_ngram)
            # Also extract standalone numbers as potential hallucination targets
            numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", pred))
            candidates = pred_entities | numbers

            if not candidates:
                per_sample_rates.append(0.0)
                per_sample_details.append({
                    "n_candidates": 0,
                    "n_hallucinated": 0,
                    "rate": 0.0,
                    "hallucinated": [],
                    "supported": [],
                })
                continue

            hallucinated = [c for c in candidates if not self._is_supported(c, source)]
            supported = [c for c in candidates if self._is_supported(c, source)]
            rate = len(hallucinated) / len(candidates)

            per_sample_rates.append(rate)
            all_hallucinated.extend(hallucinated)
            per_sample_details.append({
                "n_candidates": len(candidates),
                "n_hallucinated": len(hallucinated),
                "rate": round(rate, 4),
                "hallucinated": sorted(hallucinated),
                "supported": sorted(supported),
            })

        mean_rate = float(np.mean(per_sample_rates))

        return MetricResult(
            name=self.name,
            score=round(mean_rate, 4),
            details={
                "per_sample": per_sample_details,
                "top_hallucinated_terms": _top_n(all_hallucinated, n=10),
            },
            metadata={
                "n_samples": len(predictions),
                "max_ngram": self.max_ngram,
                "method": "capitalized_ngram_heuristic_v1",
            },
        )


@register_metric("consistency_score")
class ConsistencyScoreMetric(BaseMetric):
    """N-gram recall-based factual consistency between prediction and source.

    **Definition**

    For each (prediction, source) pair:

        consistency = |ngrams(prediction) ∩ ngrams(source)| / |ngrams(prediction)|

    This measures what fraction of n-grams in the prediction can be
    traced back to the source — a lightweight faithfulness proxy.

    Unlike hallucination rate this metric is not entity-specific and
    applies uniformly to all tokens.  It rewards outputs that stay close
    to the source vocabulary and penalises paraphrasing, so it is most
    appropriate when literal fidelity is required (e.g. medical reports).

    Args:
        ngram_order: N-gram size for overlap computation (default 2
            for bigrams, which balances precision and recall better
            than unigrams alone).

    Example::

        from naijaeval.metrics import ConsistencyScoreMetric

        metric = ConsistencyScoreMetric(ngram_order=1)
        result = metric.compute(
            predictions=["The patient has malaria and should rest."],
            references=["The doctor confirmed malaria diagnosis for the patient."],
        )
        print(result.score)  # e.g. 0.714
    """

    name = "consistency_score"
    description = (
        "N-gram recall: fraction of prediction n-grams found in the source. "
        "Range: [0, 1]; higher = more faithful to source."
    )
    higher_is_better = True

    def __init__(self, ngram_order: int = 2) -> None:
        self.ngram_order = ngram_order

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute consistency score.

        Args:
            predictions: Model outputs.
            references: Source texts (documents, source-language sentences).
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references/sources "
                f"({len(references)}) must have the same length."
            )
        if not predictions:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        per_sample: list[float] = []

        for pred, source in zip(predictions, references):
            pred_tokens = tokenize(normalize_text(pred), lowercase=True)
            src_tokens = tokenize(normalize_text(source), lowercase=True)

            if not pred_tokens:
                per_sample.append(0.0)
                continue

            pred_ngrams = extract_ngrams(pred_tokens, self.ngram_order)
            src_ngrams = extract_ngrams(src_tokens, self.ngram_order)

            if not pred_ngrams:
                per_sample.append(0.0)
                continue

            overlap = sum(
                min(pred_ngrams[ng], src_ngrams[ng]) for ng in pred_ngrams if ng in src_ngrams
            )
            total_pred = sum(pred_ngrams.values())
            score = overlap / total_pred if total_pred > 0 else 0.0
            per_sample.append(score)

        mean_score = float(np.mean(per_sample))

        return MetricResult(
            name=self.name,
            score=round(mean_score, 4),
            details={
                "per_sample_scores": [round(s, 4) for s in per_sample],
            },
            metadata={
                "n_samples": len(predictions),
                "ngram_order": self.ngram_order,
            },
        )


def _top_n(items: list[str], n: int = 10) -> list[dict[str, Any]]:
    """Return the top-n most frequent items with their counts."""
    counts = Counter(items)
    return [
        {"term": term, "count": count}
        for term, count in counts.most_common(n)
    ]
