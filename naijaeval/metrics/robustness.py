"""Robustness metrics: code-switch detection and dialectal degradation.

These metrics address a fundamental gap in standard NLP evaluation:
most benchmarks test systems on clean, monolingual, standard-dialect
text.  Real deployment in African contexts involves heavy code-switching
and pronounced dialectal variation.  A system that scores well on
standard benchmarks may fail badly in the field.

Metrics
-------
- :class:`CodeSwitchRateMetric` — quantifies how aggressively a text
  mixes languages, used both to characterise test sets and as a
  conditioning variable when interpreting model scores.
- :class:`DialectalRobustnessMetric` — measures how much a model
  degrades when input moves from standard to dialectal/regional text.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naijaeval.data.vocabularies import DEFAULT_VOCAB
from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import register_metric
from naijaeval.utils.text import count_language_switches, segment_by_language, tokenize


@register_metric("code_switch_rate")
class CodeSwitchRateMetric(BaseMetric):
    """Measure the degree of code-switching in a collection of texts.

    **Definition**

    For a single text of *N* tokens, the code-switch rate is:

        CSR = number_of_language_transitions / (N - 1)

    A transition occurs when two consecutive tokens are assigned to
    different languages.  The metric is averaged across all texts in the
    batch.

    **Language identification**

    Token-level language identification is performed using vocabulary
    lookup against the built-in African language word lists (see
    :mod:`naijaeval.data.vocabularies`).  Tokens not found in any
    vocabulary are tagged as the ``default_lang`` (default: ``"en"``).
    This approach is intentionally conservative — it is biased toward
    low false-positive rates rather than high recall.

    Users working with languages not in the built-in vocabulary can pass
    a custom ``vocab`` dict of ``{lang_code: set_of_words}``.

    Args:
        vocab: Custom vocabulary mapping.  If ``None``, uses the
            built-in :data:`~naijaeval.data.vocabularies.DEFAULT_VOCAB`.
        default_lang: Language tag for tokens not found in any vocab.

    Example::

        from naijaeval.metrics import CodeSwitchRateMetric

        metric = CodeSwitchRateMetric()
        result = metric.compute(
            predictions=["I know mo pe the man ni serious"],
            references=[],   # not used by this metric
        )
        print(result.score)   # e.g. 0.375
        print(result.details) # per-sample rates and language breakdowns
    """

    name = "code_switch_rate"
    description = (
        "Fraction of consecutive token pairs that switch language. "
        "Ranges from 0 (monolingual) to 1 (every pair switches)."
    )
    higher_is_better = False  # higher rate = more mixing (not better or worse per se)

    def __init__(
        self,
        vocab: dict[str, set[str]] | None = None,
        default_lang: str = "en",
    ) -> None:
        self.vocab = vocab if vocab is not None else DEFAULT_VOCAB
        self.default_lang = default_lang

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute average code-switch rate across ``predictions``.

        ``references`` is accepted for API consistency but not used.
        """
        if not predictions:
            return MetricResult(
                name=self.name,
                score=0.0,
                metadata={"n_samples": 0},
            )

        per_sample: list[float] = []
        per_sample_details: list[dict[str, Any]] = []

        for text in predictions:
            tokens = tokenize(text)
            if len(tokens) < 2:
                per_sample.append(0.0)
                per_sample_details.append(
                    {"tokens": len(tokens), "switches": 0, "rate": 0.0}
                )
                continue

            labeled = segment_by_language(text, self.vocab, self.default_lang)
            switches = count_language_switches(labeled)
            rate = switches / (len(labeled) - 1)

            # Language distribution
            lang_counts: dict[str, int] = {}
            for _, lang in labeled:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            per_sample.append(rate)
            per_sample_details.append(
                {
                    "tokens": len(labeled),
                    "switches": switches,
                    "rate": round(rate, 4),
                    "language_distribution": {
                        k: round(v / len(labeled), 3) for k, v in lang_counts.items()
                    },
                }
            )

        avg_rate = float(np.mean(per_sample))

        # Identify dominant non-default languages across the batch
        all_non_default = [
            lang
            for d in per_sample_details
            for lang, prop in d.get("language_distribution", {}).items()
            if lang != self.default_lang and prop > 0
        ]
        detected_langs = sorted(set(all_non_default))

        return MetricResult(
            name=self.name,
            score=avg_rate,
            details={
                "per_sample": per_sample_details,
                "detected_languages": detected_langs,
            },
            metadata={
                "n_samples": len(predictions),
                "default_lang": self.default_lang,
                "vocab_languages": sorted(self.vocab.keys()),
            },
        )


@register_metric("dialectal_robustness_score")
class DialectalRobustnessMetric(BaseMetric):
    """Measure how much a model degrades on dialectal vs. standard input.

    **Definition**

    Given a model's scores on standard text (``baseline_scores``) and on
    dialectal/regional text (``dialectal_scores``), this metric computes:

        DRS = 1 - (mean_baseline - mean_dialectal) / mean_baseline

    A DRS of 1.0 means the model performs identically on dialect as on
    standard text (perfect robustness).  A DRS of 0.5 means performance
    halved.  A DRS above 1.0 means the model actually performs *better*
    on dialect (unusual but possible for models trained on dialectal data).

    The absolute delta is also reported for interpretability:

        delta = mean_dialectal - mean_baseline

    A negative delta indicates degradation; positive indicates improvement.

    **Usage**

    This metric does not operate on raw text.  Instead it consumes
    pre-computed per-sample scores from any other metric (e.g. WER,
    BLEU, classification accuracy).  Pass the scores as the first two
    elements of ``predictions`` and ``references`` respectively, or use
    the convenience class method :meth:`from_scores`.

    Example::

        from naijaeval.metrics import DialectalRobustnessMetric

        metric = DialectalRobustnessMetric()
        result = metric.from_scores(
            baseline_scores=[0.82, 0.79, 0.85, 0.80],
            dialectal_scores=[0.71, 0.68, 0.74, 0.70],
            metric_name="accuracy",
        )
        print(result.score)          # e.g. 0.127 (12.7% degradation expressed as delta)
        print(result.details["drs"]) # e.g. 0.856
    """

    name = "dialectal_robustness_score"
    description = (
        "Relative performance retention when input moves from standard "
        "to dialectal text. 1.0 = no degradation; lower = more degradation."
    )
    higher_is_better = True

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Not the primary API for this metric — use :meth:`from_scores`."""
        raise NotImplementedError(
            "DialectalRobustnessMetric does not operate on raw text. "
            "Use DialectalRobustnessMetric().from_scores(baseline_scores, "
            "dialectal_scores) instead."
        )

    def from_scores(
        self,
        baseline_scores: list[float],
        dialectal_scores: list[float],
        metric_name: str = "metric",
        dialect_name: str = "dialect",
    ) -> MetricResult:
        """Compute dialectal robustness from two aligned score lists.

        Args:
            baseline_scores: Per-sample scores on standard-dialect input.
            dialectal_scores: Per-sample scores on dialectal input.
                Must have the same length as ``baseline_scores``.
            metric_name: Label for the underlying metric (e.g. ``"WER"``).
            dialect_name: Label for the dialect being evaluated.

        Returns:
            :class:`~naijaeval.metrics.base.MetricResult` with:
            - ``score``: absolute delta (dialectal − baseline); negative = degradation
            - ``details["drs"]``: Dialectal Robustness Score (1 − relative drop)
            - ``details["mean_baseline"]``: mean of ``baseline_scores``
            - ``details["mean_dialectal"]``: mean of ``dialectal_scores``
            - ``details["relative_change_pct"]``: percentage change
        """
        if len(baseline_scores) != len(dialectal_scores):
            raise ValueError(
                f"baseline_scores ({len(baseline_scores)}) and dialectal_scores "
                f"({len(dialectal_scores)}) must have the same length."
            )
        if not baseline_scores:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        mean_base = float(np.mean(baseline_scores))
        mean_dial = float(np.mean(dialectal_scores))
        delta = mean_dial - mean_base

        if mean_base == 0.0:
            drs = 1.0 if mean_dial == 0.0 else float("inf")
            relative_pct = 0.0
        else:
            drs = 1.0 - ((mean_base - mean_dial) / mean_base)
            relative_pct = (delta / mean_base) * 100

        return MetricResult(
            name=self.name,
            score=round(delta, 4),
            details={
                "drs": round(drs, 4),
                "mean_baseline": round(mean_base, 4),
                "mean_dialectal": round(mean_dial, 4),
                "absolute_delta": round(delta, 4),
                "relative_change_pct": round(relative_pct, 2),
                "interpretation": (
                    "no degradation"
                    if abs(delta) < 0.01
                    else ("degradation" if delta < 0 else "improvement")
                ),
            },
            metadata={
                "n_samples": len(baseline_scores),
                "underlying_metric": metric_name,
                "dialect": dialect_name,
            },
        )
