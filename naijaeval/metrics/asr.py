"""ASR metrics: WER, CER, and dialectal WER degradation.

These metrics evaluate Automatic Speech Recognition systems, with
specific attention to dialectal and accent-driven performance gaps.

Metrics
-------
- :class:`WERMetric` — Word Error Rate via jiwer.
- :class:`CERMetric` — Character Error Rate via jiwer.
- :class:`WERDeltaMetric` — WER degradation from standard to dialectal
  input; the primary metric for accent robustness evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import register_metric


def _require_jiwer() -> Any:
    try:
        import jiwer
        return jiwer
    except ImportError:
        raise ImportError(
            "jiwer is required for ASR metrics: pip install jiwer"
        )


@register_metric("wer")
class WERMetric(BaseMetric):
    """Word Error Rate (WER) for ASR evaluation.

    WER = (S + D + I) / N

    where S = substitutions, D = deletions, I = insertions,
    N = number of reference words.

    Lower is better.  0.0 = perfect transcription.

    Uses jiwer's ``process_words`` under the hood so that the
    transformation pipeline (e.g. lowercase, punctuation removal)
    is fully configurable.

    Args:
        transforms: A jiwer ``Compose`` transform pipeline applied
            to both hypotheses and references before scoring.  If
            ``None``, applies a sensible default (lowercase, strip
            punctuation, collapse whitespace).

    References:
        Morris et al. (2004). From WER and RIL to MER and WIL:
        improved evaluation measures for connected speech recognition.
        INTERSPEECH.
    """

    name = "wer"
    description = "Word Error Rate for ASR. Range: [0, ∞); lower is better."
    higher_is_better = False

    def __init__(self, transforms: Any = None) -> None:
        self.transforms = transforms

    def _get_transforms(self, jiwer: Any) -> Any:
        if self.transforms is not None:
            return self.transforms
        return jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute WER.

        Args:
            predictions: ASR hypotheses (transcriptions).
            references: Ground-truth transcriptions.

        Returns:
            MetricResult with:
            - ``score``: corpus-level WER
            - ``details["per_sample_wer"]``: per-utterance WER
            - ``details["mean_wer"]``: mean of per-sample WERs
        """
        jiwer = _require_jiwer()
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) must have the same length."
            )
        if not predictions:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        transforms = self._get_transforms(jiwer)

        corpus_wer = jiwer.wer(
            references,
            predictions,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )

        per_sample = [
            jiwer.wer(
                ref,
                pred,
                reference_transform=transforms,
                hypothesis_transform=transforms,
            )
            for pred, ref in zip(predictions, references)
        ]

        output = jiwer.process_words(
            references,
            predictions,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )

        return MetricResult(
            name=self.name,
            score=round(corpus_wer, 4),
            details={
                "per_sample_wer": [round(w, 4) for w in per_sample],
                "mean_wer": round(float(np.mean(per_sample)), 4),
                "substitutions": output.substitutions,
                "deletions": output.deletions,
                "insertions": output.insertions,
            },
            metadata={
                "n_samples": len(predictions),
            },
        )


@register_metric("cer")
class CERMetric(BaseMetric):
    """Character Error Rate (CER) for ASR evaluation.

    CER = (S + D + I) / N  at the character level.

    More appropriate than WER for agglutinative languages where a single
    word boundary error inflates the word-level error rate disproportionately.
    Lower is better.

    Args:
        transforms: jiwer transform pipeline applied before scoring.
    """

    name = "cer"
    description = "Character Error Rate for ASR. Range: [0, ∞); lower is better."
    higher_is_better = False

    def __init__(self, transforms: Any = None) -> None:
        self.transforms = transforms

    def _get_transforms(self, jiwer: Any) -> Any:
        if self.transforms is not None:
            return self.transforms
        return jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ])

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        jiwer = _require_jiwer()
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) must have the same length."
            )
        if not predictions:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        transforms = self._get_transforms(jiwer)

        corpus_cer = jiwer.cer(
            references,
            predictions,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )

        per_sample = [
            jiwer.cer(
                ref,
                pred,
                reference_transform=transforms,
                hypothesis_transform=transforms,
            )
            for pred, ref in zip(predictions, references)
        ]

        return MetricResult(
            name=self.name,
            score=round(corpus_cer, 4),
            details={
                "per_sample_cer": [round(c, 4) for c in per_sample],
                "mean_cer": round(float(np.mean(per_sample)), 4),
            },
            metadata={"n_samples": len(predictions)},
        )


@register_metric("wer_delta")
class WERDeltaMetric(BaseMetric):
    """WER degradation from standard to dialectal/accented input.

    **Definition**

    Given WER scores on a standard (neutral accent) test set and WER
    scores on a dialectal/accented test set, this metric computes:

        WER_delta = mean_dialectal_WER - mean_standard_WER

    Positive delta = model performs **worse** on the dialectal set.
    Negative delta = model performs **better** (unusual).

    Also reports the relative WER increase:

        relative_increase = WER_delta / mean_standard_WER * 100 (%)

    **Usage**

    WERDeltaMetric operates on pre-computed WER score lists, not raw text.
    Use :meth:`from_scores` as the primary API::

        from naijaeval.metrics.asr import WERDeltaMetric, WERMetric

        wer = WERMetric()
        std_wers = [wer.compute([p], [r]).score
                    for p, r in zip(standard_preds, standard_refs)]
        dia_wers = [wer.compute([p], [r]).score
                    for p, r in zip(dialectal_preds, dialectal_refs)]

        delta_metric = WERDeltaMetric()
        result = delta_metric.from_scores(std_wers, dia_wers, dialect="Nigerian English")
        print(result.score)                    # WER delta (positive = degradation)
        print(result.details["relative_pct"])  # % relative WER increase
    """

    name = "wer_delta"
    description = (
        "WER degradation on dialectal vs. standard input. "
        "Positive = model performs worse on dialect."
    )
    higher_is_better = False

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        raise NotImplementedError(
            "WERDeltaMetric does not operate on raw text. "
            "Use WERDeltaMetric().from_scores(standard_wers, dialectal_wers)."
        )

    def from_scores(
        self,
        standard_wers: list[float],
        dialectal_wers: list[float],
        dialect: str = "dialect",
    ) -> MetricResult:
        """Compute WER delta from two aligned WER score lists.

        Args:
            standard_wers: Per-utterance WER scores on standard-accent input.
            dialectal_wers: Per-utterance WER scores on dialectal input.
                Must have the same length as ``standard_wers``.
            dialect: Label for the dialect (used in metadata).

        Returns:
            MetricResult with ``score = mean_dialectal_WER - mean_standard_WER``.
        """
        if len(standard_wers) != len(dialectal_wers):
            raise ValueError(
                f"standard_wers ({len(standard_wers)}) and dialectal_wers "
                f"({len(dialectal_wers)}) must have the same length."
            )
        if not standard_wers:
            return MetricResult(name=self.name, score=0.0, metadata={"n_samples": 0})

        mean_std = float(np.mean(standard_wers))
        mean_dia = float(np.mean(dialectal_wers))
        delta = mean_dia - mean_std

        relative_pct = (delta / mean_std * 100) if mean_std > 0 else 0.0

        per_sample_deltas = [
            round(d - s, 4) for s, d in zip(standard_wers, dialectal_wers)
        ]

        return MetricResult(
            name=self.name,
            score=round(delta, 4),
            details={
                "mean_standard_wer": round(mean_std, 4),
                "mean_dialectal_wer": round(mean_dia, 4),
                "absolute_delta": round(delta, 4),
                "relative_pct": round(relative_pct, 2),
                "per_sample_deltas": per_sample_deltas,
                "interpretation": (
                    f"{abs(relative_pct):.1f}% WER {'increase' if delta > 0 else 'decrease'} "
                    f"on {dialect} vs. standard"
                ),
            },
            metadata={
                "n_samples": len(standard_wers),
                "dialect": dialect,
            },
        )
