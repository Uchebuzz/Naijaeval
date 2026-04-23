"""Translation evaluation task."""

from __future__ import annotations

from typing import Any

from naijaeval.metrics.base import MetricResult
from naijaeval.metrics.consistency import (
    ConsistencyScoreMetric,
    HallucinationRateMetric,
)
from naijaeval.metrics.fidelity import (
    BLEUMetric,
    TerminologyPreservationMetric,
    chrFMetric,
)
from naijaeval.tasks.base import BaseTask


class TranslationTask(BaseTask):
    """Evaluate a machine translation system.

    Runs BLEU, chrF, and (optionally) terminology preservation and
    hallucination rate by default.

    Args:
        domain: If provided, runs :class:`~naijaeval.metrics.fidelity.TerminologyPreservationMetric`
            with the built-in term list for this domain.
        custom_terms: Custom term dictionary for terminology preservation.
        run_consistency: If True (default), also runs
            :class:`~naijaeval.metrics.consistency.ConsistencyScoreMetric`.
        run_hallucination: If True (default), also runs
            :class:`~naijaeval.metrics.consistency.HallucinationRateMetric`
            using the source sentences as the reference.

    Example::

        from naijaeval.tasks.translation import TranslationTask

        task = TranslationTask(domain="medical")
        results = task.evaluate(
            predictions=["Ẹ jẹ ki a jẹ oúnjẹ"],
            references=["Let us eat food"],
            sources=["Let us eat food"],
        )
        for name, result in results.items():
            print(f"{name}: {result.score}")
    """

    name = "translation"
    description = "Evaluate a machine translation system end-to-end."

    def __init__(
        self,
        domain: str | None = None,
        custom_terms: dict[str, list[str]] | None = None,
        run_consistency: bool = True,
        run_hallucination: bool = True,
    ) -> None:
        self.domain = domain
        self.custom_terms = custom_terms
        self.run_consistency = run_consistency
        self.run_hallucination = run_hallucination

        self._bleu = BLEUMetric()
        self._chrf = chrFMetric()
        self._term = (
            TerminologyPreservationMetric(domain=domain, terms=custom_terms)
            if (domain or custom_terms)
            else None
        )
        self._consistency = ConsistencyScoreMetric() if run_consistency else None
        self._hallucination = HallucinationRateMetric() if run_hallucination else None

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, MetricResult]:
        results: dict[str, MetricResult] = {}

        results["bleu"] = self._bleu.compute(predictions, references)
        results["chrf"] = self._chrf.compute(predictions, references)

        if self._term is not None:
            results["terminology_preservation_rate"] = self._term.compute(
                predictions, references
            )

        if self._consistency is not None and sources is not None:
            results["consistency_score"] = self._consistency.compute(
                predictions, sources
            )

        if self._hallucination is not None and sources is not None:
            results["hallucination_rate"] = self._hallucination.compute(
                predictions, sources
            )

        return results
