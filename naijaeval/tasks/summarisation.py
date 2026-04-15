"""Summarisation evaluation task."""

from __future__ import annotations

from typing import Any

from naijaeval.metrics.base import MetricResult
from naijaeval.metrics.consistency import ConsistencyScoreMetric, HallucinationRateMetric
from naijaeval.metrics.fidelity import TerminologyPreservationMetric
from naijaeval.tasks.base import BaseTask


class SummarisationTask(BaseTask):
    """Evaluate a summarisation system for faithfulness and term preservation.

    Unlike translation evaluation, summarisation does not use BLEU/chrF
    as primary metrics (they are poorly calibrated for abstractive
    summarisation).  Instead the default metrics focus on:

    - Hallucination rate (entity-based)
    - Consistency score (n-gram faithfulness to source)
    - Terminology preservation (for domain-critical applications)

    Args:
        domain: Domain for terminology preservation (e.g. ``"medical"``).
        custom_terms: Custom term dictionary.
    """

    name = "summarisation"
    description = "Evaluate a summarisation system for faithfulness and term coverage."

    def __init__(
        self,
        domain: str | None = None,
        custom_terms: dict[str, list[str]] | None = None,
    ) -> None:
        self.domain = domain
        self.custom_terms = custom_terms
        self._hallucination = HallucinationRateMetric()
        self._consistency = ConsistencyScoreMetric()
        self._term = (
            TerminologyPreservationMetric(domain=domain, terms=custom_terms)
            if (domain or custom_terms)
            else None
        )

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, MetricResult]:
        """Evaluate summarisation.

        Args:
            predictions: Generated summaries.
            references: Human reference summaries (used for consistency
                if ``sources`` is not provided).
            sources: Original documents.  If provided, hallucination rate
                and consistency are computed against the source rather than
                the reference.
        """
        effective_source = sources if sources is not None else references
        results: dict[str, MetricResult] = {}

        results["hallucination_rate"] = self._hallucination.compute(
            predictions, effective_source
        )
        results["consistency_score"] = self._consistency.compute(
            predictions, effective_source
        )

        if self._term is not None:
            results["terminology_preservation_rate"] = self._term.compute(
                predictions, effective_source
            )

        return results
