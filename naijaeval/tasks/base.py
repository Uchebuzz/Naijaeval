"""Abstract base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from naijaeval.metrics.base import MetricResult


class BaseTask(ABC):
    """Abstract base for all NaijaEval evaluation tasks.

    A Task ties together:
    - A collection of metrics to run
    - A way to load predictions (from a model callable or pre-computed list)
    - A way to load references (from a dataset or explicit list)

    Subclasses implement :meth:`evaluate` and optionally :meth:`run_model`.
    """

    #: Task identifier used in the CLI and benchmark YAML.
    name: str = ""

    #: Human-readable description.
    description: str = ""

    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, MetricResult]:
        """Run all task metrics over ``predictions`` and ``references``.

        Args:
            predictions: Model outputs.
            references: Ground-truth targets.
            sources: Source texts (required for consistency/hallucination).
            **kwargs: Passed through to individual metrics.

        Returns:
            Dict mapping metric name → MetricResult.
        """
        ...

    def summary(self, results: dict[str, MetricResult]) -> dict[str, float]:
        """Extract scalar scores from a results dict for quick comparison."""
        return {name: result.score for name, result in results.items()}
