"""Abstract base class and result dataclass for all NaijaEval metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Standardised container for a single metric computation.

    Attributes:
        name: The metric identifier (e.g. ``"code_switch_rate"``).
        score: Primary scalar score. Semantics depend on the metric;
            see ``higher_is_better`` on the producing metric class.
        details: Per-sample or per-term breakdowns.  Arbitrary dict.
        metadata: Extra information about the computation (e.g. number
            of samples evaluated, library version used).
    """

    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MetricResult(name={self.name!r}, score={self.score:.4f})"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
            "metadata": self.metadata,
        }


class BaseMetric(ABC):
    """Abstract base class all NaijaEval metrics must inherit from.

    Subclasses must set :attr:`name` and implement :meth:`compute`.

    Example::

        from naijaeval.metrics.base import BaseMetric, MetricResult
        from naijaeval import register_metric

        @register_metric("my_score")
        class MyScore(BaseMetric):
            name = "my_score"
            description = "A custom scoring metric."
            higher_is_better = True

            def compute(self, predictions, references, **kwargs):
                score = sum(p == r for p, r in zip(predictions, references))
                score /= len(predictions)
                return MetricResult(name=self.name, score=score)
    """

    #: Unique identifier used in the registry and output reports.
    name: str = ""

    #: Human-readable description shown in ``naijaeval list metrics``.
    description: str = ""

    #: Whether a higher score is better (True) or lower is better (False).
    higher_is_better: bool = True

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric over a batch of predictions and references.

        Args:
            predictions: Model outputs, one string per sample.
            references: Ground-truth strings, aligned with ``predictions``.
            **kwargs: Metric-specific keyword arguments.

        Returns:
            A :class:`MetricResult` with ``name`` matching ``self.name``.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
