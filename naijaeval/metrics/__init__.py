"""Metrics subpackage — imports all metrics so they self-register."""

from naijaeval.metrics.asr import CERMetric, WERDeltaMetric, WERMetric
from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.metrics.consistency import (
    ConsistencyScoreMetric,
    HallucinationRateMetric,
)
from naijaeval.metrics.fidelity import (
    BLEUMetric,
    TerminologyPreservationMetric,
    chrFMetric,
)
from naijaeval.metrics.robustness import CodeSwitchRateMetric, DialectalRobustnessMetric

__all__ = [
    "BaseMetric",
    "MetricResult",
    "CodeSwitchRateMetric",
    "DialectalRobustnessMetric",
    "TerminologyPreservationMetric",
    "BLEUMetric",
    "chrFMetric",
    "WERMetric",
    "CERMetric",
    "WERDeltaMetric",
    "HallucinationRateMetric",
    "ConsistencyScoreMetric",
]
