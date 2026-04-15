"""Metrics subpackage — imports all metrics so they self-register."""

from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.metrics.robustness import CodeSwitchRateMetric, DialectalRobustnessMetric
from naijaeval.metrics.fidelity import (
    TerminologyPreservationMetric,
    BLEUMetric,
    chrFMetric,
)
from naijaeval.metrics.asr import WERMetric, CERMetric, WERDeltaMetric
from naijaeval.metrics.consistency import HallucinationRateMetric, ConsistencyScoreMetric

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
