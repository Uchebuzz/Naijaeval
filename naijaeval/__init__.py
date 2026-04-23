"""
NaijaEval: Evaluation toolkit for AI systems in African language contexts.

Provides composable, task-agnostic metrics for robustness, consistency,
and accuracy across ASR, machine translation, summarisation, and
conversational AI — with a focus on African languages, Nigerian English,
code-switching, and dialectal variation.
"""

from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import (
    DatasetRegistry,
    MetricRegistry,
    get_dataset,
    get_metric,
    list_datasets,
    list_metrics,
    register_dataset,
    register_metric,
)

__version__ = "0.1.0"
__author__ = "Uche Buzugbe"
__license__ = "Apache-2.0"

__all__ = [
    "__version__",
    "BaseMetric",
    "MetricResult",
    "MetricRegistry",
    "DatasetRegistry",
    "get_metric",
    "list_metrics",
    "register_metric",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
