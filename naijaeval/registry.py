"""
Central plugin registries for metrics and datasets.

Usage — registering a custom metric::

    from naijaeval import register_metric
    from naijaeval.metrics.base import BaseMetric, MetricResult

    @register_metric("my_score")
    class MyScore(BaseMetric):
        name = "my_score"

        def compute(self, predictions, references, **kwargs):
            score = ...
            return MetricResult(name=self.name, score=score)

Usage — registering a custom dataset::

    from naijaeval import register_dataset

    @register_dataset("my_corpus")
    def load_my_corpus(split="test", **kwargs):
        # return an iterable of {"source": ..., "target": ...} dicts
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from naijaeval.metrics.base import BaseMetric


class MetricRegistry:
    """Registry mapping metric names to BaseMetric subclasses."""

    _registry: dict[str, type[BaseMetric]] = {}

    @classmethod
    def register(cls, name: str, metric_cls: type[BaseMetric]) -> None:
        if name in cls._registry:
            raise ValueError(
                f"Metric '{name}' is already registered. "
                "Use a different name or unregister the existing one first."
            )
        cls._registry[name] = metric_cls

    @classmethod
    def get(cls, name: str) -> type[BaseMetric]:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Metric '{name}' not found. Available metrics: {available}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def all(cls) -> dict[str, type[BaseMetric]]:
        return dict(cls._registry)


class DatasetRegistry:
    """Registry mapping dataset names to loader callables."""

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, loader_fn: Callable[..., Any]) -> None:
        if name in cls._registry:
            raise ValueError(f"Dataset '{name}' is already registered.")
        cls._registry[name] = loader_fn

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())


def register_metric(name: str):
    """Class decorator to register a metric under ``name``."""

    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        MetricRegistry.register(name, cls)
        return cls

    return decorator


def register_dataset(name: str):
    """Function decorator to register a dataset loader under ``name``."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        DatasetRegistry.register(name, fn)
        return fn

    return decorator


def get_metric(name: str) -> BaseMetric:
    """Instantiate and return the metric registered under ``name``."""
    return MetricRegistry.get(name)()


def list_metrics() -> list[str]:
    """Return a sorted list of all registered metric names."""
    return MetricRegistry.list()


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Call the dataset loader registered under ``name``."""
    return DatasetRegistry.get(name)(**kwargs)


def list_datasets() -> list[str]:
    """Return a sorted list of all registered dataset names."""
    return DatasetRegistry.list()
