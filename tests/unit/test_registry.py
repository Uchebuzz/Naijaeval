"""Tests for the plugin registry system."""

import pytest

import naijaeval.metrics  # noqa: F401 — trigger registrations
import naijaeval.datasets  # noqa: F401
from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import (
    DatasetRegistry,
    MetricRegistry,
    get_metric,
    list_datasets,
    list_metrics,
    register_dataset,
    register_metric,
)


class TestMetricRegistry:
    def test_all_core_metrics_registered(self):
        names = list_metrics()
        expected = {
            "code_switch_rate",
            "dialectal_robustness_score",
            "terminology_preservation_rate",
            "bleu",
            "chrf",
            "wer",
            "cer",
            "wer_delta",
            "hallucination_rate",
            "consistency_score",
        }
        assert expected.issubset(set(names))

    def test_get_metric_returns_instance(self):
        metric = get_metric("bleu")
        assert isinstance(metric, BaseMetric)

    def test_get_unknown_metric_raises(self):
        with pytest.raises(KeyError, match="not found"):
            MetricRegistry.get("nonexistent_metric_xyz")

    def test_register_custom_metric(self):
        @register_metric("__test_custom_metric__")
        class TestMetric(BaseMetric):
            name = "__test_custom_metric__"

            def compute(self, predictions, references, **kwargs):
                return MetricResult(name=self.name, score=1.0)

        assert "__test_custom_metric__" in list_metrics()
        instance = get_metric("__test_custom_metric__")
        result = instance.compute([], [])
        assert result.score == 1.0

        # Clean up
        del MetricRegistry._registry["__test_custom_metric__"]

    def test_duplicate_registration_raises(self):
        @register_metric("__test_dup__")
        class M1(BaseMetric):
            name = "__test_dup__"
            def compute(self, p, r, **kw): ...

        with pytest.raises(ValueError, match="already registered"):
            @register_metric("__test_dup__")
            class M2(BaseMetric):
                name = "__test_dup__"
                def compute(self, p, r, **kw): ...

        del MetricRegistry._registry["__test_dup__"]


class TestDatasetRegistry:
    def test_core_datasets_registered(self):
        names = list_datasets()
        expected = {"menyo20k", "fleurs_yo", "fleurs_ha", "fleurs_sw", "naija_mt_sample"}
        assert expected.issubset(set(names))

    def test_register_custom_dataset(self):
        @register_dataset("__test_dataset__")
        def my_loader(split="test"):
            return [{"source": "hello", "target": "world"}]

        assert "__test_dataset__" in list_datasets()
        data = DatasetRegistry.get("__test_dataset__")()
        assert len(data) == 1

        del DatasetRegistry._registry["__test_dataset__"]

    def test_get_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="not found"):
            DatasetRegistry.get("__nonexistent_dataset__")
