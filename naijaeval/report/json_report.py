"""JSON report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naijaeval.metrics.base import MetricResult


def to_json(
    results: dict[str, MetricResult],
    model: str = "unknown",
    benchmark: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Serialise evaluation results to a JSON string.

    Args:
        results: Dict of metric_name → MetricResult.
        model: Model identifier string.
        benchmark: Benchmark name.
        metadata: Optional extra metadata to include in the report.

    Returns:
        Pretty-printed JSON string.
    """
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "model": model,
        "results": {name: result.to_dict() for name, result in results.items()},
        "summary": {name: round(result.score, 4) for name, result in results.items()},
    }
    if metadata:
        payload["metadata"] = metadata

    return json.dumps(payload, indent=2, ensure_ascii=False)


def save_json(
    results: dict[str, MetricResult],
    path: str | Path,
    model: str = "unknown",
    benchmark: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save evaluation results to a JSON file.

    Returns:
        The resolved output path.
    """
    out = Path(path)
    out.write_text(to_json(results, model=model, benchmark=benchmark, metadata=metadata), encoding="utf-8")
    return out
