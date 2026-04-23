"""ASR evaluation task."""

from __future__ import annotations

from typing import Any

from naijaeval.metrics.asr import CERMetric, WERMetric
from naijaeval.metrics.base import MetricResult
from naijaeval.metrics.robustness import CodeSwitchRateMetric
from naijaeval.tasks.base import BaseTask


class ASRTask(BaseTask):
    """Evaluate an ASR system, including dialectal robustness.

    Runs WER and CER by default.  If ``dialectal_predictions`` and
    ``dialectal_references`` are passed to :meth:`evaluate`, also
    computes WER delta (accent robustness score).

    Example::

        from naijaeval.tasks.asr import ASRTask

        task = ASRTask()
        results = task.evaluate(
            predictions=["the man is going to market"],
            references=["the man is going to the market"],
        )
        print(results["wer"].score)
    """

    name = "asr"
    description = "Evaluate an ASR system on standard and/or dialectal input."

    def __init__(self) -> None:
        self._wer = WERMetric()
        self._cer = CERMetric()
        self._csr = CodeSwitchRateMetric()

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        sources: list[str] | None = None,
        *,
        dialectal_predictions: list[str] | None = None,
        dialectal_references: list[str] | None = None,
        dialect_name: str = "dialect",
        compute_code_switch_rate: bool = False,
        **kwargs: Any,
    ) -> dict[str, MetricResult]:
        """Evaluate ASR predictions.

        Args:
            predictions: ASR hypotheses on standard input.
            references: Ground-truth transcriptions for standard input.
            dialectal_predictions: Hypotheses on dialectal/accented input.
            dialectal_references: Ground truths for dialectal input.
            dialect_name: Label for the dialect (used in output metadata).
            compute_code_switch_rate: If True, also computes CSR on
                ``references`` to characterise how mixed the test set is.
        """
        results: dict[str, MetricResult] = {}

        results["wer"] = self._wer.compute(predictions, references)
        results["cer"] = self._cer.compute(predictions, references)

        if compute_code_switch_rate:
            results["code_switch_rate"] = self._csr.compute(references, [])

        if dialectal_predictions is not None and dialectal_references is not None:
            from naijaeval.metrics.asr import WERDeltaMetric

            dia_wer_result = self._wer.compute(
                dialectal_predictions, dialectal_references
            )
            results[f"wer_{dialect_name}"] = dia_wer_result

            std_per_sample = results["wer"].details.get("per_sample_wer", [])
            dia_per_sample = dia_wer_result.details.get("per_sample_wer", [])

            if std_per_sample and dia_per_sample:
                delta_metric = WERDeltaMetric()
                results["wer_delta"] = delta_metric.from_scores(
                    std_per_sample,
                    dia_per_sample,
                    dialect=dialect_name,
                )

        return results
