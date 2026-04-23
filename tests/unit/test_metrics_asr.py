"""Tests for ASR metrics: WERMetric, CERMetric, WERDeltaMetric."""

import pytest

from naijaeval.metrics.asr import CERMetric, WERDeltaMetric, WERMetric


class TestWERMetric:
    def setup_method(self):
        self.metric = WERMetric()

    def test_perfect_transcription_is_zero(self):
        result = self.metric.compute(
            ["the man went to the market"],
            ["the man went to the market"],
        )
        assert result.score == pytest.approx(0.0)

    def test_completely_wrong_is_high(self):
        result = self.metric.compute(
            ["aaa bbb ccc ddd"],
            ["the man went to the market"],
        )
        assert result.score > 0.5

    def test_score_is_non_negative(self):
        result = self.metric.compute(
            ["the man went"],
            ["the man went to the market"],
        )
        assert result.score >= 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute(["a"], ["b", "c"])

    def test_empty_input(self):
        result = self.metric.compute([], [])
        assert result.score == 0.0
        assert result.metadata["n_samples"] == 0

    def test_per_sample_wer_present(self):
        result = self.metric.compute(
            ["hello world", "test case"],
            ["hello world", "test case"],
        )
        assert "per_sample_wer" in result.details
        assert len(result.details["per_sample_wer"]) == 2

    def test_result_name(self):
        result = self.metric.compute(["hello"], ["hello"])
        assert result.name == "wer"

    def test_substitution_counted(self):
        # "cat" substituted for "dog"
        result = self.metric.compute(
            ["the cat sat on the mat"],
            ["the dog sat on the mat"],
        )
        assert result.details["substitutions"] >= 1

    def test_batch_multiple_samples(self):
        result = self.metric.compute(
            ["perfect match", "wrong words"],
            ["perfect match", "the man went"],
        )
        assert result.score >= 0.0
        assert len(result.details["per_sample_wer"]) == 2


class TestCERMetric:
    def setup_method(self):
        self.metric = CERMetric()

    def test_perfect_is_zero(self):
        result = self.metric.compute(["hello world"], ["hello world"])
        assert result.score == pytest.approx(0.0)

    def test_score_is_non_negative(self):
        result = self.metric.compute(["hell world"], ["hello world"])
        assert result.score >= 0.0

    def test_result_name(self):
        result = self.metric.compute(["hello"], ["hello"])
        assert result.name == "cer"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute(["a"], ["b", "c"])

    def test_empty_input(self):
        result = self.metric.compute([], [])
        assert result.score == 0.0

    def test_per_sample_cer_present(self):
        result = self.metric.compute(["hello", "world"], ["hello", "world"])
        assert "per_sample_cer" in result.details
        assert len(result.details["per_sample_cer"]) == 2


class TestWERDeltaMetric:
    def setup_method(self):
        self.metric = WERDeltaMetric()

    def test_compute_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.metric.compute([], [])

    def test_no_delta_when_equal(self):
        scores = [0.2, 0.3, 0.25]
        result = self.metric.from_scores(scores, scores)
        assert result.score == pytest.approx(0.0)

    def test_positive_delta_on_degradation(self):
        standard = [0.1, 0.12, 0.11]
        dialectal = [0.25, 0.28, 0.26]
        result = self.metric.from_scores(standard, dialectal)
        assert result.score > 0.0

    def test_delta_computed_correctly(self):
        import numpy as np

        std = [0.10, 0.20]
        dia = [0.30, 0.40]
        result = self.metric.from_scores(std, dia)
        expected = float(np.mean(dia) - np.mean(std))
        assert result.score == pytest.approx(expected, abs=1e-4)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.from_scores([0.1, 0.2], [0.3])

    def test_empty_scores(self):
        result = self.metric.from_scores([], [])
        assert result.metadata["n_samples"] == 0

    def test_result_name(self):
        result = self.metric.from_scores([0.1], [0.2])
        assert result.name == "wer_delta"

    def test_relative_pct_correct(self):
        std = [0.20]
        dia = [0.30]
        result = self.metric.from_scores(std, dia)
        # relative_pct = (0.30 - 0.20) / 0.20 * 100 = 50%
        assert result.details["relative_pct"] == pytest.approx(50.0, abs=0.1)

    def test_metadata_dialect_name(self):
        result = self.metric.from_scores([0.1], [0.2], dialect="Yoruba English")
        assert result.metadata["dialect"] == "Yoruba English"
