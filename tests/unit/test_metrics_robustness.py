"""Tests for robustness metrics: CodeSwitchRateMetric, DialectalRobustnessMetric."""

import pytest

from naijaeval.metrics.robustness import CodeSwitchRateMetric, DialectalRobustnessMetric


class TestCodeSwitchRateMetric:
    def setup_method(self):
        self.metric = CodeSwitchRateMetric()

    def test_monolingual_english_is_zero(self):
        result = self.metric.compute(
            predictions=["The man went to the market and bought food"],
            references=[],
        )
        assert result.score == 0.0
        assert result.name == "code_switch_rate"

    def test_yoruba_words_detected(self):
        # 'mo' and 'pe' are in the Yoruba vocab
        result = self.metric.compute(
            predictions=["I know mo pe the man is serious"],
            references=[],
        )
        assert result.score > 0.0, "Should detect Yoruba tokens and count switches"

    def test_pidgin_words_detected(self):
        result = self.metric.compute(
            predictions=["I dey go market abeg make we chop"],
            references=[],
        )
        assert result.score > 0.0

    def test_empty_input(self):
        result = self.metric.compute(predictions=[], references=[])
        assert result.score == 0.0
        assert result.metadata["n_samples"] == 0

    def test_single_token_no_crash(self):
        result = self.metric.compute(predictions=["hello"], references=[])
        assert result.score == 0.0

    def test_score_in_valid_range(self):
        texts = [
            "I dey mo pe go market wetin wahala",
            "The government ni made a statement",
            "Pure English text without any mixing",
        ]
        result = self.metric.compute(predictions=texts, references=[])
        assert 0.0 <= result.score <= 1.0

    def test_per_sample_details_present(self):
        result = self.metric.compute(
            predictions=["hello world", "I dey go"],
            references=[],
        )
        assert "per_sample" in result.details
        assert len(result.details["per_sample"]) == 2

    def test_detected_languages_populated(self):
        result = self.metric.compute(
            predictions=["I know mo pe the man"],
            references=[],
        )
        assert "detected_languages" in result.details

    def test_custom_vocab(self):
        custom_vocab = {"xx": {"xylo", "zylo", "mylo"}}
        metric = CodeSwitchRateMetric(vocab=custom_vocab)
        result = metric.compute(
            predictions=["the xylo went to zylo the store"],
            references=[],
        )
        assert result.score > 0.0

    def test_batch_average(self):
        metric = CodeSwitchRateMetric()
        # Single high-mix text vs pure English
        result = metric.compute(
            predictions=[
                "mo pe ni ti the man is going",  # high mix
                "the quick brown fox jumps over",  # pure English
            ],
            references=[],
        )
        single_high = metric.compute(
            predictions=["mo pe ni ti the man is going"],
            references=[],
        )
        single_low = metric.compute(
            predictions=["the quick brown fox jumps over"],
            references=[],
        )
        expected_avg = (single_high.score + single_low.score) / 2
        assert abs(result.score - expected_avg) < 1e-6


class TestDialectalRobustnessMetric:
    def setup_method(self):
        self.metric = DialectalRobustnessMetric()

    def test_compute_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.metric.compute([], [])

    def test_no_degradation(self):
        scores = [0.8, 0.8, 0.8]
        result = self.metric.from_scores(scores, scores)
        assert result.score == 0.0
        assert result.details["drs"] == pytest.approx(1.0)

    def test_degradation_computed_correctly(self):
        baseline = [0.80, 0.82, 0.78]
        dialectal = [0.70, 0.72, 0.68]
        result = self.metric.from_scores(baseline, dialectal)
        # delta = mean([0.70, 0.72, 0.68]) - mean([0.80, 0.82, 0.78])
        import numpy as np
        expected_delta = float(np.mean(dialectal) - np.mean(baseline))
        assert result.score == pytest.approx(expected_delta, abs=1e-4)
        assert result.score < 0  # negative = degradation

    def test_drs_less_than_one_on_degradation(self):
        baseline = [0.9, 0.9]
        dialectal = [0.5, 0.5]
        result = self.metric.from_scores(baseline, dialectal)
        assert result.details["drs"] < 1.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.from_scores([0.8, 0.9], [0.7])

    def test_empty_scores(self):
        result = self.metric.from_scores([], [])
        assert result.metadata["n_samples"] == 0

    def test_result_name(self):
        result = self.metric.from_scores([0.8], [0.7])
        assert result.name == "dialectal_robustness_score"

    def test_metadata_contains_dialect(self):
        result = self.metric.from_scores([0.8], [0.7], dialect_name="Nigerian English")
        assert result.metadata["dialect"] == "Nigerian English"

    def test_relative_change_pct(self):
        baseline = [1.0]
        dialectal = [0.8]
        result = self.metric.from_scores(baseline, dialectal)
        assert result.details["relative_change_pct"] == pytest.approx(-20.0, abs=0.1)
