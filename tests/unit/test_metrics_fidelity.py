"""Tests for fidelity metrics: TerminologyPreservationMetric, BLEUMetric, chrFMetric."""

import pytest

from naijaeval.metrics.fidelity import (
    BLEUMetric,
    TerminologyPreservationMetric,
    chrFMetric,
)


class TestTerminologyPreservationMetric:
    def setup_method(self):
        self.terms = {
            "malaria": ["malaria"],
            "hypertension": ["hypertension", "high blood pressure"],
            "tuberculosis": ["tuberculosis", "TB"],
        }
        self.metric = TerminologyPreservationMetric(terms=self.terms)

    def test_perfect_preservation(self):
        pred = "The patient has malaria, hypertension, and tuberculosis."
        result = self.metric.compute([pred], [])
        assert result.score == pytest.approx(1.0)

    def test_zero_preservation(self):
        pred = "The weather is nice today."
        result = self.metric.compute([pred], [])
        assert result.score == pytest.approx(0.0)

    def test_partial_preservation(self):
        pred = "The patient has malaria."  # only 1 of 3 terms
        result = self.metric.compute([pred], [])
        assert result.score == pytest.approx(1 / 3, abs=0.01)

    def test_synonym_counts(self):
        pred = "The patient has high blood pressure."  # synonym for hypertension
        metric = TerminologyPreservationMetric(terms=self.terms)
        result = metric.compute([pred], [])
        # Should match hypertension via "high blood pressure"
        assert result.details["per_term_recall"]["hypertension"] == pytest.approx(1.0)

    def test_case_insensitive(self):
        pred = "MALARIA and TUBERCULOSIS were diagnosed."
        result = self.metric.compute([pred], [])
        assert result.details["per_term_recall"]["malaria"] == pytest.approx(1.0)
        assert result.details["per_term_recall"]["tuberculosis"] == pytest.approx(1.0)

    def test_empty_predictions(self):
        result = self.metric.compute([], [])
        assert result.metadata["n_samples"] == 0

    def test_domain_medical_loads(self):
        metric = TerminologyPreservationMetric(domain="medical")
        pred = "The patient has malaria and hypertension."
        result = metric.compute([pred], [])
        assert 0.0 <= result.score <= 1.0

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            TerminologyPreservationMetric(domain="alchemy").compute(["text"], [])

    def test_no_terms_raises(self):
        metric = TerminologyPreservationMetric()
        with pytest.raises(ValueError, match="No term list"):
            metric.compute(["text"], [])

    def test_compute_time_domain_override(self):
        metric = TerminologyPreservationMetric()
        result = metric.compute(
            ["The patient has malaria."],
            [],
            domain="medical",
        )
        assert 0.0 <= result.score <= 1.0

    def test_missing_terms_reported(self):
        pred = "The weather is nice."
        result = self.metric.compute([pred], [])
        assert "missing_terms" in result.details
        assert len(result.details["missing_terms"]) == 3

    def test_per_sample_scores_length(self):
        preds = ["malaria present", "patient healthy", "TB confirmed"]
        result = self.metric.compute(preds, [])
        assert len(result.details["per_sample_rates"]) == 3

    def test_batch_mean_is_correct(self):
        preds = [
            "malaria and hypertension and tuberculosis",  # 3/3 = 1.0
            "malaria only",  # 1/3
            "nothing",  # 0/3
        ]
        result = self.metric.compute(preds, [])
        expected = (1.0 + 1 / 3 + 0.0) / 3
        assert result.score == pytest.approx(expected, abs=0.001)


class TestBLEUMetric:
    def setup_method(self):
        self.metric = BLEUMetric()

    def test_identical_gives_high_score(self):
        sent = "the cat sat on the mat"
        result = self.metric.compute([sent], [sent])
        assert result.score > 90.0  # BLEU scale is 0-100

    def test_totally_different_gives_low_score(self):
        result = self.metric.compute(
            ["the quick brown fox"],
            ["completely different words here"],
        )
        assert result.score < 10.0

    def test_score_range(self):
        result = self.metric.compute(
            ["hello world how are you"],
            ["hello world i am fine"],
        )
        assert 0.0 <= result.score <= 100.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute(["a", "b"], ["c"])

    def test_metadata_present(self):
        result = self.metric.compute(["hello"], ["hello"])
        assert "sacrebleu_version" in result.metadata
        assert result.metadata["n_samples"] == 1

    def test_sentence_scores_present(self):
        result = self.metric.compute(["hello world"], ["hello world"])
        assert "sentence_scores" in result.details
        assert len(result.details["sentence_scores"]) == 1

    def test_bp_present(self):
        result = self.metric.compute(["hello world test"], ["hello world test longer"])
        assert "bp" in result.details

    def test_result_name(self):
        result = self.metric.compute(["hello"], ["hello"])
        assert result.name == "bleu"


class TestchrFMetric:
    def setup_method(self):
        self.metric = chrFMetric()

    def test_identical_gives_high_score(self):
        sent = "the cat sat on the mat"
        result = self.metric.compute([sent], [sent])
        assert result.score > 90.0

    def test_score_range(self):
        result = self.metric.compute(
            ["Mo n lo si oja"],
            ["Mo fẹ lo si oja"],
        )
        assert 0.0 <= result.score <= 100.0

    def test_result_name(self):
        result = self.metric.compute(["hello"], ["hello"])
        assert result.name == "chrf"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute(["a"], ["b", "c"])

    def test_sentence_scores_present(self):
        result = self.metric.compute(["hello"], ["hello"])
        assert "sentence_scores" in result.details
