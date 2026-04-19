"""Tests for consistency metrics: HallucinationRateMetric, ConsistencyScoreMetric."""

import pytest

from naijaeval.metrics.consistency import (
    ConsistencyScoreMetric,
    HallucinationRateMetric,
)


class TestHallucinationRateMetric:
    def setup_method(self):
        self.metric = HallucinationRateMetric()

    def test_all_entities_supported(self):
        source = "Dr. Okafor treated patients at Lagos General Hospital last Tuesday."
        pred = "Patients at Lagos General Hospital were treated by Dr. Okafor."
        result = self.metric.compute([pred], [source])
        # All capitalized terms in pred appear in source → rate should be low
        assert result.score < 0.5

    def test_hallucinated_entity_detected(self):
        source = "The hospital in Lagos treated patients."
        pred = "The Lagos General Hospital treated 500 patients in Kano."
        result = self.metric.compute([pred], [source])
        # "Kano" and "500" and possibly "General" are not in source
        assert result.score > 0.0

    def test_no_entities_is_zero(self):
        source = "the weather is nice today"
        pred = "it was a good day"  # no capitalized content
        result = self.metric.compute([pred], [source])
        assert result.score == pytest.approx(0.0)

    def test_empty_input(self):
        result = self.metric.compute([], [])
        assert result.score == 0.0
        assert result.metadata["n_samples"] == 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute(["pred"], ["src1", "src2"])

    def test_result_name(self):
        result = self.metric.compute(["Hello World"], ["Hello World test"])
        assert result.name == "hallucination_rate"

    def test_number_detected_as_candidate(self):
        source = "There were 100 patients."
        pred = "There were 500 patients."
        result = self.metric.compute([pred], [source])
        # "500" is not in source ("100" is)
        assert result.score > 0.0

    def test_per_sample_details_present(self):
        result = self.metric.compute(
            ["Hello World", "Another Test"],
            ["Hello there World", "Another Test case"],
        )
        assert "per_sample" in result.details
        assert len(result.details["per_sample"]) == 2

    def test_score_in_valid_range(self):
        sources = ["Lagos hospital report", "Nigerian government statement"]
        preds = ["Lagos Kano Abuja report from Nairobi", "statement from government"]
        result = self.metric.compute(preds, sources)
        assert 0.0 <= result.score <= 1.0

    def test_top_hallucinated_terms_present(self):
        source = "Patient visited the clinic."
        pred = "Patient visited Lagos General Hospital in Abuja."
        result = self.metric.compute([pred], [source])
        assert "top_hallucinated_terms" in result.details

    def test_metadata_method(self):
        result = self.metric.compute(["Hello"], ["Hello World"])
        assert result.metadata["method"] == "capitalized_ngram_heuristic_v1"


class TestConsistencyScoreMetric:
    def setup_method(self):
        self.metric = ConsistencyScoreMetric()

    def test_identical_is_one(self):
        text = "the patient has malaria and needs treatment"
        result = self.metric.compute([text], [text])
        assert result.score == pytest.approx(1.0)

    def test_no_overlap_is_low(self):
        pred = "the quick brown fox jumps over"
        source = "completely different words here today"
        result = self.metric.compute([pred], [source])
        assert result.score < 0.5

    def test_partial_overlap(self):
        pred = "the patient has malaria"
        source = "the patient was diagnosed with malaria and fever"
        result = self.metric.compute([pred], [source])
        assert 0.0 < result.score <= 1.0

    def test_empty_input(self):
        result = self.metric.compute([], [])
        assert result.score == 0.0

    def test_result_name(self):
        result = self.metric.compute(["hello world"], ["hello world"])
        assert result.name == "consistency_score"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute(["a", "b"], ["c"])

    def test_unigram_mode(self):
        metric = ConsistencyScoreMetric(ngram_order=1)
        pred = "malaria treatment hospital"
        source = "the hospital treated malaria with medication"
        result = metric.compute([pred], [source])
        assert result.score > 0.0

    def test_per_sample_scores_present(self):
        result = self.metric.compute(
            ["hello world", "test case"],
            ["hello world test", "test case study"],
        )
        assert "per_sample_scores" in result.details
        assert len(result.details["per_sample_scores"]) == 2

    def test_score_in_valid_range(self):
        preds = ["the man went to the market yesterday", "rain was falling hard"]
        sources = ["the man visited the market", "heavy rainfall was reported"]
        result = self.metric.compute(preds, sources)
        assert 0.0 <= result.score <= 1.0

    def test_metadata_ngram_order(self):
        metric = ConsistencyScoreMetric(ngram_order=3)
        result = metric.compute(["hello world test"], ["hello world test"])
        assert result.metadata["ngram_order"] == 3
