"""
NaijaEval Example: Evaluating a translation model on Nigerian English → Yoruba.

This script demonstrates how to:
1. Load test data from the built-in sample corpus
2. Evaluate a translation model using multiple metrics
3. Compute dialectal robustness by comparing two model variants
4. Generate a JSON and HTML report

Usage:
    python examples/translation_eval.py

Requirements:
    pip install naijaeval sacrebleu
"""

import json

# ---------------------------------------------------------------------------
# Step 1: Import what we need
# ---------------------------------------------------------------------------
from naijaeval.metrics.fidelity import BLEUMetric, TerminologyPreservationMetric, chrFMetric
from naijaeval.metrics.consistency import HallucinationRateMetric, ConsistencyScoreMetric
from naijaeval.metrics.robustness import DialectalRobustnessMetric
from naijaeval.registry import get_dataset
from naijaeval.report.json_report import to_json

# ---------------------------------------------------------------------------
# Step 2: Load sample data
# ---------------------------------------------------------------------------
print("Loading sample data...")
import naijaeval.datasets  # noqa — triggers loader registration

samples = get_dataset("naija_mt_sample")
sources = [s["source"] for s in samples]
references = [s["target"] for s in samples]

print(f"Loaded {len(samples)} sentence pairs\n")

# ---------------------------------------------------------------------------
# Step 3: Simulate model predictions
# (In a real evaluation you'd call your model here)
# ---------------------------------------------------------------------------
# For this demo, we use the reference translations as "perfect" predictions
# and a degraded version as a second model
perfect_predictions = references[:]
degraded_predictions = [ref + " extra hallucinated content from Lagos General Hospital" for ref in references]

# ---------------------------------------------------------------------------
# Step 4: Run metrics on the "perfect" model
# ---------------------------------------------------------------------------
print("=== Model A (reference-quality) ===")

bleu = BLEUMetric()
chrf = chrFMetric()
tpr = TerminologyPreservationMetric(domain="medical")
hal = HallucinationRateMetric()
con = ConsistencyScoreMetric()

results_a = {
    "bleu": bleu.compute(perfect_predictions, references),
    "chrf": chrf.compute(perfect_predictions, references),
    "terminology_preservation_rate": tpr.compute(perfect_predictions, references),
    "hallucination_rate": hal.compute(perfect_predictions, sources),
    "consistency_score": con.compute(perfect_predictions, sources),
}

for name, result in results_a.items():
    print(f"  {name:35s}: {result.score:.4f}")

# ---------------------------------------------------------------------------
# Step 5: Run metrics on the "degraded" model
# ---------------------------------------------------------------------------
print("\n=== Model B (with hallucinations) ===")

results_b = {
    "bleu": bleu.compute(degraded_predictions, references),
    "chrf": chrf.compute(degraded_predictions, references),
    "terminology_preservation_rate": tpr.compute(degraded_predictions, references),
    "hallucination_rate": hal.compute(degraded_predictions, sources),
    "consistency_score": con.compute(degraded_predictions, sources),
}

for name, result in results_b.items():
    print(f"  {name:35s}: {result.score:.4f}")

# ---------------------------------------------------------------------------
# Step 6: Dialectal robustness comparison
# ---------------------------------------------------------------------------
print("\n=== Dialectal Robustness (BLEU: A vs B) ===")

drs_metric = DialectalRobustnessMetric()
bleu_a_scores = results_a["bleu"].details["sentence_scores"]
bleu_b_scores = results_b["bleu"].details["sentence_scores"]

drs_result = drs_metric.from_scores(
    baseline_scores=bleu_a_scores,
    dialectal_scores=bleu_b_scores,
    metric_name="BLEU",
    dialect_name="Hallucinated outputs",
)
print(f"  DRS:              {drs_result.details['drs']:.4f}")
print(f"  Absolute delta:   {drs_result.details['absolute_delta']:.4f}")
print(f"  Relative change:  {drs_result.details['relative_change_pct']:.1f}%")

# ---------------------------------------------------------------------------
# Step 7: Generate JSON report
# ---------------------------------------------------------------------------
report_json = to_json(results_a, model="model_a_reference", benchmark="naija_mt_demo")
print(f"\n=== JSON Report (first 500 chars) ===")
print(report_json[:500] + "...\n")

# Optionally save to file:
# from naijaeval.report.json_report import save_json
# save_json(results_a, "results_a.json", model="model_a", benchmark="naija_mt_demo")

print("Done. To generate an HTML report:")
print("  naijaeval report --input results_a.json --output report.html")
