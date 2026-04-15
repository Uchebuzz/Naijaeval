"""
NaijaEval Example: ASR robustness evaluation on Nigerian English vs Pidgin.

This script demonstrates how to:
1. Evaluate an ASR system on standard Nigerian English
2. Evaluate the same system on code-switched Pidgin/English
3. Compute the WER delta (accent robustness score)
4. Characterise how mixed the test set is using CodeSwitchRateMetric

Usage:
    python examples/asr_robustness.py

Requirements:
    pip install naijaeval jiwer
"""

from naijaeval.metrics.asr import WERDeltaMetric, WERMetric
from naijaeval.metrics.robustness import CodeSwitchRateMetric
from naijaeval.tasks.asr import ASRTask

# ---------------------------------------------------------------------------
# Simulated test data
# ---------------------------------------------------------------------------

# Standard Nigerian English transcriptions and hypotheses
standard_references = [
    "the man went to buy food at the market",
    "please bring me the medicine for my mother",
    "we need to register the children at school",
    "the hospital is closed on sunday",
    "how much does this cost",
]
standard_predictions = [
    "the man went to buy food at the market",        # perfect
    "please bring me the medicine for my mother",   # perfect
    "we need to register the children at school",   # perfect
    "the hospital is closed on sunday",             # perfect
    "how much does this cost",                       # perfect
]

# Code-switched Nigerian Pidgin / English
dialectal_references = [
    "I dey go market abeg give me food",
    "wetin be the price for dis medicine",
    "make we go register the pikin for school",
    "wahala dey for the hospital dem don close",
    "how much e cost abi na free",
]
dialectal_predictions = [
    "i day go market a beg give me food",       # slight errors
    "what in be the price for this medicine",   # heavier errors
    "make we go register the pikin for school", # perfect
    "wahala di for the hospital dem don close", # slight
    "how much it cost abby na free",            # substitutions
]

# ---------------------------------------------------------------------------
# Step 1: Characterise the test sets
# ---------------------------------------------------------------------------
print("=== Test Set Characterisation ===")

csr = CodeSwitchRateMetric()

std_csr = csr.compute(standard_references, [])
dia_csr = csr.compute(dialectal_references, [])

print(f"  Standard CSR:   {std_csr.score:.4f} (low = mostly monolingual)")
print(f"  Dialectal CSR:  {dia_csr.score:.4f} (higher = more code-switching)")
print(f"  Detected languages in dialectal set: {dia_csr.details['detected_languages']}\n")

# ---------------------------------------------------------------------------
# Step 2: Run the ASR task evaluation
# ---------------------------------------------------------------------------
print("=== ASR Evaluation ===")

task = ASRTask()
results = task.evaluate(
    predictions=standard_predictions,
    references=standard_references,
    dialectal_predictions=dialectal_predictions,
    dialectal_references=dialectal_references,
    dialect_name="Nigerian Pidgin",
    compute_code_switch_rate=True,
)

print(f"  Standard WER:     {results['wer'].score:.4f}")
print(f"  Standard CER:     {results['cer'].score:.4f}")
print(f"  Dialectal WER:    {results['wer_Nigerian Pidgin'].score:.4f}")
print(f"  WER Delta:        {results['wer_delta'].score:+.4f}")
print(f"  {results['wer_delta'].details['interpretation']}\n")

# ---------------------------------------------------------------------------
# Step 3: Detailed WER delta breakdown
# ---------------------------------------------------------------------------
print("=== Per-Utterance WER Delta ===")

std_per = results["wer"].details["per_sample_wer"]
dia_per = results["wer_Nigerian Pidgin"].details["per_sample_wer"]

for i, (s, d, std_ref, dia_ref) in enumerate(
    zip(std_per, dia_per, standard_references, dialectal_references)
):
    delta = d - s
    print(f"  [{i+1}] std={s:.3f}  dia={d:.3f}  delta={delta:+.3f}")
    print(f"      std: {std_ref[:50]}")
    print(f"      dia: {dia_ref[:50]}")

print("\nDone.")
