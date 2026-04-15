# NaijaEval

**Evaluation infrastructure for AI systems that mainstream benchmarks can't assess — built for African languages, code-switching, and dialectal robustness.**

[![CI](https://github.com/Uchebuzz/naijaeval/actions/workflows/ci.yml/badge.svg)](https://github.com/Uchebuzz/naijaeval/actions)
[![PyPI](https://img.shields.io/pypi/v/naijaeval.svg)](https://pypi.org/project/naijaeval/)
[![Python](https://img.shields.io/pypi/pyversions/naijaeval.svg)](https://pypi.org/project/naijaeval/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Why this exists

Standard NLP benchmarks — GLUE, HELM, XTREME — were built for high-resource languages and standard dialects. When you build a system for Nigerian English, Yoruba, Igbo, Hausa, Nigerian Pidgin, or Swahili, none of those benchmarks tell you whether your system actually works.

The specific gaps NaijaEval addresses:

- **No metric exists for code-switch robustness.** A model that scores 0.85 on clean English may collapse when a user switches mid-sentence from English to Yoruba.
- **No standard way to measure dialectal degradation.** WER on standard British English says nothing about WER on Nigerian English.
- **Terminology preservation is unmeasured.** BLEU doesn't weight medical or legal terms differently from "the" — but in practice, getting "hypertension" wrong matters more than getting word order slightly wrong.
- **Hallucination in low-resource translation is invisible.** When a model is undertrained on Swahili, it hallucinates. Standard metrics don't flag this.

NaijaEval provides composable, task-agnostic metrics that work on real African language evaluation challenges — out of the box.

---

## Quickstart

```bash
pip install naijaeval
```

```python
from naijaeval.metrics import (
    CodeSwitchRateMetric,
    TerminologyPreservationMetric,
    HallucinationRateMetric,
    WERMetric,
)

# Measure how mixed your test data is
csr = CodeSwitchRateMetric()
result = csr.compute(
    predictions=["I dey go market abeg, wetin be the price?"],
    references=[],
)
print(f"Code-switch rate: {result.score:.3f}")
# Code-switch rate: 0.444

# Check terminology preservation in medical translation
tpr = TerminologyPreservationMetric(domain="medical")
result = tpr.compute(
    predictions=["Alaisan naa ni malaria ati hypertension."],
    references=[],
)
print(f"Term preservation: {result.score:.3f}")
# Term preservation: 0.150  (most terms not preserved → low Yoruba coverage)

# Detect hallucination in summarisation
hal = HallucinationRateMetric()
result = hal.compute(
    predictions=["The Lagos General Hospital in Kano treated 500 patients."],
    references=["The hospital in Lagos treated patients."],  # source
)
print(f"Hallucination rate: {result.score:.3f}")
print(f"Hallucinated: {result.details['per_sample'][0]['hallucinated']}")
```

---

## Supported tasks and benchmarks

| Benchmark | Task | Languages | Dataset |
|---|---|---|---|
| `naija_mt_v1` | Machine translation | English → Yoruba | MENYO-20k |
| `coswitch_asr_v1` | ASR robustness | Nigerian English / Pidgin | Common Voice |

---

## Supported metrics

| Metric | Category | Description |
|---|---|---|
| `code_switch_rate` | Robustness | Fraction of token pairs that switch language |
| `dialectal_robustness_score` | Robustness | Relative performance drop on dialectal vs standard input |
| `terminology_preservation_rate` | Fidelity | Fraction of domain terms present in output |
| `bleu` | Fidelity | Corpus BLEU (sacrebleu) |
| `chrf` | Fidelity | Character F-score — better for morphologically rich languages |
| `wer` | ASR | Word Error Rate |
| `cer` | ASR | Character Error Rate |
| `wer_delta` | ASR | WER degradation from standard to dialectal input |
| `hallucination_rate` | Consistency | Entity-based hallucination detection |
| `consistency_score` | Consistency | N-gram faithfulness to source |

### Built-in domain term lists

`medical` · `legal` · `financial` · `customer_support`

### Built-in language vocabularies

Yoruba (`yo`) · Igbo (`ig`) · Hausa (`ha`) · Nigerian Pidgin (`pcm`) · Swahili (`sw`) · Zulu (`zu`) · Amharic (`am`)

---

## CLI reference

```bash
# List everything available
naijaeval list metrics
naijaeval list datasets
naijaeval list benchmarks

# Run a benchmark
naijaeval run \
    --benchmark naija_mt_v1 \
    --predictions preds.txt \
    --references refs.txt \
    --model Helsinki-NLP/opus-mt-en-yo \
    --output results.json

# Compare two models
naijaeval compare model_a.json model_b.json

# Generate HTML report
naijaeval report --input results.json --output report.html
```

---

## Python API

```python
# Run a full task evaluation
from naijaeval.tasks.translation import TranslationTask

task = TranslationTask(domain="medical")
results = task.evaluate(
    predictions=my_translations,
    references=reference_translations,
    sources=english_sentences,
)
for name, result in results.items():
    print(f"{name}: {result.score:.4f}")

# Compare ASR performance on standard vs dialectal input
from naijaeval.tasks.asr import ASRTask

task = ASRTask()
results = task.evaluate(
    predictions=standard_preds,
    references=standard_refs,
    dialectal_predictions=dialectal_preds,
    dialectal_references=dialectal_refs,
    dialect_name="Nigerian English",
)
print(results["wer_delta"].details["interpretation"])
```

---

## Extending the toolkit

**Register a custom metric:**

```python
from naijaeval import register_metric
from naijaeval.metrics.base import BaseMetric, MetricResult

@register_metric("my_custom_score")
class MyCustomScore(BaseMetric):
    name = "my_custom_score"
    description = "My domain-specific evaluation metric."
    higher_is_better = True

    def compute(self, predictions, references, **kwargs):
        score = ...  # your implementation
        return MetricResult(name=self.name, score=score)
```

**Register a custom dataset:**

```python
from naijaeval import register_dataset

@register_dataset("my_corpus")
def load_my_corpus(split="test", **kwargs):
    # Return an iterable of {"source": ..., "target": ...} dicts
    ...
```

See [docs/contributing/adding_metrics.md](docs/contributing/adding_metrics.md) for the full contribution guide.

---

## Roadmap

**v0.1 (current)**
- 10 core metrics across 4 categories
- 2 benchmarks (naija_mt_v1, coswitch_asr_v1)
- 5 dataset loaders (MENYO-20k, FLEURS ×3, sample)
- CLI and HTML reports
- Plugin system

**v0.2 (planned)**
- COMET and BERTScore integration
- NLI-based hallucination detection (upgrade from heuristic)
- Conversational AI task
- Swahili and Igbo translation benchmarks
- Interactive Colab notebook

**v0.3 (planned)**
- Leaderboard integration
- AfricaNLP workshop benchmark track

---

## Citation

If you use NaijaEval in your research, please cite:

```bibtex
@software{buzugbe2026naijaeval,
  author    = {Buzugbe, Uche},
  title     = {{NaijaEval}: Evaluation toolkit for AI systems in African language contexts},
  year      = {2026},
  url       = {https://github.com/Uchebuzz/naijaeval},
  version   = {0.1.0},
}
```

---

## Contributing

Contributions are welcomed and encouraged. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add metrics, datasets, and benchmarks.

The fastest way to make a meaningful contribution is to:
1. Add a new metric (see `naijaeval/metrics/` for examples)
2. Add a dataset loader for an underrepresented African language
3. Run your own models against existing benchmarks and submit results

---

## Community

- [GitHub Discussions](https://github.com/Uchebuzz/naijaeval/discussions) — questions, ideas, benchmark results
- [AfricaNLP Workshop](https://africanlp.github.io/) — the primary research community this toolkit serves
- [Masakhane](https://www.masakhane.io/) — African NLP community

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

*Because good models deserve honest benchmarks.*
