# Contributing to NaijaEval

Thank you for your interest in contributing. NaijaEval is built by and for the African NLP community — every contribution, from a new metric to a corrected word list, directly improves how AI systems are evaluated for hundreds of millions of speakers.

---

## How to contribute

### 1. Find something to work on

- Browse [open issues](https://github.com/Uchebuzz/naijaeval/issues) — look for `good first issue` and `help wanted` labels
- Propose a new metric via the [metric proposal template](.github/ISSUE_TEMPLATE/metric_proposal.md)
- Propose a new dataset via the [dataset proposal template](.github/ISSUE_TEMPLATE/dataset_proposal.md)
- Fix a bug — check the `bug` label

If you're unsure, open a discussion first.

### 2. Fork and set up

```bash
git clone https://github.com/<your-username>/naijaeval.git
cd naijaeval
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install
```

### 3. Make your changes

- Create a branch: `git checkout -b feature/my-metric`
- Write your code
- Write tests in `tests/unit/`
- Run the full test suite: `pytest tests/unit/ -v`
- Format: `black naijaeval/ tests/` and `ruff check naijaeval/`

### 4. Submit a pull request

Push your branch and open a PR against `main`. Use the PR template.

---

## Adding a new metric

Every metric must:

1. **Inherit from `BaseMetric`** in `naijaeval/metrics/base.py`
2. **Set `name`, `description`, and `higher_is_better`** as class attributes
3. **Implement `compute(self, predictions, references, **kwargs)`** returning a `MetricResult`
4. **Register itself** using `@register_metric("metric_name")` from `naijaeval.registry`
5. **Have a docstring** that includes:
   - Formal definition (formula or algorithm)
   - Explanation of why it matters for African language evaluation
   - `Example::` usage block
6. **Have tests** in `tests/unit/test_metrics_<category>.py` with at least:
   - A test for the expected case
   - A test for edge cases (empty input, mismatched lengths)
   - A test that `result.name` matches the registered name

**Example skeleton:**

```python
from naijaeval.metrics.base import BaseMetric, MetricResult
from naijaeval.registry import register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    name = "my_metric"
    description = "One-line description shown in 'naijaeval list metrics'."
    higher_is_better = True  # or False

    def compute(self, predictions, references, **kwargs):
        # Your implementation
        score = ...
        return MetricResult(
            name=self.name,
            score=score,
            details={"per_sample": [...]},
            metadata={"n_samples": len(predictions)},
        )
```

Then import your metric in `naijaeval/metrics/__init__.py` to ensure it registers.

---

## Adding a new dataset loader

1. Add a function to `naijaeval/datasets/loaders.py` decorated with `@register_dataset("name")`
2. The function must accept a `split` argument and return a HuggingFace `Dataset` or a list of dicts with at minimum `"source"` and `"target"` keys
3. Include a docstring with: dataset name, citation, license, column description
4. Add the name to the list in `tests/unit/test_registry.py`

---

## Adding a benchmark definition

1. Create a new YAML file in `benchmarks/`
2. Follow the structure of `benchmarks/naija_mt_v1.yaml`
3. Required fields: `name`, `version`, `description`, `language_pair`, `dataset`, `metrics`
4. Optional: `expected_baselines`, `tags`

---

## Code style

- **Formatter:** `black` (88 char line length)
- **Linter:** `ruff`
- **Type hints:** required for all public functions
- **Docstrings:** Google style
- No emojis in code or docstrings

---

## What not to contribute (yet)

- Sub-word tokenisation — out of scope for v1
- Model fine-tuning utilities
- Data collection tools
- Leaderboard infrastructure (planned for v0.3)

---

## Questions?

Open a [GitHub Discussion](https://github.com/Uchebuzz/naijaeval/discussions) — we respond to all questions.
