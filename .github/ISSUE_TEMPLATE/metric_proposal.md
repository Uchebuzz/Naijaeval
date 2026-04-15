---
name: Metric proposal
about: Propose a new evaluation metric
title: "[METRIC] "
labels: new metric, help wanted
assignees: ''
---

**Metric name**
What should this metric be called? (snake_case)

**What does it measure?**
Describe what aspect of model performance this metric evaluates.

**Why does it matter for African language / code-switching evaluation?**
Explain the specific gap this addresses. Why can't existing metrics handle it?

**Formal definition**
If possible, provide a mathematical definition or pseudocode.

**Reference**
Link to a paper, prior implementation, or dataset that motivates this metric.

**Implementation notes**
Are there specific libraries, models, or data sources required?
What are the computational requirements?

**Example inputs/outputs**
```python
# What would using this metric look like?
metric = MyNewMetric()
result = metric.compute(
    predictions=["..."],
    references=["..."],
)
# result.score ≈ ?
```

**Are you willing to implement this?**
- [ ] Yes, I can implement and test this metric
- [ ] I can help with testing/review
- [ ] I'm proposing only — happy for someone else to implement
