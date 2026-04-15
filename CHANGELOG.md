# Changelog

All notable changes to NaijaEval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] - 2026-04-15

### Added
- Core metric framework with `BaseMetric` and `MetricResult`
- **Robustness metrics**: `CodeSwitchRateMetric`, `DialectalRobustnessMetric`
- **Fidelity metrics**: `TerminologyPreservationMetric`, `BLEUMetric`, `chrFMetric`
- **ASR metrics**: `WERMetric`, `CERMetric`, `WERDeltaMetric`
- **Consistency metrics**: `HallucinationRateMetric`, `ConsistencyScoreMetric`
- Task framework with `BaseTask`, `TranslationTask`, `ASRTask`, `SummarisationTask`
- Dataset registry and loaders for MENYO-20k and FLEURS
- Benchmark definitions: `naija_mt_v1`, `coswitch_asr_v1`
- CLI: `naijaeval run`, `naijaeval compare`, `naijaeval list`, `naijaeval report`
- Plugin/registry system for custom metrics and datasets
- Built-in domain term lists: medical, legal, financial, customer support
- Built-in vocabulary lists for Yoruba, Igbo, Hausa, Nigerian Pidgin, Swahili, Zulu
- HTML and JSON report generation
