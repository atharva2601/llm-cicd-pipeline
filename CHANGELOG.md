# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- pgvector integration for production retriever
- Vertex AI judge upgrade path (high-RPM tier)
- Grafana dashboard for drift metrics

---

## [1.0.0] — Initial Release

### Added

**Core pipeline**
- `src/mock_rag.py` — MockRAGPipeline with 8-topic in-memory KB
- `src/app.py` — CLI demo entry point
- `src/metrics.py` — 6 reusable custom GEval metrics (tone, completeness, policy, etc.)
- `src/synthesizer.py` — KBSynthesizer wrapping DeepEval Synthesizer

**Test suites** (all support `smoke` / `full` pytest markers)
- `tests/test_faithfulness.py` — Hallucination detection (FaithfulnessMetric ≥ 0.80)
- `tests/test_relevancy.py` — Answer relevancy (AnswerRelevancyMetric ≥ 0.70)
- `tests/test_contextual.py` — Contextual Precision & Recall (retriever quality)
- `tests/test_rag.py` — Combined RAG Triad
- `tests/test_safety.py` — Adversarial red-team tests (GEval Safety ≥ 0.80)
- `tests/test_custom_metrics.py` — Domain-specific quality tests
- `tests/test_golden_dataset.py` — Dataset-driven tests from golden_dataset.json
- `tests/test_integration.py` — End-to-end pipeline integration tests

**Scripts**
- `scripts/setup_check.py` — Pre-flight environment validator
- `scripts/synthesize_dataset.py` — Auto-generate golden dataset via Gemini
- `scripts/run_eval.py` — Local runner with score export
- `scripts/drift_detection.py` — Metric drift logger & analyser
- `scripts/validate_kb.py` — KB coverage validator
- `scripts/tune_thresholds.py` — Statistical threshold optimiser
- `scripts/benchmark.py` — Latency & throughput benchmarker
- `scripts/export_report.py` — Export drift log to Markdown / CSV / HTML

**CI/CD**
- `.github/workflows/llm_ci.yml` — 6-job evaluation pipeline
  - Smoke tests on every push
  - Full + Safety + Contextual on PRs to main
  - Nightly drift detection
  - HTML report upload as artifact
  - Evaluation Gate status check (for branch protection)
- `.github/workflows/lint.yml` — Parallel linting (black + ruff)
- `.github/pull_request_template.md` — PR checklist

**Infrastructure**
- `Dockerfile` — Reproducible container environment
- `docker-compose.yml` — Local dev convenience commands
- `data/golden_dataset.json` — 13 hand-curated test cases (10 standard + 3 adversarial)
- `Makefile` — All convenience commands

**Documentation**
- `README.md` — Full setup and usage guide
- `CONTRIBUTING.md` — Development workflow and KB extension guide
- `.env.example` — Environment variable template

### Design decisions
- Gemini Flash chosen as judge for free-tier cost and speed (PDF §3.2)
- `temperature=0.0` on judge for deterministic verdicts (PDF §4.2.2)
- `async_mode=False` to respect free-tier RPM limits (PDF §6.3)
- Mock RAG architecture avoids vector DB cost in CI (PDF §5)
- Smoke/Full marker split preserves GitHub Actions free minutes (PDF §8.3)
