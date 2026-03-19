# 🧪 CI/CD LLM Evaluation Pipeline

> **Zero-cost, production-grade LLM regression testing** using DeepEval + Gemini 1.5 Flash + GitHub Actions.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Test Suites Explained](#test-suites-explained)
- [Running Tests Locally](#running-tests-locally)
- [CI/CD Setup](#cicd-setup)
- [Metrics Reference](#metrics-reference)
- [Customisation Guide](#customisation-guide)
- [Scaling to Production](#scaling-to-production)

---

## Overview

This project implements the **LLMOps** evaluation pattern described in the accompanying research document. It solves the **Evaluation Gap** — the chasm between a working LLM prototype and a production-ready system that is safe, reliable, and hallucination-free.

### Why this matters

| Traditional Software | LLM Software |
|---|---|
| Deterministic output | Probabilistic / stochastic output |
| `assert output == expected` | Semantic equivalence checks |
| Binary pass/fail | Scored metrics with thresholds |
| No judge required | LLM-as-a-Judge paradigm |

### The stack (all free)

| Component | Tool | Role |
|---|---|---|
| **Test Framework** | [DeepEval](https://deepeval.com) | Pytest-native LLM evaluation |
| **LLM Judge** | Gemini 1.5 Flash (Google AI Studio) | Semantic verdict engine |
| **Application Model** | Gemini 1.5 Flash | System under test |
| **CI/CD** | GitHub Actions | Orchestration & gate enforcement |
| **Retrieval** | Mock RAG (in-memory dict) | Deterministic retriever for unit tests |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI                       │
│                                                             │
│  push (any branch)  →  Smoke Tests  (3 queries, ~3 min)    │
│  PR to main         →  Full + Safety Tests (~15 min)       │
│  push to main       →  Full + Safety Tests                  │
│  nightly schedule   →  Full + Safety Tests (drift check)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │    MockRAGPipeline          │
              │  ┌──────────┐  ┌────────┐  │
              │  │ Retriever│→ │  LLM   │  │
              │  │ (dict)   │  │(Gemini)│  │
              │  └──────────┘  └────────┘  │
              └─────────────┬──────────────┘
                            │ (answer, context)
              ┌─────────────▼──────────────┐
              │       DeepEval Metrics      │
              │  FaithfulnessMetric  (0.80) │
              │  AnswerRelevancyMetric(0.70)│
              │  GEval Safety        (0.80) │
              └─────────────┬──────────────┘
                            │ Judge: Gemini Flash (temp=0)
              ┌─────────────▼──────────────┐
              │     Pass / Fail + Reason    │
              │     Blocks merge if fail    │
              └────────────────────────────┘
```

---

## Quick Start

### 1. Get a free API key

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) → **Create API Key** (no billing required).

### 2. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/llm-cicd-project.git
cd llm-cicd-project

cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY
```

### 3. Install + configure (one command)

```bash
make setup
```

### 4. Run smoke tests

```bash
make smoke
```

### 5. Run the CLI demo

```bash
make demo
# or: python -m src.app "How do I get a refund?"
```

### 6. Generate a synthetic golden dataset

```bash
make synthesize
```

---

## Project Structure

```
llm-cicd-project/
├── .env.example                  ← Copy to .env, add your API key
├── .gitignore
├── Makefile                      ← All convenience commands
├── pyproject.toml                ← Poetry config + pytest markers
├── requirements.txt              ← pip fallback
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── app.py                    ← CLI demo / production entry point
│   └── mock_rag.py               ← MockRAGPipeline (system under test)
│
├── data/
│   └── golden_dataset.json       ← Hand-curated ground-truth dataset (13 cases)
│
├── scripts/
│   ├── synthesize_dataset.py     ← Auto-generate golden dataset via Gemini
│   ├── run_eval.py               ← Local runner with score export
│   └── drift_detection.py        ← Metric drift logger & analyser
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               ← Shared fixtures: judge + pipeline
│   ├── test_faithfulness.py      ← Hallucination detection (smoke + full)
│   ├── test_relevancy.py         ← Answer relevancy (smoke + full)
│   ├── test_contextual.py        ← Contextual precision & recall (smoke + full)
│   ├── test_rag.py               ← Combined RAG triad (smoke + full)
│   └── test_safety.py            ← Adversarial / red-team tests
│
├── reports/                      ← Generated at runtime (gitignored)
│   ├── drift_log.jsonl           ← Append-only metric history
│   ├── latest_results.json       ← Last run's scores
│   └── evaluation_report.html   ← DeepEval HTML report
│
└── .github/
    └── workflows/
        └── llm_ci.yml            ← GitHub Actions (6-job pipeline)
```

---

## Test Suites Explained

### `test_faithfulness.py` — Hallucination Detection
Checks that every claim in the generated answer is supported by the retrieved context.

**Metric:** `FaithfulnessMetric(threshold=0.80)`

**Formula:** `Claims Supported by Context / Total Claims`

**Example failure:** Context says "Support hours: Mon–Fri 9–5 EST" but the LLM says "We're available 24/7." → Score: 0.0 → CI blocked.

---

### `test_relevancy.py` — Responsiveness Check
Checks that the answer actually addresses the user's question.

**Metric:** `AnswerRelevancyMetric(threshold=0.70)`

**Mechanism:** The judge generates synthetic questions from the answer, then computes cosine similarity with the original question.

**Example failure:** User asks "How do I get a refund?" and the LLM responds "Hello! How can I help you?" → Low similarity → Score near 0.0 → CI blocked.

---

### `test_rag.py` — Combined RAG Triad
Runs both Faithfulness and Relevancy in one `assert_test()` call. This is the **primary regression gate**.

---

### `test_safety.py` — Red-Team / Adversarial Tests
Uses `GEval` with a custom "Safety Officer" prompt to catch harmful outputs.

**Adversarial cases include:**
- "Add glue to pizza" (poisoned context test)
- Self-harm queries
- Money laundering queries
- Account hacking queries
- Dangerous chemical queries

Passing: LLM refuses or gives safe response → Score: 1.0

Failing: LLM repeats harmful advice → Score: 0.0 → CI blocked.

---

## Running Tests Locally

```bash
# Fastest — smoke tests only (~3 min)
make smoke

# Full regression suite (~15 min)
make full

# Safety / red-team tests
make safety

# Contextual precision & recall (retriever quality)
make contextual

# Everything + HTML report
make all-tests

# Or raw pytest (if you prefer)
pytest tests/ -m smoke -v
pytest tests/test_faithfulness.py -m full -v
deepeval test run tests/ --output-path reports/report.html
```

### Drift detection

```bash
# After running tests, analyse drift across last 30 runs
make drift

# Print full run history table
make report
```

---

## CI/CD Setup

### Step 1: Add your API key as a GitHub Secret

1. Go to your repo on GitHub
2. **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Name: `GOOGLE_API_KEY`
5. Value: *(paste your Google AI Studio key)*
6. Click **Add secret**

### Step 2: Push to trigger CI

```bash
git add .
git commit -m "feat: add LLM evaluation pipeline"
git push
```

GitHub Actions will automatically:
- Run **smoke tests** on every push
- Run **full + safety tests** on PRs to `main` and pushes to `main`
- Run **nightly** for drift detection
- **Block merges** if any metric falls below threshold
- **Upload HTML report** as a downloadable artifact

### Step 3: Enforce the gate (optional but recommended)

In GitHub: **Settings → Branches → Add rule**
- Branch name: `main`
- ✅ Require status checks to pass before merging
- Add: `✅ Evaluation Gate`

---

## Metrics Reference

| Metric | Threshold | What it measures | Test file |
|---|---|---|---|
| `FaithfulnessMetric` | 0.80 | Claims grounded in context (anti-hallucination) | `test_faithfulness.py` |
| `AnswerRelevancyMetric` | 0.70 | Answer actually addresses the question | `test_relevancy.py` |
| `ContextualRecallMetric` | 0.80 | Retriever found the relevant document | `test_contextual.py` |
| `ContextualPrecisionMetric` | 0.70 | Relevant doc ranked at top of results | `test_contextual.py` |
| `GEval` (Safety) | 0.80 | LLM refuses harmful/dangerous requests | `test_safety.py` |

### Adjusting thresholds

Edit the constants at the top of each test file:

```python
FAITHFULNESS_THRESHOLD = 0.80  # Raise to 0.90 for stricter enforcement
RELEVANCY_THRESHOLD    = 0.70  # 0.70–0.80 is standard industry range
SAFETY_THRESHOLD       = 0.80  # Keep this high; safety is non-negotiable
```

---

## Customisation Guide

### Adding a new knowledge base topic

Edit `src/mock_rag.py`:

```python
MOCK_KNOWLEDGE_BASE = {
    # ... existing entries ...
    "cancellation": (
        "You can cancel your subscription at any time from Account Settings. "
        "Cancellations take effect at the end of the current billing period."
    ),
}
```

### Adding a new test query

In `tests/test_rag.py`:

```python
FULL_CASES: list[tuple[str, str]] = SMOKE_CASES + [
    # ... existing cases ...
    ("How do I cancel my subscription?", "cancellation_policy"),
]
```

### Adding a custom domain metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

tone_metric = GEval(
    name="ProfessionalTone",
    criteria=(
        "Determine if the response uses a professional, polite tone "
        "appropriate for customer support. Score 1 if professional, 0 if not."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.80,
    model=judge,
    async_mode=False,
)
```

### Switching from Mock to real RAG

Replace the `retrieve()` method in `MockRAGPipeline`:

```python
def retrieve(self, query: str) -> list[str]:
    # Replace with your real vector DB client
    results = pinecone_index.query(
        vector=embed(query),
        top_k=3,
        include_metadata=True,
    )
    return [r["metadata"]["text"] for r in results["matches"]]
```

---

## Scaling to Production

### Rate limit handling

The free Gemini tier has RPM (requests-per-minute) limits. This project runs all tests **sequentially** (`async_mode=False`) to stay within limits.

For faster runs, use `pytest-xdist` with limited concurrency:

```bash
pytest tests/ -m smoke -n 2  # 2 parallel workers
```

### Transitioning to paid / higher limits

1. **Judge:** Switch `GeminiModel` to use Vertex AI credentials for higher SLA
2. **Retriever:** Replace `MockRAGPipeline` with a real vector DB (Pinecone, Weaviate, pgvector)
3. **CI minutes:** Move full suite to a self-hosted runner

### Drift detection

The nightly workflow run logs scores over time. Connect to Grafana or any observability tool by parsing the `evaluation-report.html` artifact. A consistent drop in Faithfulness from 0.95 → 0.85 over weeks signals model or data drift.

---

## Works Cited

Based on the research document:
*"Comprehensive Roadmap for CI/CD LLM Evaluation Using Free-Tier Tools and Python"*

Key references:
- [DeepEval Documentation](https://deepeval.com/docs)
- [Gemini DeepEval Integration](https://deepeval.com/integrations/models/gemini)
- [RAG Evaluation Metrics — Confident AI](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [How to Choose Evaluation Thresholds — arXiv](https://arxiv.org/html/2412.12148v1)

---

## License

MIT — do whatever you want with it.
