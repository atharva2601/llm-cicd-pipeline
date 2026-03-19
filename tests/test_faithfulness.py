"""
test_faithfulness.py — Hallucination / Groundedness Tests
───────────────────────────────────────────────────────────
Faithfulness Score = Claims Supported by Context / Total Claims

A score < threshold (0.80) means the LLM invented information
not present in the retrieved context → HALLUCINATION DETECTED.

Smoke tests (marked @pytest.mark.smoke) run on EVERY push.
Full tests (marked @pytest.mark.full) run on merges to main.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# ── Test data ────────────────────────────────────────────────────
# Format: (query, description)
# The pipeline produces the actual_output + retrieval_context live.

SMOKE_QUERIES = [
    "How do I get a refund?",
    "What are your support hours?",
    "How do I reset my password?",
    "How long does shipping take?",
    "Can I return an opened software?",
]

FULL_QUERIES = SMOKE_QUERIES + [
    "Does the warranty cover accidental damage?",
    "How do I contact support?",
    "What is the age requirement to create an account?",
    "Is express shipping available?",
    "Are sale items refundable?",
]

FAITHFULNESS_THRESHOLD = 0.80  # Industry baseline: ≥ 0.80 = "Good"


# ── Smoke suite ──────────────────────────────────────────────────
@pytest.mark.smoke
@pytest.mark.parametrize("query", SMOKE_QUERIES)
def test_faithfulness_smoke(query: str, judge, rag_pipeline) -> None:
    """
    Fast hallucination check.
    Runs on every push to catch obvious regressions quickly.
    """
    actual_output, retrieval_context = rag_pipeline.answer(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=judge,
        include_reason=True,   # Prints WHY it failed in CI logs
        async_mode=False,      # Sequential → avoids free-tier rate limits
    )

    assert_test(test_case, [metric])


# ── Full suite ───────────────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query", FULL_QUERIES)
def test_faithfulness_full(query: str, judge, rag_pipeline) -> None:
    """
    Comprehensive hallucination check over the entire knowledge base.
    Runs on PRs targeting main and on merge to main.
    """
    actual_output, retrieval_context = rag_pipeline.answer(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    assert_test(test_case, [metric])
