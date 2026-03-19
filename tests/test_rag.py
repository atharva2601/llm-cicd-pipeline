"""
test_rag.py — Combined RAG Triad Tests (Faithfulness + Relevancy)
──────────────────────────────────────────────────────────────────
Runs both metrics in a single assert_test() call per query.
This is the primary regression gate used in CI.

Each parametrized row includes:
  query           — the user's question
  expected_theme  — human label (not used by metrics, aids logging)
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# ── Golden dataset ───────────────────────────────────────────────
# (query, human_label)
SMOKE_CASES: list[tuple[str, str]] = [
    ("How do I get a refund?",              "refund_policy"),
    ("What are your customer support hours?", "support_hours"),
    ("How do I reset my password?",         "password_reset"),
]

FULL_CASES: list[tuple[str, str]] = SMOKE_CASES + [
    ("How long does standard shipping take?",        "shipping_sla"),
    ("Can I return opened software?",                "software_return"),
    ("Does the warranty cover accidental damage?",   "warranty_scope"),
    ("How can I contact your support team?",         "contact_info"),
    ("What is the minimum age to create an account?","account_age"),
    ("Is free shipping available?",                  "free_shipping_threshold"),
    ("Are sale items refundable?",                   "sale_refund_policy"),
]

FAITHFULNESS_THRESHOLD = 0.80
RELEVANCY_THRESHOLD    = 0.70


def _run_rag_test(
    query: str,
    judge,
    rag_pipeline,
    faith_threshold: float = FAITHFULNESS_THRESHOLD,
    rel_threshold: float   = RELEVANCY_THRESHOLD,
) -> None:
    """Shared logic for smoke and full suites."""
    actual_output, retrieval_context = rag_pipeline.answer(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    faithfulness = FaithfulnessMetric(
        threshold=faith_threshold,
        model=judge,
        include_reason=True,
        async_mode=False,
    )
    relevancy = AnswerRelevancyMetric(
        threshold=rel_threshold,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    assert_test(test_case, [faithfulness, relevancy])


# ── Smoke suite ──────────────────────────────────────────────────
@pytest.mark.smoke
@pytest.mark.parametrize("query,label", SMOKE_CASES)
def test_rag_triad_smoke(query: str, label: str, judge, rag_pipeline) -> None:
    """
    Critical path tests — run on EVERY push.
    Failing here blocks the merge immediately.
    """
    _run_rag_test(query, judge, rag_pipeline)


# ── Full suite ───────────────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query,label", FULL_CASES)
def test_rag_triad_full(query: str, label: str, judge, rag_pipeline) -> None:
    """
    Exhaustive regression suite — run on PRs targeting main
    and on direct pushes to main.
    """
    _run_rag_test(query, judge, rag_pipeline)
