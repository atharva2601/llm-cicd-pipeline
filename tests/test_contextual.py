"""
test_contextual.py — Contextual Precision & Recall Tests
──────────────────────────────────────────────────────────
These metrics evaluate the RETRIEVER component of the RAG pipeline,
not the LLM generator itself.  Per the RAG Triad (PDF §2.2.3):

  Contextual Recall:    Did retrieval find the document containing
                        the answer?  (Coverage)

  Contextual Precision: Was the relevant document ranked first?
                        (Ranking quality)

Why this matters:
  A perfectly honest LLM can still give wrong answers if the
  retriever feeds it bad context.  Testing LLM + Retriever
  separately gives unambiguous failure signals.

Golden dataset structure (per PDF §4.3 Table 1):
  input          — user query
  expected_output — ideal human-verified answer
  context        — the documents the retriever SHOULD surface
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# ── Thresholds ───────────────────────────────────────────────────
CONTEXTUAL_RECALL_THRESHOLD    = 0.80
CONTEXTUAL_PRECISION_THRESHOLD = 0.70   # Slightly lower — ranking is harder

# ── Golden Dataset ───────────────────────────────────────────────
# Format: (input, expected_output, ideal_context_chunks)
#
# `ideal_context_chunks` is what a PERFECT retriever would return.
# We compare this against what MockRAGPipeline.retrieve() actually
# returns to score the retriever independently.
GOLDEN_CASES: list[tuple[str, str, list[str]]] = [
    (
        "How do I get a refund?",
        "Refunds are processed within 14 business days. No refunds on software.",
        [
            "Refunds are processed within 14 business days. "
            "No refunds are available on software or digital goods. "
            "Sale items are final sale and cannot be refunded."
        ],
    ),
    (
        "What are your support hours?",
        "Support is available Monday to Friday, 9 AM to 5 PM EST.",
        [
            "Customer support is available Monday to Friday, 9 AM to 5 PM EST. "
            "We are closed on public holidays."
        ],
    ),
    (
        "How do I reset my password?",
        "Click 'Forgot Password' on the login page to receive a reset link.",
        [
            "To reset your password, click 'Forgot Password' on the login page. "
            "A reset link will be sent to your registered email address."
        ],
    ),
    (
        "How long does standard shipping take?",
        "Standard shipping takes 5 to 7 business days.",
        [
            "Standard shipping takes 5–7 business days. "
            "Express shipping takes 1–2 business days at an additional cost. "
            "Free shipping is available on orders over $50."
        ],
    ),
    (
        "Can I return opened software?",
        "Opened software cannot be returned.",
        [
            "Items may be returned within 30 days of purchase in original condition. "
            "You must include the original receipt. "
            "Opened software cannot be returned."
        ],
    ),
    (
        "What does the warranty cover?",
        "The 1-year limited warranty covers manufacturing defects but not accidental damage.",
        [
            "All hardware products carry a 1-year limited warranty. "
            "The warranty covers manufacturing defects but not accidental damage."
        ],
    ),
    (
        "How do I contact support?",
        "Email support@example.com or call 1-800-555-0199 during business hours.",
        [
            "You can reach our support team by email at support@example.com "
            "or by phone at 1-800-555-0199 during business hours."
        ],
    ),
    (
        "What is the minimum age to create an account?",
        "You must be 18 years or older to create an account.",
        [
            "To create an account, visit our website and click 'Sign Up'. "
            "You must be 18 years or older to create an account."
        ],
    ),
]

SMOKE_CASES = GOLDEN_CASES[:3]
FULL_CASES  = GOLDEN_CASES


# ── Helper ────────────────────────────────────────────────────────
def _build_test_case(
    query: str,
    expected_output: str,
    ideal_context: list[str],
    rag_pipeline,
) -> LLMTestCase:
    """
    Runs the retriever and builds a LLMTestCase that compares
    retrieved_context (actual) against expected_output (golden).
    """
    actual_output, retrieved_context = rag_pipeline.answer(query)
    return LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,       # Golden answer
        retrieval_context=retrieved_context,   # What the retriever returned
        context=ideal_context,                 # What it SHOULD have returned
    )


# ── Smoke suite ──────────────────────────────────────────────────
@pytest.mark.smoke
@pytest.mark.parametrize("query,expected,ideal_ctx", SMOKE_CASES)
def test_contextual_smoke(
    query: str,
    expected: str,
    ideal_ctx: list[str],
    judge,
    rag_pipeline,
) -> None:
    """Fast retriever quality check — runs on every push."""
    test_case = _build_test_case(query, expected, ideal_ctx, rag_pipeline)

    recall = ContextualRecallMetric(
        threshold=CONTEXTUAL_RECALL_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )
    precision = ContextualPrecisionMetric(
        threshold=CONTEXTUAL_PRECISION_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    assert_test(test_case, [recall, precision])


# ── Full suite ───────────────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query,expected,ideal_ctx", FULL_CASES)
def test_contextual_full(
    query: str,
    expected: str,
    ideal_ctx: list[str],
    judge,
    rag_pipeline,
) -> None:
    """
    Comprehensive retriever evaluation — runs on PRs to main.
    Failing here means the retriever logic needs fixing, NOT the LLM.
    """
    test_case = _build_test_case(query, expected, ideal_ctx, rag_pipeline)

    recall = ContextualRecallMetric(
        threshold=CONTEXTUAL_RECALL_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )
    precision = ContextualPrecisionMetric(
        threshold=CONTEXTUAL_PRECISION_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    assert_test(test_case, [recall, precision])
