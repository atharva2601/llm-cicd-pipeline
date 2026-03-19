"""
test_custom_metrics.py — Domain-Specific Quality Tests
────────────────────────────────────────────────────────
Tests that go beyond the RAG Triad (Faithfulness + Relevancy)
and evaluate the *quality* of the response from a customer-
experience perspective using the custom metrics defined in
src/metrics.py.

Why these matter
────────────────
  Faithfulness = "Did the LLM make things up?"
  Relevancy    = "Did the LLM answer the question?"
  These tests  = "Is the answer actually GOOD?"

A response can be perfectly faithful and relevant while still
being:
  • Rude or robotic (ProfessionalTone)
  • Missing half the question (Completeness)
  • Stuffed with filler (Conciseness)
  • Claiming a fake email address (NoHallucinatedURLs)
  • Accidentally contradicting a stated policy (PolicyCompliance)
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from src.metrics import (
    action_clarity,
    completeness,
    conciseness,
    empathy,
    no_hallucinated_urls,
    policy_compliance,
    professional_tone,
)

# ── Test data ─────────────────────────────────────────────────────
# Format: (query, description)
TONE_CASES = [
    ("How do I get a refund?",              "refund_tone"),
    ("What are your support hours?",        "hours_tone"),
    ("How do I reset my password?",         "login_tone"),
    ("I need help contacting your team.",   "contact_tone"),
]

COMPLETENESS_CASES = [
    ("How do I get a refund?",                    "simple_completeness"),
    ("What are your hours and how do I contact you?", "multi_part_completeness"),
    ("How long does shipping take and is it free?",   "shipping_completeness"),
]

POLICY_CASES = [
    ("Can I return opened software?",                "software_return_policy"),
    ("Does the warranty cover accidental damage?",   "warranty_policy"),
    ("Are sale items refundable?",                   "sale_refund_policy"),
]

URL_CASES = [
    ("How do I contact support?",  "contact_url_check"),
    ("How do I reset my password?", "login_url_check"),
]

STEP_CASES = [
    ("How do I reset my password?",    "password_steps"),
    ("How do I create an account?",    "signup_steps"),
]


# ── Tone tests ────────────────────────────────────────────────────
@pytest.mark.smoke
@pytest.mark.parametrize("query,label", TONE_CASES[:2])
def test_professional_tone_smoke(query: str, label: str, judge, rag_pipeline) -> None:
    """Fast tone check — every push."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [professional_tone(judge)])


@pytest.mark.full
@pytest.mark.parametrize("query,label", TONE_CASES)
def test_professional_tone_full(query: str, label: str, judge, rag_pipeline) -> None:
    """Full tone check across all query types."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [professional_tone(judge)])


# ── Completeness tests ────────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query,label", COMPLETENESS_CASES)
def test_completeness(query: str, label: str, judge, rag_pipeline) -> None:
    """Ensures multi-part questions are fully answered."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [completeness(judge)])


# ── Conciseness tests ─────────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query,label", TONE_CASES)
def test_conciseness(query: str, label: str, judge, rag_pipeline) -> None:
    """Checks responses don't pad with filler or repetition."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [conciseness(judge)])


# ── Policy compliance tests ───────────────────────────────────────
@pytest.mark.smoke
@pytest.mark.parametrize("query,label", POLICY_CASES[:1])
def test_policy_compliance_smoke(query: str, label: str, judge, rag_pipeline) -> None:
    """Fast policy check — ensures no exceptions are invented."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [policy_compliance(judge)])


@pytest.mark.full
@pytest.mark.parametrize("query,label", POLICY_CASES)
def test_policy_compliance_full(query: str, label: str, judge, rag_pipeline) -> None:
    """Full policy compliance check across all binary policy queries."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [policy_compliance(judge)])


# ── No hallucinated URLs/contacts ─────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query,label", URL_CASES)
def test_no_hallucinated_urls(query: str, label: str, judge, rag_pipeline) -> None:
    """
    High-severity: any fabricated URL/email/phone is an immediate failure.
    The LLM must only cite contact details that are in the KB context.
    """
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [no_hallucinated_urls(judge, threshold=0.90)])


# ── Action clarity tests ──────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query,label", STEP_CASES)
def test_action_clarity(query: str, label: str, judge, rag_pipeline) -> None:
    """Checks that step-by-step responses are clear and well-ordered."""
    actual_output, ctx = rag_pipeline.answer(query)
    tc = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=ctx)
    assert_test(tc, [action_clarity(judge)])


# ── Combined quality suite ────────────────────────────────────────
@pytest.mark.full
def test_full_quality_suite(judge, rag_pipeline) -> None:
    """
    The kitchen-sink test: runs ALL quality metrics on a single
    representative query to get a holistic quality score.

    If this test degrades over time, something fundamental changed
    in the model or prompting that affects ALL quality dimensions.
    """
    query = "How do I contact your support team and what are their hours?"
    actual_output, ctx = rag_pipeline.answer(query)

    tc = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=ctx,
    )

    assert_test(
        tc,
        [
            professional_tone(judge),
            completeness(judge),
            conciseness(judge),
            no_hallucinated_urls(judge),
            policy_compliance(judge),
        ],
    )
