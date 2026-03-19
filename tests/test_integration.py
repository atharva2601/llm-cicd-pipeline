"""
test_integration.py — End-to-End Pipeline Integration Tests
──────────────────────────────────────────────────────────────
These tests exercise the FULL STACK:

  User query
    → MockRAGPipeline.retrieve()   (retriever)
    → MockRAGPipeline.generate()   (LLM generator)
    → DeepEval metrics              (judge)
    → assert_test()                 (gate)

Unlike unit tests (which test one metric in isolation), integration
tests verify that:

  1. The pipeline boots correctly end-to-end
  2. Retrieval + generation + evaluation complete without error
  3. Latency stays within acceptable bounds
  4. The judge produces deterministic verdicts across repeated runs

These are intentionally slower and are only run on PRs to main
and direct pushes to main (not on every feature-branch push).
"""

from __future__ import annotations

import time

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.metrics import no_hallucinated_urls, policy_compliance, professional_tone

# ── Latency budget ────────────────────────────────────────────────
# The full generate() call (LLM + judge) should complete within this
# many seconds.  Adjust based on your Gemini Flash latency profile.
MAX_LATENCY_SECONDS = 30.0

# ── Sanity queries — should always produce valid, grounded answers ─
INTEGRATION_QUERIES = [
    "How do I get a refund?",
    "What are your support hours?",
    "How do I reset my password?",
]


# ── Pipeline boot test ────────────────────────────────────────────
def test_pipeline_initialises(rag_pipeline) -> None:
    """
    Verifies that MockRAGPipeline can be instantiated and its
    components are correctly wired before any test runs.
    """
    assert rag_pipeline is not None
    assert hasattr(rag_pipeline, "retrieve")
    assert hasattr(rag_pipeline, "generate")
    assert hasattr(rag_pipeline, "answer")
    assert len(rag_pipeline.knowledge_base) > 0, (
        "Knowledge base is empty — check src/mock_rag.py"
    )


# ── Retriever unit test ───────────────────────────────────────────
def test_retriever_returns_relevant_context(rag_pipeline) -> None:
    """
    The retriever must return the correct KB chunk for known queries.
    This is a deterministic test — no judge or LLM call needed.
    """
    tests = [
        ("refund policy",    "refund"),
        ("support hours",    "Monday"),
        ("reset my password","Forgot Password"),
        ("shipping time",    "business days"),
    ]
    for query, expected_substring in tests:
        context = rag_pipeline.retrieve(query)
        assert context, f"retrieve('{query}') returned empty list"
        combined = " ".join(context).lower()
        assert expected_substring.lower() in combined, (
            f"Expected '{expected_substring}' in context for query '{query}'.\n"
            f"Got: {context}"
        )


# ── Full pipeline latency test ────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query", INTEGRATION_QUERIES)
def test_pipeline_latency(query: str, rag_pipeline) -> None:
    """
    The generate() call must complete within MAX_LATENCY_SECONDS.
    A consistent timeout here signals rate-limit issues or
    network problems in the CI environment.
    """
    start = time.monotonic()
    actual_output, context = rag_pipeline.answer(query)
    elapsed = time.monotonic() - start

    assert actual_output, f"generate() returned empty string for '{query}'"
    assert context,       f"generate() returned empty context for '{query}'"
    assert elapsed < MAX_LATENCY_SECONDS, (
        f"Pipeline took {elapsed:.1f}s for '{query}' "
        f"(budget: {MAX_LATENCY_SECONDS}s). "
        "Possible rate-limit or network issue."
    )


# ── Full stack evaluation test ────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query", INTEGRATION_QUERIES)
def test_full_stack_evaluation(query: str, judge, rag_pipeline) -> None:
    """
    Runs the complete pipeline including the judge evaluation.
    Verifies that Faithfulness + Relevancy both pass on core queries.
    This is the most comprehensive single-query integration test.
    """
    actual_output, context = rag_pipeline.answer(query)

    tc = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=context,
    )

    assert_test(
        tc,
        [
            FaithfulnessMetric(threshold=0.80, model=judge,
                               include_reason=True, async_mode=False),
            AnswerRelevancyMetric(threshold=0.70, model=judge,
                                  include_reason=True, async_mode=False),
        ],
    )


# ── Judge determinism test ────────────────────────────────────────
@pytest.mark.full
def test_judge_determinism(judge, rag_pipeline) -> None:
    """
    Runs the same query twice and checks that the judge assigns
    the same pass/fail verdict both times.

    The judge uses temperature=0.0, so verdicts should be stable.
    A flaky judge would produce inconsistent CI results.

    Note: Scores may vary slightly (±0.05) but pass/fail should match.
    """
    query = "How do I get a refund?"

    scores: list[float] = []
    for _ in range(2):
        actual_output, context = rag_pipeline.answer(query)
        tc = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=context,
        )
        metric = FaithfulnessMetric(
            threshold=0.0,    # Don't fail — just collect score
            model=judge,
            async_mode=False,
        )
        metric.measure(tc)
        if metric.score is not None:
            scores.append(metric.score)
        time.sleep(1.5)   # Rate limit

    if len(scores) == 2:
        delta = abs(scores[0] - scores[1])
        assert delta < 0.20, (
            f"Judge score variance too high: {scores[0]:.3f} vs {scores[1]:.3f} "
            f"(Δ={delta:.3f}). "
            "This may indicate judge instability. "
            "Check: is temperature=0.0 set in conftest.py?"
        )


# ── Fallback behaviour test ───────────────────────────────────────
def test_retriever_fallback_for_unknown_query(rag_pipeline) -> None:
    """
    When a query has no matching KB topic, the retriever must return
    the fallback message rather than an empty list (which would cause
    the generator to produce a context-free hallucination).
    """
    context = rag_pipeline.retrieve("quantum entanglement teleportation")
    assert context, "Retriever returned empty list for unknown query"
    assert any("No relevant context" in c for c in context), (
        f"Expected fallback message in context for unknown query. Got: {context}"
    )


# ── Output format test ────────────────────────────────────────────
@pytest.mark.parametrize("query", INTEGRATION_QUERIES)
def test_output_is_non_empty_string(query: str, rag_pipeline) -> None:
    """
    The pipeline must always return a non-empty string.
    An empty string or None would break downstream metric evaluation.
    """
    actual_output, context = rag_pipeline.answer(query)

    assert isinstance(actual_output, str),  (
        f"actual_output is not a string: {type(actual_output)}"
    )
    assert len(actual_output.strip()) > 0, (
        f"actual_output is empty for query: '{query}'"
    )
    assert isinstance(context, list),  (
        f"context is not a list: {type(context)}"
    )
    assert len(context) > 0, (
        f"context list is empty for query: '{query}'"
    )
