"""
test_relevancy.py — Answer Relevancy Tests
────────────────────────────────────────────
Relevancy Score = avg CosineSimilarity(original_question,
                                       synthetic_questions_from_answer)

A low score means the LLM gave a generic or evasive answer that
doesn't actually address what the user asked.

Threshold: 0.70  (slightly lower than Faithfulness because answer
phrasing can legitimately vary more than factual grounding).
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

SMOKE_QUERIES = [
    "How do I get a refund?",
    "What are your support hours?",
    "How do I reset my password?",
]

FULL_QUERIES = SMOKE_QUERIES + [
    "How long does shipping take?",
    "Can I return an opened software product?",
    "Does the warranty cover accidental damage?",
    "How do I contact customer support?",
    "What is the age requirement to create an account?",
    "Is express shipping available, and how much does it cost?",
    "Are items bought on sale eligible for a refund?",
]

RELEVANCY_THRESHOLD = 0.70


# ── Smoke suite ──────────────────────────────────────────────────
@pytest.mark.smoke
@pytest.mark.parametrize("query", SMOKE_QUERIES)
def test_relevancy_smoke(query: str, judge, rag_pipeline) -> None:
    """Fast relevancy check — runs on every push."""
    actual_output, retrieval_context = rag_pipeline.answer(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    assert_test(test_case, [metric])


# ── Full suite ───────────────────────────────────────────────────
@pytest.mark.full
@pytest.mark.parametrize("query", FULL_QUERIES)
def test_relevancy_full(query: str, judge, rag_pipeline) -> None:
    """Comprehensive relevancy check — runs on PRs and merges to main."""
    actual_output, retrieval_context = rag_pipeline.answer(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    assert_test(test_case, [metric])
