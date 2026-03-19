"""
src/metrics.py — Reusable Custom Metric Definitions
──────────────────────────────────────────────────────
Implements PDF §3.1: "Customizability: allows for the definition of
custom metrics using the Judge, which is essential for domain-specific
testing (e.g., 'Medical Accuracy' or 'Tone Consistency')."

All metrics here use DeepEval's GEval class with a Gemini Flash judge.
They are importable from any test file without repeating the prompt logic.

Available metrics
─────────────────
  professional_tone(judge)   — Is the response appropriately formal?
  completeness(judge)        — Does the response address all parts of the query?
  conciseness(judge)         — Is the response free of unnecessary filler?
  no_hallucinated_urls(judge)— Are all URLs/emails from the context only?
  policy_compliance(judge)   — Does the response align with policy constraints?
  empathy(judge)             — Is the tone appropriately empathetic?

Usage
─────
    from src.metrics import professional_tone, completeness
    from deepeval import assert_test

    def test_my_response(judge, rag_pipeline):
        actual_output, ctx = rag_pipeline.answer("How do I get a refund?")
        tc = LLMTestCase(input=..., actual_output=actual_output, ...)
        assert_test(tc, [professional_tone(judge), completeness(judge)])
"""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# ── Shared evaluation params used across most metrics ────────────
_STANDARD_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]
_WITH_CONTEXT = _STANDARD_PARAMS + [LLMTestCaseParams.RETRIEVAL_CONTEXT]


def professional_tone(judge, threshold: float = 0.75) -> GEval:
    """
    Checks that the response uses a professional, polite tone
    appropriate for a customer-support context.

    Penalises: slang, rudeness, excessive informality, sarcasm.
    Does NOT penalise: warmth, contractions, plain English.
    """
    return GEval(
        name="ProfessionalTone",
        criteria=(
            "Evaluate whether the actual output uses a professional and "
            "polite tone appropriate for a customer-support representative. "
            "Score 1.0 if the tone is consistently professional. "
            "Score 0.5 if it is acceptable but slightly informal. "
            "Score 0.0 if the tone is rude, sarcastic, or inappropriate."
        ),
        evaluation_params=_STANDARD_PARAMS,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def completeness(judge, threshold: float = 0.75) -> GEval:
    """
    Checks that the response addresses ALL parts of the user's query,
    not just the most obvious one.

    Example failure: User asks "What are your hours AND your email?"
    and the response only provides the hours.
    """
    return GEval(
        name="Completeness",
        criteria=(
            "Determine whether the actual output fully addresses all distinct "
            "sub-questions or requests in the input. "
            "Score 1.0 if every part of the question is answered. "
            "Score 0.5 if most parts are answered but one is missed or vague. "
            "Score 0.0 if a significant part of the question is ignored."
        ),
        evaluation_params=_STANDARD_PARAMS,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def conciseness(judge, threshold: float = 0.70) -> GEval:
    """
    Checks that the response is concise and free of unnecessary filler,
    repetition, or padding.

    Does NOT penalise necessary detail — only waste.
    """
    return GEval(
        name="Conciseness",
        criteria=(
            "Evaluate whether the actual output is appropriately concise. "
            "A good response provides the needed information without padding, "
            "repetition, or unnecessary preamble (e.g. 'Great question!'). "
            "Score 1.0 if the response is tight and direct. "
            "Score 0.5 if there is minor padding but the core is fine. "
            "Score 0.0 if the response is excessively verbose or repetitive."
        ),
        evaluation_params=_STANDARD_PARAMS,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def no_hallucinated_urls(judge, threshold: float = 0.90) -> GEval:
    """
    Checks that any URLs, email addresses, or phone numbers in the
    response appear verbatim in the retrieval context.

    This is a HIGH-SEVERITY metric: hallucinated contact details can
    lead to users reaching the wrong destination, which is a liability.
    Threshold is deliberately high (0.90).
    """
    return GEval(
        name="NoHallucinatedURLs",
        criteria=(
            "Check whether the actual output contains any URLs, email addresses, "
            "or phone numbers. If it does, verify that each one appears verbatim "
            "in the retrieval context. "
            "Score 1.0 if: (a) there are no contact details at all, or "
            "(b) all contact details are present in the context. "
            "Score 0.0 if any URL, email, or phone number in the response "
            "does NOT appear in the context (hallucinated contact detail)."
        ),
        evaluation_params=_WITH_CONTEXT,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def policy_compliance(judge, threshold: float = 0.80) -> GEval:
    """
    Checks that the response does not contradict stated policies.

    Example: If the context says "No refunds on software" and the
    response says "We can make an exception for software", this fails.
    """
    return GEval(
        name="PolicyCompliance",
        criteria=(
            "Determine whether the actual output is consistent with the policies "
            "and rules stated in the retrieval context. "
            "Score 1.0 if the response correctly represents all applicable policies. "
            "Score 0.5 if the response is mostly correct but introduces a minor ambiguity. "
            "Score 0.0 if the response contradicts, softens, or overrides a stated policy "
            "(e.g. offering exceptions that the context does not allow)."
        ),
        evaluation_params=_WITH_CONTEXT,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def empathy(judge, threshold: float = 0.65) -> GEval:
    """
    Checks that the response acknowledges the customer's situation
    appropriately — especially for sensitive topics like refunds,
    complaints, or account issues.

    Lower default threshold (0.65) because empathy is more subjective
    and varies by brand voice. Adjust to your use case.
    """
    return GEval(
        name="Empathy",
        criteria=(
            "Evaluate whether the actual output appropriately acknowledges the "
            "customer's situation and responds with empathy when the context warrants it. "
            "For routine queries (e.g. 'what are your hours?'), neutrality is fine (score: 0.7). "
            "For queries involving problems, complaints, or frustration, the response "
            "should acknowledge the difficulty before diving into the solution (score: 1.0 if done). "
            "Score 0.0 if the response is robotic or dismissive in a sensitive context."
        ),
        evaluation_params=_STANDARD_PARAMS,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def action_clarity(judge, threshold: float = 0.75) -> GEval:
    """
    Checks that when a response involves steps or actions, the steps
    are clearly numbered or sequenced so the user knows what to do next.
    """
    return GEval(
        name="ActionClarity",
        criteria=(
            "If the actual output describes a process, procedure, or set of steps, "
            "evaluate whether those steps are clearly presented in logical order. "
            "Score 1.0 if steps are clear, ordered, and actionable. "
            "Score 0.7 if steps are present but could be clearer. "
            "Score 0.3 if steps are jumbled or hard to follow. "
            "If the response does not involve steps (e.g. a simple factual answer), "
            "score 1.0 automatically — this metric is not applicable."
        ),
        evaluation_params=_STANDARD_PARAMS,
        threshold=threshold,
        model=judge,
        async_mode=False,
    )
