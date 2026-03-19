"""
conftest.py — Shared pytest fixtures
──────────────────────────────────────
All fixtures defined here are automatically available to every
test file in the tests/ directory without an import.

Key design choices
──────────────────
• temperature=0.0 on the judge → deterministic verdicts (no
  randomness in the grading process itself)
• A single judge instance is created per test SESSION to minimise
  API calls to Gemini Flash
• The MockRAGPipeline is also session-scoped; its retriever is
  deterministic so sharing one instance across tests is safe
"""

from __future__ import annotations

import os

import pytest
from deepeval.models import GeminiModel
from dotenv import load_dotenv

from src.mock_rag import MockRAGPipeline

load_dotenv()


# ── Gemini Judge (session-scoped — one instance for whole run) ───
@pytest.fixture(scope="session")
def judge() -> GeminiModel:
    """
    Returns a Gemini 1.5 Flash instance configured as the LLM Judge.

    temperature=0.0  → maximally deterministic evaluation scores
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip(
            "GOOGLE_API_KEY not set. "
            "Add it to .env (local) or GitHub Secrets (CI)."
        )
    return GeminiModel(
        model_name="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.0,
    )


# ── Application under test ───────────────────────────────────────
@pytest.fixture(scope="session")
def rag_pipeline() -> MockRAGPipeline:
    """Returns the Mock RAG pipeline (system under test)."""
    return MockRAGPipeline()
