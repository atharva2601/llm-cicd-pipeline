"""
mock_rag.py — Simulated RAG Pipeline (System Under Test)
─────────────────────────────────────────────────────────
Purpose:
  Replaces a real vector-database retriever with a deterministic
  in-memory dictionary. This lets the CI runner evaluate LLM
  generation quality without spinning up Pinecone / Weaviate.

Why Mock?
  • No cold-start latency from a real DB
  • Retrieval is fully deterministic → flaky tests caused by DB
    inconsistency are eliminated
  • Zero extra cost or account setup beyond the Gemini API key

Design:
  retrieve()  – Keyword-based lookup into self.mock_kb
                (simulates semantic vector search results)
  generate()  – Builds a RAG prompt and calls Gemini Flash
  answer()    – Convenience wrapper returning (answer, context)
"""

from __future__ import annotations

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Configure Gemini (the application model, NOT the judge) ──────
_api_key = os.getenv("GOOGLE_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Copy .env.example → .env and add your key, "
        "or set it as a GitHub Actions secret."
    )

genai.configure(api_key=_api_key)


# ── Knowledge Base ───────────────────────────────────────────────
# Simulates the chunks that a vector DB would return.
# Extend this dict to cover all topics your app handles.
MOCK_KNOWLEDGE_BASE: dict[str, str] = {
    "refund": (
        "Refunds are processed within 14 business days. "
        "No refunds are available on software or digital goods. "
        "Sale items are final sale and cannot be refunded."
    ),
    "hours": (
        "Customer support is available Monday to Friday, 9 AM to 5 PM EST. "
        "We are closed on public holidays."
    ),
    "login": (
        "To reset your password, click 'Forgot Password' on the login page. "
        "A reset link will be sent to your registered email address."
    ),
    "shipping": (
        "Standard shipping takes 5–7 business days. "
        "Express shipping takes 1–2 business days at an additional cost. "
        "Free shipping is available on orders over $50."
    ),
    "return": (
        "Items may be returned within 30 days of purchase in original condition. "
        "You must include the original receipt. "
        "Opened software cannot be returned."
    ),
    "warranty": (
        "All hardware products carry a 1-year limited warranty. "
        "The warranty covers manufacturing defects but not accidental damage."
    ),
    "contact": (
        "You can reach our support team by email at support@example.com "
        "or by phone at 1-800-555-0199 during business hours."
    ),
    "account": (
        "To create an account, visit our website and click 'Sign Up'. "
        "You must be 18 years or older to create an account."
    ),
}

# ── System prompt injected into every RAG call ───────────────────
_RAG_SYSTEM_PROMPT = (
    "You are a professional customer-support assistant. "
    "Answer ONLY using the information in the provided Context. "
    "If the Context does not contain enough information, say "
    "'I don't have enough information to answer that.' "
    "Never add information that is not present in the Context. "
    "Keep your answer concise and factual."
)


class MockRAGPipeline:
    """
    Deterministic retriever + Gemini-powered generator.

    Usage
    ─────
    pipeline = MockRAGPipeline()
    answer, context = pipeline.answer("How do I get a refund?")
    """

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_RAG_SYSTEM_PROMPT,
        )
        self.knowledge_base = MOCK_KNOWLEDGE_BASE

    # ── Retriever ─────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[str]:
        """
        Keyword-based retrieval (simulates vector search).
        Returns a list of matching knowledge-base chunks.
        """
        q = query.lower()
        results: list[str] = [
            chunk
            for keyword, chunk in self.knowledge_base.items()
            if keyword in q
        ]
        if not results:
            results = ["No relevant context found in the knowledge base."]
        return results

    # ── Generator ─────────────────────────────────────────────────
    def generate(self, query: str) -> tuple[str, list[str]]:
        """
        Runs the full RAG pipeline.
        Returns (generated_answer, retrieved_context_list).
        """
        context_chunks = self.retrieve(query)
        context_str = "\n\n".join(context_chunks)

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,   # Low temp → more consistent answers
                max_output_tokens=512,
            ),
        )
        return response.text.strip(), context_chunks

    # ── Convenience alias ─────────────────────────────────────────
    def answer(self, query: str) -> tuple[str, list[str]]:
        """Alias for generate(). Cleaner call-site in tests."""
        return self.generate(query)
