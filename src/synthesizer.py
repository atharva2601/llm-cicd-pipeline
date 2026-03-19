"""
src/synthesizer.py — DeepEval Synthesizer Wrapper
───────────────────────────────────────────────────
Wraps the DeepEval Synthesizer (PDF §4.3) to produce goldenset
LLMTestCase objects directly from raw text documents.

Two modes
─────────
1. Document mode (preferred):
   Pass a list of plain-text strings representing knowledge-base
   chunks.  The synthesizer generates questions + expected answers.

2. KB mode (convenience):
   Pass MOCK_KNOWLEDGE_BASE directly — each value becomes a chunk.

The synthesizer uses the same Gemini Flash judge model, so no
extra API keys are needed.

Usage (script)
──────────────
    from src.synthesizer import KBSynthesizer
    synth = KBSynthesizer()
    cases = synth.generate(n_per_chunk=2)
    # returns list[LLMTestCase] ready for DeepEval assert_test()

Usage (integrated in tests)
───────────────────────────
    # In a test file:
    from src.synthesizer import KBSynthesizer
    _synth_cases = KBSynthesizer().generate(n_per_chunk=1)

    @pytest.mark.parametrize("tc", _synth_cases)
    def test_synthesized(tc, judge):
        assert_test(tc, [FaithfulnessMetric(threshold=0.8, model=judge)])
"""

from __future__ import annotations

import os
import time
from typing import Iterator

import google.generativeai as genai
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "gemini-1.5-flash"
_RATE_LIMIT_DELAY = 1.5  # seconds between API calls (free-tier safety)


class KBSynthesizer:
    """
    Generates LLMTestCase objects from knowledge-base chunks using
    Gemini Flash.

    Parameters
    ──────────
    model_name : str
        Gemini model to use for synthesis (default: gemini-1.5-flash).
    delay : float
        Seconds to wait between API calls (respects free-tier RPM).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        delay: float = _RATE_LIMIT_DELAY,
    ) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. "
                "Copy .env.example → .env and add your key."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name,
            system_instruction=(
                "You are a dataset synthesis assistant. "
                "Always respond with valid JSON only. No markdown fences."
            ),
        )
        self._delay = delay

    # ── Private helpers ───────────────────────────────────────────
    def _call(self, prompt: str, temperature: float = 0.7) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=512,
            ),
        )
        raw = resp.text.strip()
        # Strip accidental markdown fences
        return raw.replace("```json", "").replace("```", "").strip()

    def _generate_questions(self, chunk: str, n: int) -> list[str]:
        import json

        prompt = (
            f'Given this knowledge-base chunk:\n"{chunk}"\n\n'
            f"Generate {n} realistic customer questions this chunk answers. "
            f"Each should test a DIFFERENT aspect.\n"
            f'Return ONLY a JSON array: ["question 1", ...]'
        )
        try:
            raw = self._call(prompt, temperature=0.8)
            result = json.loads(raw)
            return [q for q in result if isinstance(q, str)][:n]
        except Exception:
            return []

    def _generate_answer(self, chunk: str, question: str) -> str:
        import json

        prompt = (
            f'Given this knowledge-base chunk:\n"{chunk}"\n\n'
            f'And this question:\n"{question}"\n\n'
            "Write the ideal, concise customer-support answer based ONLY "
            "on the chunk.  Do not add external knowledge.\n"
            'Return ONLY: {"expected_output": "your answer"}'
        )
        try:
            raw = self._call(prompt, temperature=0.0)
            return json.loads(raw).get("expected_output", "")
        except Exception:
            return ""

    # ── Public API ────────────────────────────────────────────────
    def generate_from_chunks(
        self,
        chunks: list[str],
        n_per_chunk: int = 2,
    ) -> Iterator[LLMTestCase]:
        """
        Yield LLMTestCase objects synthesized from raw text chunks.

        Parameters
        ──────────
        chunks : list[str]
            Text chunks to generate questions from.
        n_per_chunk : int
            Number of questions to generate per chunk.

        Yields
        ──────
        LLMTestCase with input (question) and expected_output (answer).
        """
        for chunk in chunks:
            questions = self._generate_questions(chunk, n_per_chunk)
            for question in questions:
                time.sleep(self._delay)
                expected = self._generate_answer(chunk, question)
                if question and expected:
                    yield LLMTestCase(
                        input=question,
                        actual_output="",        # Filled in by the test
                        expected_output=expected,
                        retrieval_context=[chunk],
                    )
                time.sleep(self._delay)

    def generate(
        self,
        kb: dict[str, str] | None = None,
        n_per_chunk: int = 2,
    ) -> list[LLMTestCase]:
        """
        Generate test cases from the knowledge base.

        Parameters
        ──────────
        kb : dict[str, str] | None
            Knowledge base dict (topic → chunk text).
            Defaults to MOCK_KNOWLEDGE_BASE from src/mock_rag.py.
        n_per_chunk : int
            Number of questions per KB topic.

        Returns
        ───────
        list[LLMTestCase] — ready for DeepEval assert_test().
        """
        if kb is None:
            from src.mock_rag import MOCK_KNOWLEDGE_BASE
            kb = MOCK_KNOWLEDGE_BASE

        chunks = list(kb.values())
        return list(self.generate_from_chunks(chunks, n_per_chunk=n_per_chunk))
