#!/usr/bin/env python3
"""
scripts/synthesize_dataset.py — Automated Golden Dataset Generator
────────────────────────────────────────────────────────────────────
Implements PDF §4.3 "Synthetic Data Generation Strategy":

  1. Feed your raw source documents (text or JSON)
  2. Gemini Flash generates realistic user questions from the text
  3. Gemini Flash generates the expected ("Golden") answer
  4. Output is saved to data/golden_dataset_synthesized.json

This bootstraps evaluation from ZERO to 50+ test cases in minutes.

Usage
─────
    python scripts/synthesize_dataset.py
    python scripts/synthesize_dataset.py --source data/golden_dataset.json
    python scripts/synthesize_dataset.py --variations 3 --output data/my_dataset.json

Arguments
─────────
  --source    Path to source knowledge base (JSON or .txt dir)
              Defaults to src/mock_rag.py's MOCK_KNOWLEDGE_BASE
  --variations Number of question variations per KB topic (default: 2)
  --output    Output path (default: data/golden_dataset_synthesized.json)
  --dry-run   Print generated cases without saving
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent

import google.generativeai as genai
from dotenv import load_dotenv

# ── Bootstrap path so we can import src ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mock_rag import MOCK_KNOWLEDGE_BASE

load_dotenv()

# ── Gemini setup ─────────────────────────────────────────────────
_api_key = os.getenv("GOOGLE_API_KEY")
if not _api_key:
    sys.exit(
        "ERROR: GOOGLE_API_KEY not set.\n"
        "Copy .env.example → .env and add your key."
    )

genai.configure(api_key=_api_key)
_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction=(
        "You are a dataset generation assistant for LLM evaluation. "
        "Always respond with valid JSON only. No markdown, no preamble."
    ),
)


# ── Prompts ──────────────────────────────────────────────────────
_QUESTION_GEN_PROMPT = dedent("""\
    Given this customer support knowledge-base chunk:

    "{chunk}"

    Generate {n} realistic customer questions that this chunk would answer.
    Each question should test a DIFFERENT aspect of the content.

    Return ONLY a JSON array of strings:
    ["question 1", "question 2", ...]
""")

_ANSWER_GEN_PROMPT = dedent("""\
    Given this knowledge-base chunk:

    "{chunk}"

    And this customer question:

    "{question}"

    Write the ideal, concise answer a support agent would give based ONLY
    on the information in the chunk.  Do not add external knowledge.

    Return ONLY a JSON object:
    {{"expected_output": "your answer here"}}
""")


# ── Core synthesis functions ──────────────────────────────────────
def generate_questions(chunk: str, n: int = 2) -> list[str]:
    """Use Gemini to generate n questions from a KB chunk."""
    prompt = _QUESTION_GEN_PROMPT.format(chunk=chunk, n=n)
    try:
        resp = _model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,   # Some variety in questions
                max_output_tokens=512,
            ),
        )
        raw = resp.text.strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        questions = json.loads(raw)
        return [q for q in questions if isinstance(q, str)]
    except Exception as exc:
        print(f"  ⚠️  Question generation failed: {exc}")
        return []


def generate_expected_answer(chunk: str, question: str) -> str:
    """Use Gemini to produce the golden answer for a question."""
    prompt = _ANSWER_GEN_PROMPT.format(chunk=chunk, question=question)
    try:
        resp = _model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,   # Deterministic gold answers
                max_output_tokens=256,
            ),
        )
        raw = resp.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return data.get("expected_output", raw)
    except Exception as exc:
        print(f"  ⚠️  Answer generation failed: {exc}")
        return ""


def synthesize_from_kb(
    kb: dict[str, str],
    variations: int = 2,
    rate_limit_delay: float = 1.5,
) -> list[dict]:
    """
    Main synthesis loop.
    Iterates over each KB topic → generates questions → generates answers.
    Returns a list of golden test case dicts.
    """
    cases: list[dict] = []
    case_id = 1

    for topic, chunk in kb.items():
        print(f"\n📖 Topic: '{topic}'")
        print(f"   Chunk: {chunk[:80]}...")

        questions = generate_questions(chunk, n=variations)
        if not questions:
            print("   ⏭️  Skipping — no questions generated.")
            continue

        for question in questions:
            print(f"   ❓ {question}")
            time.sleep(rate_limit_delay)  # Respect free-tier RPM

            expected = generate_expected_answer(chunk, question)
            if not expected:
                continue

            print(f"   ✅ {expected[:60]}...")

            cases.append(
                {
                    "id": f"SYN-{case_id:03d}",
                    "category": topic,
                    "input": question,
                    "expected_output": expected,
                    "context": [chunk],
                    "notes": f"Synthesized from KB topic '{topic}'.",
                    "thresholds": {"faithfulness": 0.80, "relevancy": 0.70},
                    "synthesized": True,
                }
            )
            case_id += 1
            time.sleep(rate_limit_delay)

    return cases


# ── Dataset loader ────────────────────────────────────────────────
def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a golden dataset from the knowledge base."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Path to existing JSON dataset to augment (optional).",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=2,
        help="Question variations per KB topic (default: 2).",
    )
    parser.add_argument(
        "--output",
        default="data/golden_dataset_synthesized.json",
        help="Output file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results to stdout without saving.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧬 DeepEval Golden Dataset Synthesizer")
    print(f"   Model    : gemini-1.5-flash")
    print(f"   Topics   : {len(MOCK_KNOWLEDGE_BASE)}")
    print(f"   Variations: {args.variations} per topic")
    print(f"   Expected  : ~{len(MOCK_KNOWLEDGE_BASE) * args.variations} cases")
    print("=" * 60)

    # Start with existing dataset if provided
    existing: list[dict] = []
    if args.source:
        existing = load_dataset(args.source)
        print(f"\n📂 Loaded {len(existing)} existing cases from {args.source}")

    # Synthesize new cases
    print("\n🔄 Synthesizing new cases from MOCK_KNOWLEDGE_BASE...")
    new_cases = synthesize_from_kb(
        MOCK_KNOWLEDGE_BASE,
        variations=args.variations,
    )

    combined = existing + new_cases

    print(f"\n{'=' * 60}")
    print(f"✨ Synthesis complete!")
    print(f"   Existing cases : {len(existing)}")
    print(f"   New cases      : {len(new_cases)}")
    print(f"   Total          : {len(combined)}")
    print("=" * 60)

    if args.dry_run:
        print("\n📋 DRY RUN — output (not saved):")
        print(json.dumps(combined, indent=2))
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to {output_path}")
    print("\nNext step:")
    print(f"  pytest tests/ -m smoke  (uses the synthesized dataset)")


if __name__ == "__main__":
    main()
