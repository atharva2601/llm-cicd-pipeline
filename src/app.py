"""
app.py — Production Application Entry Point
────────────────────────────────────────────
In a real deployment this would be a FastAPI / Flask endpoint.
For the purposes of this project it is a simple CLI so you can
manually interact with the RAG pipeline before running CI tests.

Run:
    python -m src.app "How do I get a refund?"
"""

from __future__ import annotations

import sys

from src.mock_rag import MockRAGPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.app \"<your question>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    pipeline = MockRAGPipeline()

    print(f"\n{'─' * 60}")
    print(f"Query   : {query}")
    print(f"{'─' * 60}")

    answer, context = pipeline.answer(query)

    print("Context :")
    for i, chunk in enumerate(context, 1):
        print(f"  [{i}] {chunk}")

    print(f"\nAnswer  : {answer}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
