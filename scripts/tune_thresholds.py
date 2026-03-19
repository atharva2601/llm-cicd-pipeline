#!/usr/bin/env python3
"""
scripts/tune_thresholds.py — Statistical Threshold Optimiser
──────────────────────────────────────────────────────────────
Implements the arXiv-referenced approach from PDF §5.2:
"utilising statistical methods to identify optimal thresholds
based on historical performance."

How it works
────────────
1. Runs a sample of test cases (default: smoke suite)
2. Collects per-metric raw scores from DeepEval
3. Fits a distribution to the scores
4. Recommends a threshold at the Nth percentile (default: 10th)
   so that only the bottom N% of historically passing answers fail

This is better than a fixed "industry baseline" of 0.80 because:
  • Your specific KB and queries may produce systematically higher
    or lower scores than the generic baseline
  • You avoid "threshold drift" — a threshold that was right at
    project launch may be wrong after 6 months of data

Usage
─────
    python scripts/tune_thresholds.py
    python scripts/tune_thresholds.py --percentile 5   # more lenient
    python scripts/tune_thresholds.py --percentile 15  # stricter
    python scripts/tune_thresholds.py --apply          # write to config

Output
──────
    reports/threshold_recommendations.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev, quantiles

# Bootstrap path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


# ── Scoring helpers ───────────────────────────────────────────────
def _collect_scores(queries: list[str]) -> dict[str, list[float]]:
    """
    Run a sample of queries through the pipeline and the judge,
    collecting raw metric scores (not just pass/fail).

    Returns a dict of metric_name → list[float].
    """
    import google.generativeai as genai
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase

    from src.mock_rag import MockRAGPipeline

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not set.")

    judge = GeminiModel(
        model_name="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.0,
    )
    pipeline = MockRAGPipeline()

    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "relevancy": [],
    }

    print(f"\n📐 Collecting scores from {len(queries)} sample queries...")
    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:60]}")
        try:
            actual_output, retrieval_context = pipeline.answer(query)

            tc = LLMTestCase(
                input=query,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )

            # Faithfulness
            fm = FaithfulnessMetric(
                threshold=0.0,   # 0.0 = never fail, just collect scores
                model=judge,
                async_mode=False,
            )
            fm.measure(tc)
            if fm.score is not None:
                scores["faithfulness"].append(fm.score)

            time.sleep(1.5)   # Rate limit

            # Relevancy
            rm = AnswerRelevancyMetric(
                threshold=0.0,
                model=judge,
                async_mode=False,
            )
            rm.measure(tc)
            if rm.score is not None:
                scores["relevancy"].append(rm.score)

            time.sleep(1.5)

        except Exception as exc:
            print(f"  ⚠️  Skipping ({exc})")

    return scores


def _recommend_threshold(
    scores: list[float],
    percentile: int = 10,
) -> dict:
    """
    Given a list of raw scores, compute statistics and recommend
    a threshold at the given percentile.
    """
    if len(scores) < 3:
        return {
            "samples":     len(scores),
            "recommended": 0.80,
            "note":        "Too few samples — using industry default 0.80",
        }

    n = len(scores)
    avg = mean(scores)
    med = median(scores)
    sd  = stdev(scores) if n > 1 else 0.0
    mn  = min(scores)
    mx  = max(scores)

    # Percentile via quantiles (Python ≥ 3.8)
    q = quantiles(scores, n=100, method="inclusive")
    pct_value = q[max(0, percentile - 1)]

    # Round DOWN to nearest 0.05 for a clean threshold
    recommended = round(pct_value * 20) / 20  # snap to 0.05 grid

    return {
        "samples":     n,
        "mean":        round(avg, 4),
        "median":      round(med, 4),
        "stdev":       round(sd,  4),
        "min":         round(mn,  4),
        "max":         round(mx,  4),
        f"p{percentile}": round(pct_value, 4),
        "recommended": recommended,
        "note": (
            f"Threshold set at {percentile}th percentile ({pct_value:.3f}) "
            f"rounded to {recommended:.2f}. "
            f"{percentile}% of samples score below this value."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistically tune evaluation thresholds."
    )
    parser.add_argument(
        "--percentile", type=int, default=10,
        help="Recommend threshold at this percentile (default: 10).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write recommendations to reports/threshold_recommendations.json.",
    )
    args = parser.parse_args()

    # Sample queries (extended set for better statistics)
    sample_queries = [
        "How do I get a refund?",
        "What are your support hours?",
        "How do I reset my password?",
        "How long does shipping take?",
        "Can I return opened software?",
        "Does the warranty cover accidental damage?",
        "How do I contact support?",
        "What is the age requirement to create an account?",
        "Is free shipping available?",
        "Are sale items refundable?",
    ]

    print("\n🎛️  Threshold Tuning Tool")
    print(f"   Percentile  : {args.percentile}th")
    print(f"   Queries     : {len(sample_queries)}")
    print("=" * 55)

    scores = _collect_scores(sample_queries)

    print("\n📊 Analysis")
    print("=" * 55)

    recommendations: dict[str, dict] = {}
    for metric, raw in scores.items():
        if not raw:
            print(f"  {metric}: no data collected")
            continue
        rec = _recommend_threshold(raw, percentile=args.percentile)
        recommendations[metric] = rec

        print(f"\n  {metric.upper()}")
        print(f"    Samples    : {rec['samples']}")
        print(f"    Mean       : {rec.get('mean', 'N/A')}")
        print(f"    Std Dev    : {rec.get('stdev', 'N/A')}")
        print(f"    Min / Max  : {rec.get('min', 'N/A')} / {rec.get('max', 'N/A')}")
        print(f"    P{args.percentile:<2}       : {rec.get(f'p{args.percentile}', 'N/A')}")
        print(f"    ✅ Recommended threshold: {rec['recommended']}")
        print(f"    💬 {rec['note']}")

    print("\n" + "=" * 55)
    print("📝 Suggested updates to your test files:")
    for metric, rec in recommendations.items():
        const = "FAITHFULNESS_THRESHOLD" if metric == "faithfulness" else "RELEVANCY_THRESHOLD"
        print(f"   {const} = {rec['recommended']}")

    if args.apply:
        out_path = ROOT / "reports" / "threshold_recommendations.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "percentile": args.percentile,
                "query_count": len(sample_queries),
                "recommendations": recommendations,
            }, f, indent=2)
        print(f"\n💾 Saved to {out_path}")

    print()


if __name__ == "__main__":
    main()
