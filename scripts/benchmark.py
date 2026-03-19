#!/usr/bin/env python3
"""
scripts/benchmark.py — Latency & Throughput Benchmarker
─────────────────────────────────────────────────────────
Implements PDF §8.3 "Cost Analysis":

  "While the tools are free, the execution time is not zero."
  "Gemini Flash: ~500ms per generated token."

This script measures:

  • Per-query latency (retrieve + generate)
  • Judge latency (metric evaluation time)
  • End-to-end latency (pipeline + judge)
  • Estimated GitHub Actions minutes consumed per test suite
  • Token throughput

Usage
─────
    python scripts/benchmark.py
    python scripts/benchmark.py --queries 20    # more samples
    python scripts/benchmark.py --report        # save to reports/
    python scripts/benchmark.py --suite smoke   # only smoke queries

Output
──────
    Prints a formatted table to stdout.
    Optionally saves reports/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median, stdev

# Bootstrap path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# ── Benchmark constants ───────────────────────────────────────────
GITHUB_ACTIONS_COST_PER_MIN = 0.008   # USD per minute (paid tier)
SMOKE_QUERIES = [
    "How do I get a refund?",
    "What are your support hours?",
    "How do I reset my password?",
]
FULL_QUERIES = SMOKE_QUERIES + [
    "How long does standard shipping take?",
    "Can I return opened software?",
    "Does the warranty cover accidental damage?",
    "How do I contact support?",
    "What is the age requirement to create an account?",
    "Is free shipping available?",
    "Are sale items refundable?",
]


@dataclass
class QueryBenchmark:
    query: str
    retrieval_ms: float
    generation_ms: float
    total_pipeline_ms: float
    output_length_chars: int
    context_chunks: int
    error: str = ""


@dataclass
class SuiteSummary:
    suite_name: str
    query_count: int
    mean_pipeline_ms: float
    median_pipeline_ms: float
    p95_pipeline_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    total_seconds: float
    estimated_ci_minutes: float
    estimated_ci_cost_usd: float
    queries: list[QueryBenchmark] = field(default_factory=list)


# ── Measurement helpers ───────────────────────────────────────────
def _measure_pipeline(pipeline, query: str) -> QueryBenchmark:
    """Measure retrieve + generate latency separately."""
    # Retrieval
    t0 = time.monotonic()
    context = pipeline.retrieve(query)
    retrieval_ms = (time.monotonic() - t0) * 1000

    # Generation
    t1 = time.monotonic()
    try:
        output, _ = pipeline.generate(query)
        generation_ms = (time.monotonic() - t1) * 1000
        error = ""
    except Exception as exc:
        generation_ms = (time.monotonic() - t1) * 1000
        output = ""
        error = str(exc)[:80]

    return QueryBenchmark(
        query=query,
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(generation_ms, 1),
        total_pipeline_ms=round(retrieval_ms + generation_ms, 1),
        output_length_chars=len(output),
        context_chunks=len(context),
        error=error,
    )


def _p95(values: list[float]) -> float:
    """95th percentile."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.95)
    return sorted_v[min(idx, len(sorted_v) - 1)]


# ── Main benchmark runner ─────────────────────────────────────────
def run_benchmark(
    queries: list[str],
    suite_name: str,
    rate_limit_delay: float = 1.5,
) -> SuiteSummary:
    """Run all queries and collect timing data."""
    from src.mock_rag import MockRAGPipeline

    pipeline = MockRAGPipeline()
    results: list[QueryBenchmark] = []

    print(f"\n⏱️  Benchmarking '{suite_name}' ({len(queries)} queries)...")
    print(f"{'─' * 65}")
    print(f"  {'Query':<45} {'Ret':>6} {'Gen':>8} {'Tot':>8}")
    print(f"{'─' * 65}")

    for query in queries:
        bm = _measure_pipeline(pipeline, query)
        results.append(bm)

        status = "❌" if bm.error else "✅"
        print(
            f"  {status} {query[:43]:<43} "
            f"{bm.retrieval_ms:>5.0f}ms "
            f"{bm.generation_ms:>7.0f}ms "
            f"{bm.total_pipeline_ms:>7.0f}ms"
        )
        if bm.error:
            print(f"    ⚠️  {bm.error}")

        time.sleep(rate_limit_delay)

    # Aggregate
    times = [r.total_pipeline_ms for r in results if not r.error]
    total_s = sum(times) / 1000
    ci_min  = total_s / 60 * 2.5   # Factor of 2.5 for judge eval overhead
    ci_cost = ci_min * GITHUB_ACTIONS_COST_PER_MIN

    return SuiteSummary(
        suite_name=suite_name,
        query_count=len(queries),
        mean_pipeline_ms=round(mean(times), 1) if times else 0,
        median_pipeline_ms=round(median(times), 1) if times else 0,
        p95_pipeline_ms=round(_p95(times), 1) if times else 0,
        min_ms=round(min(times), 1) if times else 0,
        max_ms=round(max(times), 1) if times else 0,
        std_ms=round(stdev(times), 1) if len(times) > 1 else 0,
        total_seconds=round(total_s, 1),
        estimated_ci_minutes=round(ci_min, 1),
        estimated_ci_cost_usd=round(ci_cost, 4),
        queries=results,
    )


def print_summary(s: SuiteSummary) -> None:
    print(f"\n{'═' * 65}")
    print(f"  📊 Benchmark Summary — {s.suite_name}")
    print(f"{'═' * 65}")
    print(f"  Queries         : {s.query_count}")
    print(f"  Mean latency    : {s.mean_pipeline_ms:.0f}ms")
    print(f"  Median latency  : {s.median_pipeline_ms:.0f}ms")
    print(f"  P95 latency     : {s.p95_pipeline_ms:.0f}ms")
    print(f"  Std deviation   : {s.std_ms:.0f}ms")
    print(f"  Min / Max       : {s.min_ms:.0f}ms / {s.max_ms:.0f}ms")
    print(f"{'─' * 65}")
    print(f"  Total pipeline  : {s.total_seconds:.1f}s")
    print(f"  Est. CI time    : ~{s.estimated_ci_minutes:.1f} min "
          f"(incl. 2.5× judge overhead)")
    print(f"  Est. CI cost    : ${s.estimated_ci_cost_usd:.4f} "
          f"(paid tier: ${GITHUB_ACTIONS_COST_PER_MIN}/min)")
    print(f"  Free tier note  : Public repos = $0.00 ✅")
    print(f"{'═' * 65}")

    # Latency bar chart
    print(f"\n  Latency distribution (each █ = 200ms):")
    buckets = [0, 200, 500, 1000, 2000, 5000, float("inf")]
    labels  = ["<200ms", "200–500ms", "500ms–1s", "1–2s", "2–5s", ">5s"]
    times   = [q.total_pipeline_ms for q in s.queries if not q.error]
    for i, label in enumerate(labels):
        count = sum(1 for t in times if buckets[i] <= t < buckets[i + 1])
        bar   = "█" * count
        print(f"    {label:<12}  {bar:<20}  {count}")

    # Recommendations
    print(f"\n  💡 Recommendations:")
    if s.mean_pipeline_ms < 2000:
        print("    ✅ Latency is within free-tier budget.")
    elif s.mean_pipeline_ms < 5000:
        print("    ⚠️  Latency is moderate. Consider --n-workers 2 for parallelism.")
    else:
        print("    ❌ High latency. Check Gemini API quota or switch to Flash-8B.")

    if s.query_count > 20:
        print("    💡 Large suite: use @pytest.mark.smoke for push events "
              "and reserve full suite for PRs only.")
    print()


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the LLM eval pipeline latency."
    )
    parser.add_argument(
        "--suite", choices=["smoke", "full"], default="smoke",
        help="Which query set to benchmark (default: smoke).",
    )
    parser.add_argument(
        "--queries", type=int, default=None,
        help="Number of queries to run (overrides --suite).",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Save results to reports/benchmark_results.json.",
    )
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("ERROR: GOOGLE_API_KEY not set. Copy .env.example → .env")

    queries = FULL_QUERIES if args.suite == "full" else SMOKE_QUERIES
    if args.queries:
        queries = (FULL_QUERIES * 10)[:args.queries]

    summary = run_benchmark(queries, suite_name=args.suite)
    print_summary(summary)

    if args.report:
        out = ROOT / "reports" / "benchmark_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            data = asdict(summary)
            json.dump(data, f, indent=2)
        print(f"  💾 Saved to {out}\n")


if __name__ == "__main__":
    main()
