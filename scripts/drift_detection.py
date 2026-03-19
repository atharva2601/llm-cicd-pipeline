#!/usr/bin/env python3
"""
scripts/drift_detection.py — Metric Drift Logger & Analyser
─────────────────────────────────────────────────────────────
Implements PDF §8.1 "Drift Detection".

Every CI run writes its metric scores to reports/drift_log.jsonl.
This script reads the log and surfaces:

  • Metric averages per run
  • Trend over time (rising / stable / falling)
  • ALERT if any metric has dropped > DRIFT_THRESHOLD from its
    30-day rolling average

Usage
─────
  # Log a completed test run (called by the CI workflow automatically)
  python scripts/drift_detection.py log \
      --run-id "$GITHUB_RUN_NUMBER" \
      --commit "$GITHUB_SHA" \
      --results path/to/results.json

  # Analyse drift on the last N runs
  python scripts/drift_detection.py analyse --window 30

  # Print a formatted drift report
  python scripts/drift_detection.py report

Exit codes
──────────
  0 — No drift detected
  1 — Drift detected (use this to optionally block CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# ── Constants ────────────────────────────────────────────────────
DRIFT_LOG_PATH   = Path("reports/drift_log.jsonl")
DRIFT_THRESHOLD  = 0.10   # Alert if metric drops > 10 pts from baseline
WINDOW_SIZE      = 30     # Number of runs to include in rolling average
METRICS_TRACKED  = ["faithfulness", "relevancy", "contextual_recall",
                     "contextual_precision", "safety"]


# ── Log entry structure ───────────────────────────────────────────
def make_log_entry(
    run_id: str,
    commit: str,
    scores: dict[str, float],
    branch: str = "unknown",
) -> dict[str, Any]:
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "run_id":     run_id,
        "commit":     commit[:8] if len(commit) > 8 else commit,
        "branch":     branch,
        "scores":     scores,
    }


# ── I/O helpers ───────────────────────────────────────────────────
def _read_log() -> list[dict]:
    if not DRIFT_LOG_PATH.exists():
        return []
    entries: list[dict] = []
    with open(DRIFT_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _append_log(entry: dict) -> None:
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_results_file(path: str) -> dict[str, float]:
    """
    Parses a results JSON file produced by DeepEval or pytest-json-report.
    Expected format (simplified):
      {
        "faithfulness_avg": 0.92,
        "relevancy_avg": 0.87,
        ...
      }
    Falls back to empty dict on error.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Support flat or nested structures
        scores: dict[str, float] = {}
        for metric in METRICS_TRACKED:
            for key in [metric, f"{metric}_avg", f"avg_{metric}"]:
                if key in data:
                    scores[metric] = float(data[key])
                    break
        return scores
    except Exception as exc:
        print(f"⚠️  Could not parse results file: {exc}")
        return {}


# ── Commands ──────────────────────────────────────────────────────
def cmd_log(args: argparse.Namespace) -> int:
    """Append a new run's scores to the drift log."""
    scores: dict[str, float] = {}

    if args.results:
        scores = _load_results_file(args.results)

    # Allow manual score overrides via --score metric=value
    if hasattr(args, "score") and args.score:
        for item in args.score:
            k, _, v = item.partition("=")
            scores[k.strip()] = float(v.strip())

    if not scores:
        print("⚠️  No scores provided. Use --results or --score metric=value.")
        return 1

    entry = make_log_entry(
        run_id=args.run_id,
        commit=args.commit,
        scores=scores,
        branch=getattr(args, "branch", "unknown"),
    )
    _append_log(entry)

    print(f"✅ Logged run {args.run_id}")
    for metric, score in scores.items():
        print(f"   {metric:<25} {score:.4f}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """
    Analyse drift over the last N runs.
    Returns exit code 1 if drift is detected.
    """
    entries = _read_log()
    if not entries:
        print("📊 No drift log entries found. Run a CI pipeline first.")
        return 0

    window = min(args.window, len(entries))
    recent = entries[-window:]

    print(f"\n📊 Drift Analysis — last {window} runs")
    print("=" * 55)

    drift_detected = False

    for metric in METRICS_TRACKED:
        values = [
            e["scores"][metric]
            for e in recent
            if metric in e.get("scores", {})
        ]
        if len(values) < 2:
            continue

        avg      = mean(values)
        latest   = values[-1]
        baseline = mean(values[:-1])  # All runs except the latest
        delta    = latest - baseline
        std      = stdev(values) if len(values) > 1 else 0.0

        trend = "📈" if delta > 0.02 else "📉" if delta < -0.02 else "➡️ "
        alert = " ⚠️  DRIFT ALERT" if delta < -DRIFT_THRESHOLD else ""

        if delta < -DRIFT_THRESHOLD:
            drift_detected = True

        print(
            f"  {trend}  {metric:<22} "
            f"latest={latest:.3f}  "
            f"avg={avg:.3f}  "
            f"Δ={delta:+.3f}  "
            f"σ={std:.3f}"
            f"{alert}"
        )

    print("=" * 55)

    if drift_detected:
        print(
            f"\n🚨 DRIFT DETECTED — one or more metrics dropped >{DRIFT_THRESHOLD:.0%} "
            f"from the {window}-run baseline."
        )
        print("   Possible causes:")
        print("   • Prompt was modified and degraded generation quality")
        print("   • Knowledge base content was changed")
        print("   • Gemini model behaviour shifted (model drift)")
        print("   Action: Review the last prompt/KB change and re-evaluate.\n")
        return 1

    print("\n✅ No significant drift detected.\n")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print a human-readable table of all logged runs."""
    entries = _read_log()
    if not entries:
        print("📊 No drift log entries found.")
        return 0

    # Header
    metrics_in_log = sorted({
        m for e in entries for m in e.get("scores", {})
    })
    col_w = 12
    header = f"{'Run':<8} {'Commit':<10} {'Date':<12} " + \
             "".join(f"{m[:col_w]:<{col_w}}" for m in metrics_in_log)
    print("\n" + header)
    print("─" * len(header))

    for e in entries[-50:]:   # Show last 50 rows
        date = e.get("timestamp", "")[:10]
        row = (
            f"{str(e.get('run_id','?')):<8} "
            f"{e.get('commit','?'):<10} "
            f"{date:<12} "
        )
        for m in metrics_in_log:
            val = e.get("scores", {}).get(m)
            cell = f"{val:.3f}" if val is not None else "  —  "
            row += f"{cell:<{col_w}}"
        print(row)

    print()
    return 0


# ── CLI ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM metric drift detection tool."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # log
    p_log = sub.add_parser("log", help="Append a run's scores to the log.")
    p_log.add_argument("--run-id",  required=True)
    p_log.add_argument("--commit",  required=True)
    p_log.add_argument("--branch",  default="unknown")
    p_log.add_argument("--results", default=None,
                       help="Path to results JSON file.")
    p_log.add_argument("--score",   nargs="*",
                       help="Manual scores, e.g. faithfulness=0.91")

    # analyse
    p_analyse = sub.add_parser("analyse", help="Analyse drift over last N runs.")
    p_analyse.add_argument("--window", type=int, default=WINDOW_SIZE)

    # report
    sub.add_parser("report", help="Print a formatted run history table.")

    args = parser.parse_args()

    dispatch = {
        "log":     cmd_log,
        "analyse": cmd_analyse,
        "report":  cmd_report,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
