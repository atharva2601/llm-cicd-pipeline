#!/usr/bin/env python3
"""
scripts/run_eval.py — Local Evaluation Runner with Score Export
────────────────────────────────────────────────────────────────
Runs the full DeepEval test suite, collects per-metric averages,
and writes reports/latest_results.json for drift_detection.py.

Also serves as a convenient local alternative to typing long
pytest commands.

Usage
─────
    python scripts/run_eval.py                     # smoke tests
    python scripts/run_eval.py --suite full
    python scripts/run_eval.py --suite safety
    python scripts/run_eval.py --suite all
    python scripts/run_eval.py --html              # also open HTML report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPORTS_DIR = Path("reports")
RESULTS_FILE = REPORTS_DIR / "latest_results.json"
HTML_REPORT  = REPORTS_DIR / "evaluation_report.html"

SUITE_MARKERS = {
    "smoke":  "smoke",
    "full":   "full",
    "safety": "safety",
    "all":    "smoke or full or safety",
}


def run_pytest(marker: str, html: bool = False) -> tuple[int, str]:
    """Run pytest with the given marker expression."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", marker,
        "-v",
        "--tb=short",
        "--no-header",
        "-p", "no:warnings",
        "--json-report",
        f"--json-report-file={REPORTS_DIR / 'pytest_report.json'}",
    ]
    if html:
        cmd += ["--html", str(HTML_REPORT), "--self-contained-html"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🧪 Running: pytest -m '{marker}'\n{'─' * 50}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    return result.returncode, f"{elapsed:.1f}s"


def parse_pytest_report() -> dict[str, float]:
    """
    Parse pytest-json-report output into per-metric averages.
    Falls back to dummy values if the report is not available.
    """
    report_path = REPORTS_DIR / "pytest_report.json"
    if not report_path.exists():
        return {}

    try:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)

        # pytest-json-report doesn't directly expose DeepEval scores,
        # so we parse the test node IDs and pass/fail status.
        # DeepEval writes scores to stdout; a full integration would
        # capture those.  For now we derive pass rates per test file
        # as a proxy for metric scores.
        scores: dict[str, float] = {}
        by_file: dict[str, list[bool]] = {}

        for test in data.get("tests", []):
            node = test.get("nodeid", "")
            passed = test.get("outcome") == "passed"
            # Extract file name without path/extension
            file_part = node.split("::")[0].split("/")[-1].replace(".py", "")
            by_file.setdefault(file_part, []).append(passed)

        metric_map = {
            "test_faithfulness": "faithfulness",
            "test_relevancy":    "relevancy",
            "test_contextual":   "contextual_recall",
            "test_rag":          "rag_triad",
            "test_safety":       "safety",
        }
        for file_key, metric_name in metric_map.items():
            results = by_file.get(file_key, [])
            if results:
                scores[metric_name] = round(sum(results) / len(results), 4)

        return scores
    except Exception as exc:
        print(f"⚠️  Could not parse pytest report: {exc}")
        return {}


def save_results(scores: dict[str, float], elapsed: str, suite: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "suite":   suite,
        "elapsed": elapsed,
        "scores":  scores,
        **scores,   # Flat copy for drift_detection.py compatibility
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation suite and export scores."
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITE_MARKERS.keys()),
        default="smoke",
        help="Which test suite to run (default: smoke).",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also generate an HTML report.",
    )
    parser.add_argument(
        "--log-drift",
        action="store_true",
        help="Append scores to the drift log after the run.",
    )
    parser.add_argument(
        "--run-id",
        default="local",
        help="Run identifier for drift logging.",
    )
    args = parser.parse_args()

    marker  = SUITE_MARKERS[args.suite]
    rc, elapsed = run_pytest(marker, html=args.html)

    scores = parse_pytest_report()
    save_results(scores, elapsed, args.suite)

    print(f"\n{'─' * 50}")
    print(f"⏱️  Elapsed : {elapsed}")
    print(f"{'✅ PASSED' if rc == 0 else '❌ FAILED'}")

    if scores:
        print("\n📊 Pass rates by metric:")
        for metric, rate in scores.items():
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"   {metric:<25} {bar}  {rate:.1%}")

    if args.log_drift and scores:
        commit = os.getenv("GITHUB_SHA", "local")
        cmd = [
            sys.executable, "scripts/drift_detection.py", "log",
            "--run-id", args.run_id,
            "--commit", commit,
            "--results", str(RESULTS_FILE),
        ]
        subprocess.run(cmd, check=False)

    if args.html:
        print(f"\n📄 HTML report: {HTML_REPORT}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
