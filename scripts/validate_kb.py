#!/usr/bin/env python3
"""
scripts/validate_kb.py — Knowledge Base Coverage Validator
────────────────────────────────────────────────────────────
Compares the knowledge base (src/mock_rag.py MOCK_KNOWLEDGE_BASE)
against the golden dataset (data/golden_dataset.json) and reports:

  • Which KB topics have NO test coverage
  • Which golden dataset cases have NO matching KB topic
  • Coverage percentage per topic
  • Duplicate question detection

This prevents the "silent gap" failure mode where a new KB topic
gets added but no eval test is written for it, so regressions
on that topic would never be caught.

Usage
─────
    python scripts/validate_kb.py
    python scripts/validate_kb.py --strict    # Exit 1 if any gap found
    python scripts/validate_kb.py --report    # Write reports/kb_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Bootstrap path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mock_rag import MOCK_KNOWLEDGE_BASE

ROOT        = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden_dataset.json"
SYNTH_PATH  = ROOT / "data" / "golden_dataset_synthesized.json"


# ── Load dataset ─────────────────────────────────────────────────
def load_all_cases() -> list[dict]:
    cases: dict[str, dict] = {}
    for path in [GOLDEN_PATH, SYNTH_PATH]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for entry in json.load(f):
                    cases[entry["id"]] = entry
    return list(cases.values())


# ── Coverage analysis ─────────────────────────────────────────────
def analyse_coverage(
    kb: dict[str, str],
    cases: list[dict],
) -> dict:
    kb_topics = set(kb.keys())
    dataset_topics: dict[str, list[str]] = defaultdict(list)

    for case in cases:
        topic = case.get("category", "unknown")
        dataset_topics[topic].append(case["id"])

    covered     = kb_topics & set(dataset_topics.keys())
    uncovered   = kb_topics - set(dataset_topics.keys())
    orphaned    = set(dataset_topics.keys()) - kb_topics - {"safety", "unknown"}

    # Duplicate question detection
    questions: list[str] = [c["input"].lower().strip() for c in cases]
    duplicates: list[str] = [q for q in set(questions) if questions.count(q) > 1]

    coverage_pct = (len(covered) / len(kb_topics) * 100) if kb_topics else 0

    return {
        "total_kb_topics":       len(kb_topics),
        "covered_topics":        sorted(covered),
        "uncovered_topics":      sorted(uncovered),
        "orphaned_dataset_topics": sorted(orphaned),
        "coverage_pct":          round(coverage_pct, 1),
        "total_cases":           len(cases),
        "standard_cases":        len([c for c in cases if not c.get("adversarial")]),
        "adversarial_cases":     len([c for c in cases if c.get("adversarial")]),
        "cases_per_topic":       {
            k: len(v) for k, v in sorted(dataset_topics.items())
        },
        "duplicate_questions":   duplicates,
        "is_fully_covered":      len(uncovered) == 0,
        "has_orphans":           len(orphaned) > 0,
        "has_duplicates":        len(duplicates) > 0,
    }


# ── Printing ──────────────────────────────────────────────────────
def print_report(analysis: dict, kb: dict[str, str]) -> None:
    pct   = analysis["coverage_pct"]
    bar_w = 30
    fill  = int(bar_w * pct / 100)
    bar   = "█" * fill + "░" * (bar_w - fill)

    print(f"\n{'=' * 60}")
    print(f"  📊 Knowledge Base Coverage Report")
    print(f"{'=' * 60}")
    print(f"  Coverage : [{bar}] {pct:.1f}%")
    print(f"  KB topics: {analysis['total_kb_topics']}")
    print(f"  Test cases: {analysis['total_cases']} "
          f"({analysis['standard_cases']} standard, "
          f"{analysis['adversarial_cases']} adversarial)")
    print()

    # Covered topics
    print("  ✅ Covered topics:")
    for topic in analysis["covered_topics"]:
        count = analysis["cases_per_topic"].get(topic, 0)
        snippet = kb[topic][:50].replace("\n", " ")
        print(f"     {topic:<18} {count} case(s)   \"{snippet}...\"")

    # Uncovered topics
    if analysis["uncovered_topics"]:
        print()
        print("  ❌ UNCOVERED topics (no test cases):")
        for topic in analysis["uncovered_topics"]:
            snippet = kb[topic][:60].replace("\n", " ")
            print(f"     {topic:<18} \"{snippet}...\"")
        print()
        print("  👉 Fix: Add test cases or run:  make synthesize")

    # Orphaned dataset categories
    if analysis["orphaned_dataset_topics"]:
        print()
        print("  ⚠️  Dataset categories with no matching KB topic:")
        for topic in analysis["orphaned_dataset_topics"]:
            count = analysis["cases_per_topic"].get(topic, 0)
            print(f"     {topic:<18} {count} case(s)")
        print("  👉 Fix: Add the topic to MOCK_KNOWLEDGE_BASE in src/mock_rag.py")

    # Duplicate questions
    if analysis["duplicate_questions"]:
        print()
        print("  ⚠️  Duplicate questions detected:")
        for q in analysis["duplicate_questions"]:
            print(f"     \"{q[:70]}\"")
        print("  👉 Fix: Remove duplicate entries from data/golden_dataset.json")

    print(f"\n{'=' * 60}")

    if analysis["is_fully_covered"] and not analysis["has_duplicates"]:
        print("  🎉 Dataset is complete — all KB topics have test coverage.")
    else:
        issues: list[str] = []
        if analysis["uncovered_topics"]:
            issues.append(f"{len(analysis['uncovered_topics'])} uncovered topic(s)")
        if analysis["has_duplicates"]:
            issues.append(f"{len(analysis['duplicate_questions'])} duplicate question(s)")
        print(f"  ⚠️  Issues found: {', '.join(issues)}")
    print()


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate knowledge base coverage against the golden dataset."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any coverage gap or duplicate is found.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write reports/kb_coverage.json.",
    )
    args = parser.parse_args()

    cases    = load_all_cases()
    analysis = analyse_coverage(MOCK_KNOWLEDGE_BASE, cases)

    print_report(analysis, MOCK_KNOWLEDGE_BASE)

    if args.report:
        report_path = ROOT / "reports" / "kb_coverage.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"  💾 Report saved to {report_path}\n")

    if args.strict:
        has_issues = (
            bool(analysis["uncovered_topics"])
            or analysis["has_duplicates"]
            or analysis["has_orphans"]
        )
        sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
