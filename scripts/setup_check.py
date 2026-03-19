#!/usr/bin/env python3
"""
scripts/setup_check.py — Pre-flight Environment Validator
───────────────────────────────────────────────────────────
Run this BEFORE your first `make smoke` or CI run.
It validates every dependency and configuration step so you
get a clear, actionable error message instead of a cryptic
stack trace inside pytest.

Usage
─────
    python scripts/setup_check.py           # full check
    python scripts/setup_check.py --fix     # attempt auto-fix
    python scripts/setup_check.py --json    # machine-readable output

Exit codes
──────────
    0  — All checks passed
    1  — One or more checks failed
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Bootstrap path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    "deepeval",
    "google.generativeai",
    "dotenv",
    "pytest",
]

REQUIRED_FILES = [
    ".env",
    "data/golden_dataset.json",
    "src/mock_rag.py",
    "tests/conftest.py",
]

MIN_PYTHON = (3, 10)


# ─────────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fix_hint: str = ""
    auto_fix: Callable | None = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────
def check_python_version() -> CheckResult:
    ver = sys.version_info
    passed = (ver.major, ver.minor) >= MIN_PYTHON
    return CheckResult(
        name="Python version",
        passed=passed,
        message=f"Python {ver.major}.{ver.minor}.{ver.micro}",
        fix_hint=f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ "
                 f"from https://python.org",
    )


def check_package(pkg: str) -> CheckResult:
    try:
        importlib.import_module(pkg)
        return CheckResult(
            name=f"Package: {pkg}",
            passed=True,
            message="installed",
        )
    except ImportError:
        install_name = pkg.replace(".", "-").replace("google-generativeai", "google-generativeai")
        # Map import name → pip name
        pip_map = {
            "google.generativeai": "google-generativeai",
            "dotenv": "python-dotenv",
        }
        pip_name = pip_map.get(pkg, pkg)
        return CheckResult(
            name=f"Package: {pkg}",
            passed=False,
            message=f"NOT installed",
            fix_hint=f"pip install {pip_name}",
            auto_fix=lambda p=pip_name: subprocess.run(
                [sys.executable, "-m", "pip", "install", p], check=False
            ),
        )


def check_required_file(rel_path: str) -> CheckResult:
    root = Path(__file__).resolve().parents[1]
    full = root / rel_path
    return CheckResult(
        name=f"File: {rel_path}",
        passed=full.exists(),
        message="exists" if full.exists() else "MISSING",
        fix_hint=(
            "Copy .env.example → .env and add GOOGLE_API_KEY"
            if rel_path == ".env"
            else f"Restore {rel_path} from git"
        ),
    )


def check_api_key() -> CheckResult:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return CheckResult(
            name="GOOGLE_API_KEY",
            passed=False,
            message="NOT SET",
            fix_hint="Add GOOGLE_API_KEY=your_key to .env",
        )
    masked = key[:6] + "..." + key[-4:] if len(key) > 12 else "****"
    return CheckResult(
        name="GOOGLE_API_KEY",
        passed=True,
        message=f"set ({masked})",
    )


def check_api_connectivity() -> CheckResult:
    """Attempt a minimal Gemini API call to verify credentials."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return CheckResult(
            name="Gemini API connectivity",
            passed=False,
            message="Skipped — no API key",
        )
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(
            "Reply with the single word: OK",
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, max_output_tokens=5
            ),
        )
        reply = resp.text.strip()
        return CheckResult(
            name="Gemini API connectivity",
            passed=True,
            message=f"Connected — model replied: '{reply}'",
        )
    except Exception as exc:
        return CheckResult(
            name="Gemini API connectivity",
            passed=False,
            message=f"FAILED: {exc}",
            fix_hint=(
                "Check your GOOGLE_API_KEY at "
                "https://aistudio.google.com/app/apikey"
            ),
        )


def check_deepeval_config() -> CheckResult:
    """Verify DeepEval is configured for Gemini (not defaulting to OpenAI)."""
    config_paths = [
        Path.home() / ".deepeval" / "config",
        Path(".deepeval") / "config",
    ]
    found_gemini = False
    for p in config_paths:
        if p.exists():
            content = p.read_text(errors="ignore").lower()
            if "gemini" in content:
                found_gemini = True
                break

    if found_gemini:
        return CheckResult(
            name="DeepEval config",
            passed=True,
            message="Gemini judge configured",
        )
    return CheckResult(
        name="DeepEval config",
        passed=False,
        message="Gemini NOT configured (may default to OpenAI)",
        fix_hint=(
            "Run: deepeval set-gemini "
            '--model="gemini-1.5-flash" '
            '--google-api-key="$GOOGLE_API_KEY"'
        ),
        auto_fix=lambda: _run_deepeval_set_gemini(),
    )


def _run_deepeval_set_gemini() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        print("  ⚠️  Cannot auto-fix: GOOGLE_API_KEY not set.")
        return
    subprocess.run(
        [
            sys.executable, "-m", "deepeval", "set-gemini",
            "--model", "gemini-1.5-flash",
            "--google-api-key", key,
        ],
        check=False,
    )


def check_golden_dataset() -> CheckResult:
    path = Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"
    if not path.exists():
        return CheckResult(
            name="Golden dataset",
            passed=False,
            message="data/golden_dataset.json missing",
            fix_hint="Restore from git or run: make synthesize",
        )
    try:
        with open(path) as f:
            data = json.load(f)
        n = len(data)
        adv = sum(1 for c in data if c.get("adversarial"))
        return CheckResult(
            name="Golden dataset",
            passed=n > 0,
            message=f"{n} cases ({n - adv} standard, {adv} adversarial)",
        )
    except Exception as exc:
        return CheckResult(
            name="Golden dataset",
            passed=False,
            message=f"Parse error: {exc}",
            fix_hint="Fix JSON syntax in data/golden_dataset.json",
        )


def check_github_secret_hint() -> CheckResult:
    """
    Non-blocking informational check: reminds about GitHub secret.
    Always returns passed=True so it never blocks the run.
    """
    ci = os.getenv("CI", "")
    if ci:
        key = os.getenv("GOOGLE_API_KEY", "")
        return CheckResult(
            name="GitHub Secret (CI)",
            passed=bool(key),
            message="GOOGLE_API_KEY present in CI environment"
            if key else "GOOGLE_API_KEY NOT set in CI",
            fix_hint=(
                "Go to: Settings → Secrets and variables → Actions "
                "→ New repository secret → GOOGLE_API_KEY"
            ),
        )
    return CheckResult(
        name="GitHub Secret (CI)",
        passed=True,
        message="(running locally — CI check skipped)",
    )


# ─────────────────────────────────────────────────────────────────
ALL_CHECKS: list[Callable[[], CheckResult]] = [
    check_python_version,
    *[lambda p=pkg: check_package(p) for pkg in REQUIRED_PACKAGES],
    *[lambda f=fp: check_required_file(f) for fp in REQUIRED_FILES],
    check_api_key,
    check_api_connectivity,
    check_deepeval_config,
    check_golden_dataset,
    check_github_secret_hint,
]

ICONS = {True: "✅", False: "❌"}


def run_checks(auto_fix: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    for fn in ALL_CHECKS:
        result = fn()

        status = ICONS[result.passed]
        print(f"  {status}  {result.name:<35} {result.message}")

        if not result.passed:
            if result.fix_hint:
                print(f"      💡 {result.fix_hint}")
            if auto_fix and result.auto_fix:
                print(f"      🔧 Auto-fixing...")
                result.auto_fix()
                result = fn()   # Re-run check
                print(f"      {ICONS[result.passed]} After fix: {result.message}")

        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight environment validator for the LLM eval pipeline."
    )
    parser.add_argument("--fix",  action="store_true", help="Attempt auto-fix on failures.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = parser.parse_args()

    if not args.json:
        print("\n🔍 LLM CI/CD Pre-flight Check")
        print("=" * 55)

    results = run_checks(auto_fix=args.fix)

    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if not r.passed]

    if args.json:
        print(json.dumps(
            [{"name": r.name, "passed": r.passed, "message": r.message} for r in results],
            indent=2,
        ))
        sys.exit(0 if not failed else 1)

    print("=" * 55)
    print(f"\n  Passed : {len(passed)}/{len(results)}")

    if failed:
        print(f"  Failed : {len(failed)}")
        print("\n🚫 Fix the issues above before running tests.\n")
        if not args.fix:
            print("  Tip: run with --fix to attempt automatic resolution.\n")
        sys.exit(1)
    else:
        print("\n🎉 All checks passed — you're ready to go!\n")
        print("  Next:  make smoke\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
