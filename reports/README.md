# reports/

This folder is populated at runtime by the evaluation pipeline.

## Files generated here

| File | Created by | Description |
|---|---|---|
| `drift_log.jsonl` | `scripts/drift_detection.py log` | Append-only metric history per CI run |
| `latest_results.json` | `scripts/run_eval.py` | Last run's scores (faithfulness, relevancy, etc.) |
| `evaluation_report.html` | `deepeval test run` / CI workflow | Full HTML evaluation report |
| `evaluation_report.md` | `scripts/export_report.py --format markdown` | Markdown report for GitHub |
| `evaluation_report.csv` | `scripts/export_report.py --format csv` | CSV for spreadsheets |
| `benchmark_results.json` | `scripts/benchmark.py --report` | Latency benchmark output |
| `kb_coverage.json` | `scripts/validate_kb.py --report` | KB coverage analysis |
| `threshold_recommendations.json` | `scripts/tune_thresholds.py --apply` | Statistical threshold tuning output |
| `pytest_report.json` | `pytest --json-report` | Raw pytest results |

## Quick commands

```bash
# Generate all reports after a test run
make export

# View drift across last 30 CI runs
make drift

# Print run history table
make report
```

## Notes

- `drift_log.jsonl` is uploaded as a GitHub Actions artifact after every nightly run
  so drift history persists across CI runners.
- All files in this folder are gitignored (see `.gitignore`) except this README.
- The GitHub Actions workflow uploads `evaluation-report.html` as a downloadable
  artifact — find it under Actions → your run → Artifacts.
