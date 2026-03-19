#!/usr/bin/env python3
"""
scripts/export_report.py — Evaluation Report Exporter
───────────────────────────────────────────────────────
Converts the append-only drift_log.jsonl into shareable formats
for engineering leads, product managers, and stakeholders who
don't want to read raw JSON.

Output formats
──────────────
  markdown    — A formatted .md report with tables and trend arrows
  csv         — Raw tabular data for import into spreadsheets
  html        — Self-contained HTML page with inline sparklines
  summary     — A one-paragraph executive summary printed to stdout

Usage
─────
    python scripts/export_report.py                         # stdout summary
    python scripts/export_report.py --format markdown       # save .md
    python scripts/export_report.py --format csv            # save .csv
    python scripts/export_report.py --format html           # save .html
    python scripts/export_report.py --format all            # all formats
    python scripts/export_report.py --window 14             # last 14 runs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT           = Path(__file__).resolve().parents[1]
DRIFT_LOG_PATH = ROOT / "reports" / "drift_log.jsonl"
REPORTS_DIR    = ROOT / "reports"

METRICS = ["faithfulness", "relevancy", "contextual_recall", "safety"]
TREND   = {True: "📈", False: "📉", None: "➡️ "}


# ── Loader ────────────────────────────────────────────────────────
def load_entries(window: int = 0) -> list[dict]:
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
    if window:
        entries = entries[-window:]
    return entries


# ── Trend helper ──────────────────────────────────────────────────
def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "➡️ "
    delta = values[-1] - mean(values[:-1])
    if delta > 0.02:
        return "📈"
    if delta < -0.02:
        return "📉"
    return "➡️ "


def _avg(entries: list[dict], metric: str) -> float | None:
    vals = [e["scores"][metric] for e in entries if metric in e.get("scores", {})]
    return round(mean(vals), 3) if vals else None


# ── Exporters ─────────────────────────────────────────────────────
def export_summary(entries: list[dict]) -> str:
    if not entries:
        return "No drift log entries found. Run CI at least once to generate data."

    n    = len(entries)
    date = entries[-1].get("timestamp", "")[:10]
    run  = entries[-1].get("run_id", "?")

    lines = [f"LLM Evaluation Pipeline — Summary Report"]
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Window: last {n} CI run(s) — latest run #{run} on {date}")
    lines.append("")

    all_good = True
    for metric in METRICS:
        vals = [e["scores"][metric] for e in entries if metric in e.get("scores", {})]
        if not vals:
            continue
        avg = mean(vals)
        latest = vals[-1]
        trend_sym = _trend(vals)
        status = "✅" if latest >= 0.80 else "⚠️ "
        if latest < 0.80:
            all_good = False
        lines.append(
            f"  {status} {trend_sym}  {metric:<22} "
            f"latest={latest:.3f}  avg(n={len(vals)})={avg:.3f}"
        )

    lines.append("")
    if all_good:
        lines.append("Overall: ✅ All tracked metrics are above threshold.")
    else:
        lines.append("Overall: ⚠️  One or more metrics are below 0.80 — review needed.")

    return "\n".join(lines)


def export_markdown(entries: list[dict]) -> str:
    if not entries:
        return "# LLM Evaluation Report\n\nNo data yet."

    lines: list[str] = []
    lines.append("# LLM Evaluation Pipeline — Drift Report")
    lines.append(f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")
    lines.append(f"**Window:** {len(entries)} run(s)\n")

    # Metric summary table
    lines.append("## Metric Summary\n")
    lines.append("| Metric | Latest | Avg | Trend | Status |")
    lines.append("|---|---|---|---|---|")

    for metric in METRICS:
        vals = [e["scores"][metric] for e in entries if metric in e.get("scores", {})]
        if not vals:
            continue
        latest = vals[-1]
        avg    = mean(vals)
        trend_sym = _trend(vals)
        status = "✅ Pass" if latest >= 0.80 else "⚠️ Review"
        lines.append(
            f"| `{metric}` | {latest:.3f} | {avg:.3f} | {trend_sym} | {status} |"
        )

    # Run history table
    lines.append("\n## Run History\n")
    metric_cols = sorted({m for e in entries for m in e.get("scores", {})})
    header = "| Run | Commit | Date | " + " | ".join(metric_cols) + " |"
    separator = "|---|---|---|" + "|---|" * len(metric_cols)
    lines.append(header)
    lines.append(separator)

    for e in entries[-30:]:
        date = e.get("timestamp", "")[:10]
        row  = (
            f"| {e.get('run_id','?')} "
            f"| `{e.get('commit','?')}` "
            f"| {date} "
        )
        for m in metric_cols:
            v = e.get("scores", {}).get(m)
            cell = f"{v:.3f}" if v is not None else "—"
            row += f"| {cell} "
        row += "|"
        lines.append(row)

    # Recommendations
    lines.append("\n## Recommendations\n")
    for metric in METRICS:
        vals = [e["scores"][metric] for e in entries if metric in e.get("scores", {})]
        if not vals:
            continue
        latest = vals[-1]
        if latest < 0.80:
            lines.append(
                f"- ⚠️  **{metric}** is below threshold ({latest:.3f} < 0.80). "
                f"Review recent prompt changes or KB updates."
            )

    return "\n".join(lines)


def export_csv(entries: list[dict]) -> str:
    if not entries:
        return "run_id,commit,timestamp,metric,score\n"

    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_id", "commit", "date", "branch"] + METRICS)

    for e in entries:
        row = [
            e.get("run_id", ""),
            e.get("commit", ""),
            e.get("timestamp", "")[:10],
            e.get("branch", ""),
        ]
        for m in METRICS:
            row.append(e.get("scores", {}).get(m, ""))
        writer.writerow(row)

    return buf.getvalue()


def export_html(entries: list[dict]) -> str:
    """Generate a self-contained HTML report with inline sparklines."""
    md_content = export_markdown(entries)

    # Build sparkline data
    sparklines: dict[str, list[float]] = {m: [] for m in METRICS}
    for e in entries:
        for m in METRICS:
            if m in e.get("scores", {}):
                sparklines[m].append(e["scores"][m])

    spark_js = json.dumps({k: v[-20:] for k, v in sparklines.items()})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1   {{ color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 8px; }}
  h2   {{ color: #333; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th   {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .pass {{ color: #22863a; font-weight: bold; }}
  .warn {{ color: #d93025; font-weight: bold; }}
  .spark {{ display: inline-block; }}
  canvas {{ vertical-align: middle; }}
  pre  {{ background: #f6f8fa; border-radius: 6px; padding: 16px;
          overflow-x: auto; font-size: 13px; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
  .meta {{ color: #666; font-size: 13px; }}
</style>
</head>
<body>
<h1>🧪 LLM Evaluation Pipeline Report</h1>
<p class="meta">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |
Window: {len(entries)} run(s)</p>

<h2>Metric Overview</h2>
<table>
  <tr><th>Metric</th><th>Latest</th><th>Average</th><th>Trend (last 20)</th><th>Status</th></tr>
  {"".join(
      f"<tr>"
      f"<td><code>{m}</code></td>"
      f"<td>{(lambda v: f'{v:.3f}' if v else '—')(sparklines[m][-1] if sparklines[m] else None)}</td>"
      f"<td>{(lambda v: f'{v:.3f}' if v else '—')(_avg(entries, m))}</td>"
      f"<td><canvas id='spark_{m}' width='120' height='30'></canvas></td>"
      f"<td class='{'pass' if (sparklines[m] or [0])[-1] >= 0.80 else 'warn'}'>"
      f"{'✅ Pass' if (sparklines[m] or [0])[-1] >= 0.80 else '⚠️ Review'}</td>"
      f"</tr>"
      for m in METRICS if sparklines.get(m)
  )}
</table>

<h2>Run History (last 30)</h2>
<table>
  <tr><th>Run</th><th>Commit</th><th>Date</th>
  {"".join(f"<th>{m}</th>" for m in METRICS)}</tr>
  {"".join(
      "<tr>"
      f"<td>{e.get('run_id','?')}</td>"
      f"<td><code>{e.get('commit','?')}</code></td>"
      f"<td>{e.get('timestamp','')[:10]}</td>"
      + "".join(
          f"<td class='{'pass' if (e.get('scores',{{}}).get(m) or 0) >= 0.80 else 'warn'}'>"
          f"{f\"{e.get('scores', {{}}).get(m):.3f}\" if e.get('scores',{{}}).get(m) is not None else '—'}"
          f"</td>"
          for m in METRICS
      )
      + "</tr>"
      for e in entries[-30:]
  )}
</table>

<script>
const data = {spark_js};
function drawSparkline(canvasId, values) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas || !values.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const pad = 3;
  ctx.clearRect(0, 0, w, h);
  // threshold line
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1; ctx.setLineDash([2,2]);
  const ty = h - pad - (0.80 * (h - 2*pad));
  ctx.beginPath(); ctx.moveTo(pad, ty); ctx.lineTo(w-pad, ty); ctx.stroke();
  ctx.setLineDash([]);
  // sparkline
  ctx.strokeStyle = '#0066cc'; ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {{
    const x = pad + (i / (values.length - 1)) * (w - 2*pad);
    const y = h - pad - (v * (h - 2*pad));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();
  // last point
  const lx = w - pad, ly = h - pad - (values[values.length-1] * (h - 2*pad));
  ctx.fillStyle = values[values.length-1] >= 0.80 ? '#22863a' : '#d93025';
  ctx.beginPath(); ctx.arc(lx, ly, 3, 0, 2*Math.PI); ctx.fill();
}}
Object.entries(data).forEach(([m, v]) => drawSparkline('spark_' + m, v));
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LLM evaluation drift log to shareable formats."
    )
    parser.add_argument(
        "--format",
        choices=["summary", "markdown", "csv", "html", "all"],
        default="summary",
    )
    parser.add_argument("--window", type=int, default=0,
                        help="Last N runs to include (0 = all).")
    args = parser.parse_args()

    entries = load_entries(window=args.window)
    if not entries:
        print("⚠️  No drift log data found.")
        print(f"   Expected: {DRIFT_LOG_PATH}")
        print("   Run some CI pipelines first, or:")
        print("   python scripts/drift_detection.py log --run-id 1 --commit abc123 "
              "--score faithfulness=0.92")
        sys.exit(0)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    if fmt in ("summary", "all"):
        print(export_summary(entries))

    if fmt in ("markdown", "all"):
        path = REPORTS_DIR / "evaluation_report.md"
        path.write_text(export_markdown(entries), encoding="utf-8")
        print(f"📝 Markdown report saved: {path}")

    if fmt in ("csv", "all"):
        path = REPORTS_DIR / "evaluation_report.csv"
        path.write_text(export_csv(entries), encoding="utf-8")
        print(f"📊 CSV report saved: {path}")

    if fmt in ("html", "all"):
        path = REPORTS_DIR / "evaluation_report.html"
        path.write_text(export_html(entries), encoding="utf-8")
        print(f"🌐 HTML report saved: {path}")


if __name__ == "__main__":
    main()
