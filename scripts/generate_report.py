#!/usr/bin/env python3
"""
OptiS Benchmark - Report Generator

Generate HTML and markdown reports from evaluation results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load evaluation results from file."""
    path = Path(results_path)
    
    if path.suffix == ".jsonl":
        results = {"individual": []}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results["individual"].append(json.loads(line))
        
        # Load aggregated results if available
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                aggregated = json.load(f)
                results.update(aggregated)
    else:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    return results


def generate_html_report(results: dict[str, Any], output_path: str | Path) -> None:
    """Generate HTML report from results."""
    
    # Extract data
    total_tasks = results.get("total_tasks", 0)
    successful = results.get("successful_tasks", 0)
    success_rate = results.get("success_rate", 0) * 100
    avg_score = results.get("avg_score", 0) * 100
    total_cost = results.get("total_cost", 0)
    avg_time = results.get("avg_execution_time", 0)
    model = results.get("model", "Unknown")
    task_id = results.get("task_id", "Unknown")
    timestamp = results.get("timestamp", datetime.now().isoformat())
    
    metrics = results.get("metrics_summary", {})
    
    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OptiS Benchmark Results - {task_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .card {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2d3748;
            margin-bottom: 0.5rem;
            font-size: 2rem;
        }}
        
        h2 {{
            color: #4a5568;
            margin-bottom: 1rem;
            font-size: 1.25rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }}
        
        .meta {{
            color: #718096;
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.25rem;
        }}
        
        .stat-label {{
            color: #718096;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .metrics-table th,
        .metrics-table td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .metrics-table th {{
            background: #f7fafc;
            font-weight: 600;
            color: #4a5568;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-success {{
            background: #c6f6d5;
            color: #22543d;
        }}
        
        .badge-warning {{
            background: #fefcbf;
            color: #744210;
        }}
        
        .badge-danger {{
            background: #fed7d7;
            color: #742a2a;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            font-size: 0.875rem;
            margin-top: 2rem;
            opacity: 0.8;
        }}
        
        .progress-bar {{
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>OptiS Benchmark Results</h1>
            <div class="meta">
                Task: {task_id} | Model: {model} | Date: {timestamp}
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_tasks}</div>
                    <div class="stat-label">Total Tasks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{successful}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {success_rate}%"></div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_score:.1f}%</div>
                    <div class="stat-label">Avg Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${total_cost:.4f}</div>
                    <div class="stat-label">Total Cost</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_time:.1f}s</div>
                    <div class="stat-label">Avg Time</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Metrics Summary</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Std Dev</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for metric_name, values in metrics.items():
        html += f"""
                    <tr>
                        <td><strong>{metric_name}</strong></td>
                        <td>{values.get('mean', 0):.4f}</td>
                        <td>{values.get('min', 0):.4f}</td>
                        <td>{values.get('max', 0):.4f}</td>
                        <td>{values.get('std', 0):.4f}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            Generated by OptiS Benchmark | """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    
    print(f"HTML report generated: {output_path}")


def generate_markdown_report(results: dict[str, Any], output_path: str | Path) -> None:
    """Generate Markdown report from results."""
    
    # Extract data
    total_tasks = results.get("total_tasks", 0)
    successful = results.get("successful_tasks", 0)
    success_rate = results.get("success_rate", 0) * 100
    avg_score = results.get("avg_score", 0) * 100
    total_cost = results.get("total_cost", 0)
    avg_time = results.get("avg_execution_time", 0)
    model = results.get("model", "Unknown")
    task_id = results.get("task_id", "Unknown")
    timestamp = results.get("timestamp", datetime.now().isoformat())
    
    metrics = results.get("metrics_summary", {})
    
    md = f"""# OptiS Benchmark Results

## Overview

| Field | Value |
|-------|-------|
| Task | {task_id} |
| Model | {model} |
| Date | {timestamp} |
| Total Tasks | {total_tasks} |
| Successful | {successful} |
| Success Rate | {success_rate:.1f}% |
| Average Score | {avg_score:.1f}% |
| Total Cost | ${total_cost:.4f} |
| Average Time | {avg_time:.1f}s |

## Metrics Summary

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|---------|
"""
    
    for metric_name, values in metrics.items():
        md += f"| {metric_name} | {values.get('mean', 0):.4f} | {values.get('min', 0):.4f} | {values.get('max', 0):.4f} | {values.get('std', 0):.4f} |\n"
    
    md += f"""
---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by OptiS Benchmark*
"""
    
    # Write Markdown file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    
    print(f"Markdown report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports from OptiS Benchmark results"
    )
    parser.add_argument(
        "results",
        type=str,
        help="Path to results file (JSON or JSONL)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path (default: results/report.html)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "markdown", "both"],
        default="html",
        help="Report format",
    )
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        results_path = Path(args.results)
        if args.format in ("html", "both"):
            output_path = results_path.parent / "report.html"
        else:
            output_path = results_path.parent / "report.md"
    
    # Generate reports
    if args.format in ("html", "both"):
        html_path = output_path.parent / "report.html"
        generate_html_report(results, html_path)
    
    if args.format in ("markdown", "both"):
        md_path = output_path.parent / "report.md"
        generate_markdown_report(results, md_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
