"""
OptiS Benchmark - Paper Evaluation & Report Generator

End-to-end pipeline:
1. Read a paper file (PDF or TXT)
2. Parse structured fields
3. Evaluate using RubricBasedEvaluator (offline or online via vLLM)
4. Generate a detailed Markdown report with Mermaid charts (xychart-beta)

Usage::

    # Offline mode (no LLM judge — all scores 0)
    uv run python -m src.tools.paper_eval_report --online=false

    # Online mode (vLLM endpoint)
    uv run python -m src.tools.paper_eval_report --online

    # Custom input
    uv run python -m src.tools.paper_eval_report ^
        --input "dataset/info_extraction/AO/李儒佳*_analysis.txt" ^
        --output results/report.md --online
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.evaluators import RubricBasedEvaluator
from src.module import EvaluationResult
from src.tools.paper_reader import parse_ao_analysis, read_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_commit_id() -> str:
    """Return the short commit hash of HEAD, or ``"unknown"``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _build_predicted(parsed: dict[str, Any]) -> dict[str, str]:
    """Map parsed fields → paper_info_extract JSON keys for evaluation."""
    return {
        "ten keywords": parsed.get("keywords", ""),
        "objective": parsed.get("objective", ""),
        "novelty": parsed.get("novelty", ""),
        "method": parsed.get("method", ""),
        "performance metrics": parsed.get("performance_metrics", ""),
    }


def _build_expected(parsed: dict[str, Any]) -> dict[str, str]:
    """Build gold-standard expected dict (same as predicted for self-test)."""
    return dict(_build_predicted(parsed))


def _parse_fields_from_text(raw_text: str, filename: str) -> dict[str, Any]:
    """Parse structured fields from a paper text file (AO format)."""
    parsed = parse_ao_analysis(raw_text)
    logger.debug("Parsed fields: title={}, keywords={!r:.60}",
                 parsed.get("title"), parsed.get("keywords"))
    if not parsed.get("title"):
        parsed["title"] = filename.replace("_analysis.txt", "").replace("_", " ")
        parsed["method"] = raw_text[:2000]
        logger.warning("AO parser returned empty title, using fallback: {}",
                       parsed["title"])
    return parsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate_paper(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    judge_config: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Run RubricBasedEvaluator on a single paper.

    Args:
        predicted: Predicted/extracted fields.
        expected: Expected/gold-standard fields.
        judge_config: LLM judge config (None = offline → zeros).

    Returns:
        EvaluationResult with metrics and details.
    """
    config = {}
    if judge_config:
        config["judge_config"] = judge_config
        logger.info("Online mode — judge endpoint: {} model: {}",
                     judge_config.get("api_base"),
                     judge_config.get("model"))
    else:
        logger.info("Offline mode — scores will be 0 (no LLM judge)")

    ev = RubricBasedEvaluator(config)

    logger.debug("Calling setup()...")
    await ev.setup()
    logger.debug("setup() done — llm_callable={}", ev._llm_callable is not None)

    try:
        logger.debug("Running evaluate() with predicted keys: {}",
                     list(predicted.keys()))
        result = await ev.evaluate(
            task_id="paper_eval",
            predicted_output=predicted,
            expected_output=expected,
        )
        logger.debug("evaluate() returned metrics={}", result.metrics)
    except Exception as exc:
        logger.error("Evaluation failed: {}", exc)
        result = EvaluationResult(
            task_id="paper_eval",
            metrics={"accuracy": 0.0, "completeness": 0.0,
                     "readability": 0.0, "hallucination_rate": 0.0},
            details={"error": str(exc)},
        )
    finally:
        logger.debug("Calling teardown()...")
        await ev.teardown()
        logger.debug("teardown() done")

    return result


# ---------------------------------------------------------------------------
# Mermaid chart helpers
# ---------------------------------------------------------------------------


def _mermaid_bar(title: str, labels: list[str], values: list[float],
                 color: str = "#4a90d9") -> str:
    """Generate a vertical bar chart via Mermaid ``xychart``.

    Per the official docs (https://mermaid.js.org/syntax/xyChart.html)::

        xychart
            title "Title"
            x-axis ["A", "B", "C"]
            y-axis "Score (1-5)" 0 --> 5
            bar [4.5, 3.0, 5.0]

    The diagram keyword is ``xychart`` (NOT ``xychart-beta``).
    """
    quoted = ", ".join(f'"{l}"' for l in labels)
    vals = ", ".join(f"{v:.2f}" for v in values)
    return (
        "```mermaid\n"
        "xychart\n"
        f'    title "{title}"\n'
        f"    x-axis [{quoted}]\n"
        '    y-axis "Score (1-5)" 0 --> 5\n'
        f"    bar [{vals}]\n"
        "```"
    )


def _mermaid_pie(title: str, data: dict[str, float]) -> str:
    """Generate a Mermaid pie chart."""
    lines = ["```mermaid", f"pie title {title}"]
    for label, val in data.items():
        lines.append(f'    "{label}" : {val}')
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    parsed: dict[str, Any],
    result: EvaluationResult,
    commit_id: str,
    source_file: str,
    duration_sec: float,
) -> str:
    """Generate a complete Markdown evaluation report with Mermaid charts.

    Args:
        parsed: Parsed paper fields.
        result: EvaluationResult from RubricBasedEvaluator.
        commit_id: Git commit hash.
        source_file: Original input file path.
        duration_sec: Total execution time.

    Returns:
        Markdown string.
    """
    metrics = result.metrics
    details: dict = result.details or {}
    field_scores: dict = details.get("field_scores", {})
    field_justifications: dict = details.get("field_justifications", {})
    hallucination: dict = details.get("hallucination", {})

    hallu_items: list[dict] = hallucination.get("hallucinated_items", [])
    hallu_count = len(hallu_items)
    total_checked = hallucination.get("total_checked_items", 0)

    lines: list[str] = []

    # ---- header ----
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.extend([
        "# Paper Evaluation Report",
        "",
        f"**Generated:** {ts}",
        f"**Commit:** `{commit_id}`",
        f"**Source:** `{source_file}`",
        f"**Duration:** {duration_sec:.1f}s",
        "",
    ])

    # Offline-mode warning
    all_zero = all(
        metrics.get(k, 0.0) == 0.0
        for k in ("accuracy", "completeness", "readability")
    )
    if all_zero:
        lines.extend([
            "> **⚠️ Offline mode** — all scores are 0.0 because no LLM judge "
            "was configured. Run with `--online` to use the vLLM endpoint for "
            "real LLM-based scoring.",
            "",
        ])

    # ---- paper meta ----
    lines.append("## Paper Meta Information")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    for key in ("title", "year", "doi", "journal", "keywords", "authors"):
        val = parsed.get(key, "")
        lines.append(f"| {key.capitalize()} | {val or '—'} |")
    lines.append("")

    # ---- metric summary ----
    lines.append("## Evaluation Metrics Summary")
    lines.append("")
    lines.append("| Metric | Score (1-5) |")
    lines.append("|--------|-------------|")
    for name in ("accuracy", "completeness", "readability"):
        val = metrics.get(name, 0.0)
        stars = "★" * round(val) + "☆" * (5 - round(val))
        lines.append(f"| {name.capitalize()} | {val:.2f} {stars} |")
    hr = metrics.get("hallucination_rate", 0.0)
    lines.append(
        f"| Hallucination Rate | {hr:.2f} ({hallu_count}/{total_checked} items) |"
    )
    lines.append("")

    # ---- mermaid charts ----
    lines.append("## Visual Summary")
    lines.append("")

    # 1) Overall rubric scores bar chart
    lines.append(_mermaid_bar(
        "Rubric Scores (1-5)",
        ["Accuracy", "Completeness", "Readability"],
        [metrics.get("accuracy", 0),
         metrics.get("completeness", 0),
         metrics.get("readability", 0)],
    ))
    lines.append("")
    lines.append("*Figure 1: Overall rubric scores across all fields.*")
    lines.append("")

    # 2) Per-field bar charts (one per dimension)
    if field_scores:
        fig_counter = 2
        for dim in ("accuracy", "completeness", "readability"):
            labels: list[str] = []
            vals: list[float] = []
            for fn in RubricBasedEvaluator.FIELDS:
                fs = field_scores.get(fn, {})
                labels.append(RubricBasedEvaluator.DISPLAY_NAMES.get(fn, fn))
                vals.append(fs.get(dim, 0.0))
            lines.append(_mermaid_bar(f"Per-Field {dim.capitalize()}", labels, vals))
            lines.append("")
            lines.append(
                f"*Figure {fig_counter}: {dim.capitalize()} per field.*"
            )
            lines.append("")
            fig_counter += 1

    # 3) Hallucination pie
    lines.append(_mermaid_pie(
        "Hallucination Breakdown",
        {"Hallucinated": hallu_count, "Valid": total_checked - hallu_count},
    ))
    lines.append("")
    lines.append("*Figure 5: Hallucinated vs valid items.*")
    lines.append("")

    # ---- per-field detail ----
    lines.append("## Per-Field Detail")
    lines.append("")
    for fn in RubricBasedEvaluator.FIELDS:
        display = RubricBasedEvaluator.DISPLAY_NAMES.get(fn, fn)
        fs = field_scores.get(fn, {})
        fj = field_justifications.get(fn, {})
        lines.append(f"### {display} (`{fn}`)")
        lines.append("")
        lines.append("| Dimension | Score | Justification |")
        lines.append("|-----------|-------|---------------|")
        for dim in ("accuracy", "completeness", "readability"):
            lines.append(
                f"| {dim.capitalize()} | {fs.get(dim, 0.0):.2f} | "
                f"{fj.get(dim, '')} |"
            )
        lines.append("")

    # ---- hallucination detail ----
    if hallu_items:
        lines.append("### Hallucinated Items")
        lines.append("")
        lines.append("| Field | Content |")
        lines.append("|-------|--------|")
        for item in hallu_items:
            lines.append(f"| {item.get('field', '?')} | `{item.get('item', '?')}` |")
        lines.append("")

    # ---- raw metrics ----
    lines.append("## Raw Metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    # ---- footer ----
    lines.append("---")
    lines.append("")
    lines.append(
        "*Report generated by [OptiS Benchmark]"
        "(https://github.com/ywzhang909/OpticsBenchmark) "
        "RubricBasedEvaluator*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_judge_config() -> dict[str, Any]:
    """Build the vLLM judge config (same as online tests)."""
    return {
        "provider": "openai",
        "model": "qwen",
        "api_base": "https://impecunious909.asia/vllm/v1",
        "api_key": "sk-11235813",
        "temperature": 0.0,
        "raw_http": True,
    }


async def _run(args: argparse.Namespace) -> None:
    """Main async entry point."""
    # Resolve input path (support glob)
    src_path = Path(args.input)
    if not src_path.exists():
        import glob as glob_mod
        matches = glob_mod.glob(str(src_path))
        if not matches:
            logger.error("No file matches: {}", args.input)
            sys.exit(1)
        src_path = Path(matches[0])

    logger.info("Step 1/4 — Reading file: {}", src_path)
    raw_text = read_file(src_path)
    logger.debug("File size: {} chars", len(raw_text))

    logger.info("Step 2/4 — Parsing structured fields...")
    parsed = _parse_fields_from_text(raw_text, src_path.name)
    predicted = _build_predicted(parsed)
    logger.debug("Predicted keys with data: {}",
                 {k for k, v in predicted.items() if v})

    judge_cfg = _build_judge_config() if args.online else None
    mode = "online (vLLM)" if args.online else "offline"

    logger.info("Step 3/4 — Evaluating ({})...", mode)
    expected = _build_expected(parsed)
    t0 = datetime.now()
    result = await evaluate_paper(predicted, expected, judge_cfg)
    duration = (datetime.now() - t0).total_seconds()

    logger.info("Metrics: {}", result.metrics)
    logger.info("Field scores present: {} fields",
                len(result.details.get("field_scores", {})))

    commit_id = _get_commit_id()

    logger.info("Step 4/4 — Generating report...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = generate_report(
        parsed=parsed, result=result, commit_id=commit_id,
        source_file=str(src_path), duration_sec=duration,
    )
    out_path.write_text(report_md, encoding="utf-8")

    logger.success("Report saved to: {}", out_path.resolve())
    logger.info("Commit: {} | Mode: {} | Time: {:.1f}s",
                commit_id, mode, duration)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Evaluation & Report Generator",
    )
    parser.add_argument(
        "--input", "-i",
        default=("dataset/info_extraction/AO/"
                 "李儒佳 等 - 2021 - 相位型空间光调制器的自参考标定方法_analysis.txt"),
        help="Input file path (PDF, TXT, or glob pattern)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/paper_eval_report.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--online", action="store_true",
        help="Enable online LLM judge (vLLM endpoint) — required for non-zero scores",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.set_defaults(func=_run)

    args = parser.parse_args()

    # Enable debug logging if --verbose
    if args.verbose:
        from loguru import logger as loguru_logger
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="DEBUG")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
