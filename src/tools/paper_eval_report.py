"""
OptiS Benchmark - Paper Evaluation & Report Generator

End-to-end pipeline:
1. Read a paper file (PDF or TXT)
2. Parse structured fields
3. Evaluate using RubricBasedEvaluator (offline or online via vLLM)
4. Generate a detailed Markdown report with Mermaid charts

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
from src.tools.md_report_builder import MdReportBuilder
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
# Report generation (uses MdReportBuilder)
# ---------------------------------------------------------------------------


def _star_rating(value: float, total: int = 5) -> str:
    """Generate a star-rating string like ``★★★★★``."""
    filled = round(value)
    return "★" * filled + "☆" * (total - filled)


def generate_report(
    parsed: dict[str, Any],
    result: EvaluationResult,
    commit_id: str,
    source_file: str,
    duration_sec: float,
    lang: str = "en",
) -> str:
    """Generate a complete Markdown evaluation report.

    Args:
        parsed: Parsed paper fields.
        result: EvaluationResult from RubricBasedEvaluator.
        commit_id: Git commit hash.
        source_file: Original input file path.
        duration_sec: Total execution time.
        lang: Language code for the report (``"en"`` or ``"zh_CN"``).

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

    md = MdReportBuilder(lang=lang)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- Header + metadata --
    md.h1(md.t("report_title"))
    md.meta_block({
        md.t("generated"): ts,
        md.t("commit"): md.code(commit_id),
        md.t("source"): md.code(source_file),
        md.t("duration"): f"{duration_sec:.1f}s",
    })

    # -- Offline-mode warning --
    all_zero = all(
        metrics.get(k, 0.0) == 0.0
        for k in ("accuracy", "completeness", "readability")
    )
    if all_zero:
        md.blockquote(md.t("offline_warning"))

    # -- Paper meta table --
    md.h2(md.t("paper_meta"))
    meta_rows: list[list[str]] = []
    for key in ("title", "year", "doi", "journal", "keywords", "authors"):
        val = parsed.get(key, "")
        meta_rows.append([key.capitalize(), val or "—"])
    md.table([md.t("field"), md.t("value")], meta_rows)

    # -- Metric summary --
    md.h2(md.t("eval_metrics"))
    metric_rows: list[list[str]] = []
    for name in ("accuracy", "completeness", "readability"):
        val = metrics.get(name, 0.0)
        label = md.t(name, name.capitalize())
        metric_rows.append([label, f"{val:.2f} {_star_rating(val)}"])
    hr = metrics.get("hallucination_rate", 0.0)
    metric_rows.append([
        md.t("hallucination_rate"),
        f"{hr:.2f} ({hallu_count}/{total_checked} {md.t('items')})",
    ])
    md.table([md.t("metric"), md.t("score")], metric_rows)

    # -- Visual summary --
    md.h2(md.t("visual_summary"))

    # 1) Overall rubric scores bar chart
    md.mermaid_bar(
        md.t("rubric_scores_chart"),
        [
            md.t("accuracy", "Accuracy"),
            md.t("completeness", "Completeness"),
            md.t("readability", "Readability"),
        ],
        [
            metrics.get("accuracy", 0),
            metrics.get("completeness", 0),
            metrics.get("readability", 0),
        ],
    )
    md.figure_caption(1, "Overall rubric scores across all fields", lang=lang)

    # 2) Per-field bar charts
    if field_scores:
        fig_num = 2
        for dim in ("accuracy", "completeness", "readability"):
            labels: list[str] = []
            vals: list[float] = []
            for fn in RubricBasedEvaluator.FIELDS:
                fs = field_scores.get(fn, {})
                labels.append(RubricBasedEvaluator.DISPLAY_NAMES.get(fn, fn))
                vals.append(fs.get(dim, 0.0))
            md.mermaid_bar(
                f"{md.t('per_field')} {dim.capitalize()}",
                labels, vals,
            )
            md.figure_caption(
                fig_num, f"{dim.capitalize()} per field", lang=lang
            )
            fig_num += 1

    # 3) Hallucination pie
    md.mermaid_pie(
        md.t("hallucination_chart"),
        {
            md.t("hallucinated"): hallu_count,
            md.t("valid"): total_checked - hallu_count,
        },
    )
    md.figure_caption(5, md.t("hallucinated_vs_valid"), lang=lang)

    # -- Per-field detail --
    md.h2(md.t("per_field_detail"))
    for fn in RubricBasedEvaluator.FIELDS:
        display = RubricBasedEvaluator.DISPLAY_NAMES.get(fn, fn)
        fs = field_scores.get(fn, {})
        fj = field_justifications.get(fn, {})

        md.h3(f"{display} (``{fn}``)")
        rows: list[list[str]] = []
        for dim in ("accuracy", "completeness", "readability"):
            rows.append([
                md.t(dim, dim.capitalize()),
                f"{fs.get(dim, 0.0):.2f}",
                fj.get(dim, ""),
            ])
        md.table(
            [md.t("dimension"), md.t("score"), md.t("justification")],
            rows,
        )

    # -- Hallucination detail --
    if hallu_items:
        md.h3(md.t("hallucinated_items"))
        hallu_rows = [
            [item.get("field", "?"), md.code(item.get("item", "?"))]
            for item in hallu_items
        ]
        md.table([md.t("field"), md.t("content")], hallu_rows)

    # -- Raw metrics --
    md.h2(md.t("raw_metrics"))
    md.code_block(json.dumps(metrics, indent=2, ensure_ascii=False), lang="json")

    # -- Footer --
    md.hr()
    md.p(md.t("footer_generated_by"))

    return md.build()


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
        lang=args.lang,
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
        "--lang", default="en", choices=["en", "zh_CN"],
        help="Report language (default: en)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.set_defaults(func=_run)

    args = parser.parse_args()

    if args.verbose:
        from loguru import logger as loguru_logger
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="DEBUG")

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
