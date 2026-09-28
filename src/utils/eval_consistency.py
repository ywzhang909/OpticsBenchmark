#!/usr/bin/env python3
"""
Optis Benchmark - Consistency Evaluation Script

Reads two judge score files (``.json`` or ``.jsonl``) and assesses how
consistent they are, using the Pearson correlation.

Each record looks like::

    {"id": int, "output_model": str, "context": str, "judge_model": str, "score": float}

``id`` is unique within a single file and ``score`` is a scalar; the evaluation
only uses ``id`` and ``score``, while ``output_model`` / ``context`` /
``judge_model`` drive the cross-check: a warning is raised when both files come
from the same judge, or when one ``id`` maps to a different ``output_model`` /
``context``.

Usage::

    python -m src.utils.eval_consistency
    python -m src.utils.eval_consistency --scores_a a.jsonl --scores_b b.jsonl
    python -m src.utils.eval_consistency --scores_a a.jsonl --scores_b b.jsonl \\
        -o results/consistency.json
    python -m src.utils.eval_consistency --scores_a eval_results.jsonl \\
        --scores_b gold.jsonl --metric exact_match_avg
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.algorithm.consistency import evaluate_consistency
from src.utils import logger
from src.utils.score_io import cross_check_metadata, load_scores

# =============================================================================
# Private Functions
# =============================================================================


def _describe_meta(summary: dict[str, Any]) -> str:
    """Format a metadata summary as one readable line."""
    judges = "/".join(summary["judge_models"]) or "-"
    outputs = "/".join(summary["output_models"]) or "-"
    context = "yes" if summary["has_context"] else "no"
    return f"judge_model={judges} | output_models={outputs} | context={context}"


def _cross_check(list_a: list[Any], list_b: list[Any]) -> dict[str, Any]:
    """Cross-check whether the metadata of two score records is comparable.

    The field-level comparison (``output_model`` / ``context``) is done by
    :func:`src.utils.score_io.cross_check_metadata`; this only adds the
    consistency-specific "same judge" check and the legacy-format notice.

    Args:
        list_a: First score records
        list_b: Second score records

    Returns:
        Dict holding the metadata summaries of both records and the comparison:
        - ``a`` / ``b``: their ``judge_model``, ``output_models``, ``context`` summaries
        - ``same_judge_model``: whether both files were scored by the same judge
        - ``output_model_mismatch_ids``: ids whose ``output_model`` differs between the sides
        - ``context_mismatch_ids``: ids whose ``context`` differs between the sides
    """
    report = cross_check_metadata({"a": list_a, "b": list_b})
    summary_a = report["judges"]["a"]
    summary_b = report["judges"]["b"]

    same_judge = bool(report["duplicate_judges"])
    if same_judge:
        logger.warning(
            f"both files were scored by the same judge "
            f"({'/'.join(report['duplicate_judges'])}); the consistency check is meaningless"
        )
    if not summary_a["judge_models"] and not summary_b["judge_models"]:
        logger.info(
            "neither file carries judge_model metadata; treating them as the legacy "
            "{id, score} format and skipping the metadata check"
        )

    return {
        "a": summary_a,
        "b": summary_b,
        "same_judge_model": same_judge,
        "output_model_mismatch_ids": report["output_model_mismatch_ids"],
        "context_mismatch_ids": report["context_mismatch_ids"],
    }


# =============================================================================
# Functions
# =============================================================================


def _log_summary(
    result: dict[str, Any],
    meta: dict[str, Any],
    path_a: str,
    path_b: str,
) -> None:
    """Print the consistency evaluation summary."""
    if result["defined"]:
        p_value = result["pearson_p"]
        p_text = f", p = {p_value:.4e}" if p_value is not None else ""
        logger.info(f"Pearson r = {result['pearson_r']:.4f}{p_text}")
    else:
        logger.warning(
            f"Pearson r is undefined (recorded as {result['pearson_r']:.4f}): "
            f"{result['undefined_reason']}"
        )

    logger.info(f"paired samples: {result['num_pairs']}")
    logger.info(f"mean(scores_a) = {result['mean_a']:.4f}")
    logger.info(f"mean(scores_b) = {result['mean_b']:.4f}")
    logger.info(f"unpaired ids: {result['num_only_a']} only in A, {result['num_only_b']} only in B")
    logger.info(f"A: {path_a} | {_describe_meta(meta['a'])}")
    logger.info(f"B: {path_b} | {_describe_meta(meta['b'])}")


def main() -> int:
    """Command-line entry point: load two score lists and report consistency.

    Returns:
        Process exit code; 0 means success.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate consistency between two score lists via Pearson correlation"
    )
    parser.add_argument(
        "--scores_a",
        type=str,
        default="dataset/simulat_data/consistency_scores_a.jsonl",
        help="Path to the first score file (.json or .jsonl), records shaped like "
        "{'id', 'output_model', 'context', 'judge_model', 'score'}",
    )
    parser.add_argument(
        "--scores_b",
        type=str,
        default="dataset/simulat_data/consistency_scores_b.jsonl",
        help="Path to the second score file (.json or .jsonl), same record format",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional: write the result to this JSON file",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to read from evaluation result records (the {task_id, metrics} "
        "form); it has no effect on the new format, whose score is a scalar",
    )

    args = parser.parse_args()

    try:
        list_a = load_scores(args.scores_a, args.metric)
        list_b = load_scores(args.scores_b, args.metric)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"failed to load the score files: {e}")
        return 1

    if not list_a or not list_b:
        logger.error("input is empty; cannot evaluate consistency")
        return 1

    meta = _cross_check(list_a, list_b)

    try:
        result = evaluate_consistency(list_a, list_b)
    except Exception as e:
        logger.error(f"consistency evaluation failed: {e}")
        return 1

    _log_summary(result, meta, args.scores_a, args.scores_b)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "scores_a": args.scores_a,
            "scores_b": args.scores_b,
            "metric": args.metric,
            **result,
            "meta": meta,
        }
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"result written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
