#!/usr/bin/env python3
"""
Optis Benchmark - Bias Evaluation Script (Judge Self-Preference)

Reads a **single** judge score file and measures how much that judge prefers its
own model's outputs::

    ErrorRate_SE = y_self / y_other - 1

The file holds one judge's complete scores over n samples x m model outputs, i.e.
``m * n`` records, each shaped like::

    {"id": int, "output_model": str, "context": str, "judge_model": str, "score": float}

``id`` repeats across the file (one record per model per sample), so the pairing
key is ``(id, output_model)``. The judge model must be one of the m scored
models, otherwise its own output cannot be told apart from the others.

Evaluation steps:

1. average the scores of the other m-1 models per sample to get ``y_other``;
2. compute ``y_self / y_other - 1`` per sample, giving n bias values;
3. report the mean of those n values (the median and the per-sample breakdown
   are reported as well).

The judge name defaults to the unique ``judge_model`` in the records and can be
overridden with ``--judge-model``; a repeated ``(id, output_model)`` pair keeps
its last occurrence.

Usage::

    python -m src.utils.eval_bias
    python -m src.utils.eval_bias --scores path/to/judge_scores.jsonl
    python -m src.utils.eval_bias --scores judge_scores.json --judge-model gpt-4o
    python -m src.utils.eval_bias --scores judge_scores.jsonl -o results/bias.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from src.algorithm.bias import evaluate_judge_bias
from src.utils import logger
from src.utils.score_io import load_scores

# =============================================================================
# Constants
# =============================================================================

_COLUMN_WIDTHS = (6, 8, 8, 13)  # id, y_self, y_other, ErrorRate_SE

# =============================================================================
# Private Functions
# =============================================================================


def _row(cells: Sequence[str]) -> str:
    """Format one row of cells as a right-aligned pipe table row.

    Args:
        cells: Cell texts, one per column width

    Returns:
        A string like ``|      1 |   2.8700 |   2.3700 |        0.2110 |``.
    """
    body = "|".join(f" {cell:>{width}} " for cell, width in zip(cells, _COLUMN_WIDTHS, strict=True))
    return f"|{body}|"


def _format_table(result: dict[str, Any]) -> list[str]:
    """Format the per-sample bias results as table text lines.

    Args:
        result: Return value of :func:`~src.algorithm.bias.evaluate_judge_bias`

    Returns:
        One line for the header, one for the separator, and one per sample.
    """
    lines = [
        _row(("id", "y_self", "y_other", "ErrorRate_SE")),
        _row(tuple("-" * width for width in _COLUMN_WIDTHS)),
    ]
    for sample in result["per_sample"]:
        lines.append(
            _row(
                (
                    str(sample["id"]),
                    f"{sample['y_self']:.4f}",
                    f"{sample['y_other']:.4f}",
                    f"{sample['error_rate_se']:.4f}",
                )
            )
        )
    return lines


def _log_report(result: dict[str, Any], path: str) -> None:
    """Print the bias evaluation report."""
    logger.info("=" * 60)
    logger.info("Bias evaluation: judge self-preference (ErrorRate_SE = y_self / y_other - 1)")
    logger.info("=" * 60)
    logger.info(f"score file: {path}")
    logger.info(
        f"judge: {result['judge_model']} | {result['num_records']} records "
        f"({result['num_scored']} scored, {result['num_invalid']} invalid)"
    )
    scored = "/".join(result["output_models"])
    logger.info(f"scored models ({len(result['output_models'])}): {scored}")
    logger.info(f"samples: {result['num_samples']}")

    if not result["defined"]:
        logger.warning(f"cannot assess self-preference: {result['undefined_reason']}")
        logger.info(f"undefined samples: {result['undefined_ids']}")
        return

    # Print the table row by row: loguru prefixes only the first line of a single
    # message, which would shift the header right. Keeping logger.info on a single
    # source line inside the loop makes the {function}:{line} prefix identical for
    # every row, so the columns line up.
    for line in _format_table(result):
        logger.info(line)

    logger.info(
        f"defined samples: {result['num_defined']} "
        f"({result['num_undefined']} undefined"
        f"{', ids ' + str(result['undefined_ids']) if result['undefined_ids'] else ''})"
    )
    logger.info(
        f"mean(y_self) = {result['mean_y_self']:.4f}, mean(y_other) = {result['mean_y_other']:.4f}"
    )
    logger.info(f"mean ErrorRate_SE = {result['mean_error_rate_se']:.4f}")
    logger.info(f"median ErrorRate_SE = {result['median_error_rate_se']:.4f}")


# =============================================================================
# Functions
# =============================================================================


def load_judge_records(path: str) -> list[Any]:
    """Load a single judge's score file and run a basic sanity check.

    Args:
        path: Path to the score file (.json or .jsonl)

    Returns:
        The judge score records.

    Raises:
        FileNotFoundError: The file does not exist
        ValueError: The file content cannot be parsed or is empty
    """
    records = load_scores(path)
    if not records:
        raise ValueError(f"{path} holds no score records")
    return records


def main() -> int:
    """Command-line entry point: measure one judge model's self-preference.

    Returns:
        Process exit code; 0 means success.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate one judge model's self-preference bias via ErrorRate_SE"
    )
    parser.add_argument(
        "--scores",
        type=str,
        default="dataset/simulat_data/bias_judge_gpt-4o.jsonl",
        help="Path to the judge score file (.json or .jsonl) holding m * n records over "
        "n samples x m model outputs, each shaped like "
        "{'id', 'output_model', 'context', 'judge_model', 'score'}",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Judge model name; defaults to the unique judge_model in the records, "
        "and must be one of the scored models",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional: write the result to this JSON file",
    )

    args = parser.parse_args()

    try:
        records = load_judge_records(args.scores)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"failed to load the score file: {e}")
        return 1

    try:
        result = evaluate_judge_bias(records, args.judge_model)
    except Exception as e:
        logger.error(f"bias evaluation failed: {e}")
        return 1

    _log_report(result, args.scores)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "scores_file": args.scores,
            **result,
        }
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"result written to: {output_path}")

    return 0 if result["defined"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
