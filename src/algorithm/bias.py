"""
Optis Benchmark - Bias Evaluation Utilities (Judge Self-Preference)

In LLM-as-judge harnesses, judges commonly show a "self-preference": they tend
to give higher scores to answers whose style or distribution resembles their own
generation.

The bias is measured as::

    ErrorRate_SE = y_self / y_other - 1

where ``y_self`` is the judge's score for its **own model's output** and
``y_other`` is the mean score the same judge gave the **outputs of the other
m-1 models for that sample**.

The input is a single file holding one judge's complete scores over n samples x m
model outputs, i.e. ``m * n`` records. ``id`` repeats across the file, so the
pairing key is ``(id, output_model)``::

    {"id": int, "output_model": str, "context": str, "judge_model": str, "score": float}

The judge must be one of the m scored models, otherwise its own output cannot be
told apart from the others. Only ``id`` / ``output_model`` / ``score`` enter the
computation; the remaining metadata is ignored.

Steps: average the other m-1 model scores per sample to get ``y_other``, compute
``y_self / y_other - 1`` for each of the n samples, then report the mean.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logger import logger
from src.utils.score_io import records_by_key, summarize_records

# =============================================================================
# Constants
# =============================================================================

_EPS = 1e-12

# =============================================================================
# Private Functions
# =============================================================================


def _resolve_judge_model(records: list[Any], judge_model: str | None) -> str:
    """Resolve the judge model name used for this evaluation.

    Args:
        records: Judge score records
        judge_model: Judge name given explicitly on the command line; derived
            from the records when ``None``

    Returns:
        The judge model name.

    Raises:
        ValueError: Not given explicitly and the records carry no
            ``judge_model``, or the records hold several different
            ``judge_model`` values
    """
    declared = summarize_records(records, key="output_model")["judge_models"]

    if judge_model is None:
        if not declared:
            raise ValueError("records carry no judge_model metadata; pass --judge-model")
        if len(declared) > 1:
            raise ValueError(
                f"records hold several judge_model values ({'/'.join(declared)}); "
                "pass --judge-model"
            )
        return declared[0]

    if declared and judge_model not in declared:
        logger.warning(f"records name {'/'.join(declared)} as judge; using the given {judge_model}")
    return judge_model


def _collect_scores(
    indexed: dict[int, dict[str, dict[str, Any]]],
) -> tuple[dict[int, dict[str, float]], list[int]]:
    """Extract ``{id: {output_model: score}}`` from the two-level index, skipping invalid scores.

    Args:
        indexed: ``id -> {output_model: record}`` index produced by
            :func:`records_by_key`

    Returns:
        Tuple ``({id: {output_model: score}}, number of invalid entries)``.
    """
    scores: dict[int, dict[str, float]] = {}
    invalid = 0

    for entry_id, models in indexed.items():
        per_model: dict[str, float] = {}
        for output_model, record in models.items():
            try:
                per_model[output_model] = float(record["score"])
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
        if per_model:
            scores[entry_id] = per_model

    if invalid:
        logger.warning(f"skipped {invalid} records whose score is not a number")

    return scores, invalid


def _log_incomplete_grid(
    scores: dict[int, dict[str, float]],
    output_models: list[str],
) -> list[int]:
    """Record and warn about samples that do not cover every scored model.

    Args:
        scores: ``{id: {output_model: score}}``
        output_models: Every scored model seen in the file

    Returns:
        Sorted sample ids whose model set differs from the full set.
    """
    expected = set(output_models)
    incomplete = [
        entry_id for entry_id, per_model in sorted(scores.items()) if set(per_model) != expected
    ]
    if incomplete:
        logger.warning(
            f"{len(incomplete)} samples miss some of the {len(expected)} models: {incomplete}"
        )
    return incomplete


# =============================================================================
# Functions
# =============================================================================


def evaluate_judge_bias(
    records: list[Any],
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Measure how much a judge model prefers its own outputs.

    For each sample, ``y_self`` is the score the judge gave its own model's
    output and ``y_other`` is the mean score it gave the other models' outputs,
    yielding ``y_self / y_other - 1``. The mean of the per-sample values is the
    overall bias.

    Args:
        records: Judge score records shaped like
            ``{"id": int, "output_model": str, "context": str, "judge_model": str,
            "score": float}``, covering n samples x m model outputs
        judge_model: Judge model name; the unique ``judge_model`` in the records
            when ``None``

    Returns:
        Dict with the following keys:
        - ``judge_model`` / ``self_output_model``: judge name (also one of the scored models)
        - ``defined`` / ``undefined_reason``: whether a valid result was obtained;
          when not, the reason is a non-empty string
        - ``num_samples`` / ``num_defined`` / ``num_undefined``: sample count, defined count,
          undefined count
        - ``undefined_ids``: sorted ids of the undefined samples
        - ``mean_error_rate_se`` / ``median_error_rate_se``: mean and median of the
          per-sample bias
        - ``mean_y_self`` / ``mean_y_other``: mean ``y_self`` / ``y_other`` over defined samples
        - ``output_models``: every scored model in the file (deduplicated, sorted)
        - ``num_records`` / ``num_scored`` / ``num_invalid``: total records, successfully
          parsed records, invalid records
        - ``incomplete_ids``: sample ids that do not cover every scored model
        - ``per_sample``: per-sample breakdown (defined samples only)

    Raises:
        ValueError: The judge model name cannot be determined

    Examples:
        >>> records = [
        ...     {"id": 1, "output_model": "gpt-4o", "judge_model": "gpt-4o", "score": 4.0},
        ...     {"id": 1, "output_model": "qwen", "judge_model": "gpt-4o", "score": 2.0},
        ...     {"id": 1, "output_model": "gemini", "judge_model": "gpt-4o", "score": 2.0},
        ... ]
        >>> round(evaluate_judge_bias(records)["mean_error_rate_se"], 3)
        1.0
    """
    judge = _resolve_judge_model(records, judge_model)
    summary = summarize_records(records, key="output_model")
    output_models = summary["output_models"]

    indexed = records_by_key(records, "output_model")
    scores, num_invalid = _collect_scores(indexed)
    if judge not in output_models:
        logger.warning(f"judge {judge} is not among the scored models {output_models}")

    incomplete_ids = _log_incomplete_grid(scores, output_models)

    per_sample: list[dict[str, Any]] = []
    undefined_ids: list[int] = []
    self_values: list[float] = []
    other_values: list[float] = []
    error_rates: list[float] = []

    for entry_id, per_model in scores.items():
        y_self = per_model.get(judge)
        if y_self is None:
            logger.warning(f"sample {entry_id} has no score for its own output {judge}; undefined")
            undefined_ids.append(entry_id)
            continue

        others = [value for model, value in per_model.items() if model != judge]
        if not others:
            logger.warning(f"sample {entry_id} has no other model scores; y_other is undefined")
            undefined_ids.append(entry_id)
            continue

        y_other = float(np.mean(others))
        if y_other <= _EPS:
            logger.warning(f"sample {entry_id} has y_other = {y_other:.4f} <= 0; undefined")
            undefined_ids.append(entry_id)
            continue

        error_rate = y_self / y_other - 1.0
        per_sample.append(
            {
                "id": entry_id,
                "output_model": judge,
                "y_self": float(y_self),
                "y_other": y_other,
                "error_rate_se": float(error_rate),
            }
        )
        self_values.append(float(y_self))
        other_values.append(y_other)
        error_rates.append(float(error_rate))

    defined = bool(error_rates)
    if not defined:
        reason = "no valid sample; every ErrorRate_SE is undefined"
        logger.warning(f"{reason} ({len(scores)} samples, {len(undefined_ids)} undefined)")
    else:
        reason = ""

    return {
        "judge_model": judge,
        "self_output_model": judge,
        "defined": defined,
        "undefined_reason": reason,
        "num_samples": len(scores),
        "num_defined": len(error_rates),
        "num_undefined": len(undefined_ids),
        "undefined_ids": sorted(undefined_ids),
        "mean_error_rate_se": float(np.mean(error_rates)) if defined else 0.0,
        "median_error_rate_se": float(np.median(error_rates)) if defined else 0.0,
        "mean_y_self": float(np.mean(self_values)) if defined else 0.0,
        "mean_y_other": float(np.mean(other_values)) if defined else 0.0,
        "output_models": output_models,
        "num_records": len(records),
        "num_scored": sum(len(per_model) for per_model in scores.values()),
        "num_invalid": num_invalid,
        "incomplete_ids": incomplete_ids,
        "per_sample": per_sample,
    }
