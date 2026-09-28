"""
Optis Benchmark - Consistency Evaluation Utilities

Assesses how consistent two sets of scores are, using Pearson's r.

Both the correlation and its p-value come from ``scipy.stats.pearsonr`` (scipy is
a hard dependency, imported at module level). Degenerate cases (too few paired
samples, zero variance on either side) are decided up front and reported as
"undefined + reason" rather than relying on a scipy exception or NaN.

The inputs are two lists whose canonical element is a judge score record
``{"id", "output_model", "context", "judge_model", "score"}``
(``id`` is unique within a file, ``score`` is a scalar); the legacy
``{"id", "score"}`` form is also accepted. Pairing intersects on ``id`` and
depends only on ``id`` and ``score``; every other metadata key is ignored.
Pairing iterates ``list_a`` in its original order, so the relative ordering of
the two lists does not matter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from src.utils.logger import logger
from src.utils.score_io import extract_scores

# =============================================================================
# Constants
# =============================================================================

MIN_PAIRED_SAMPLES = 2

# =============================================================================
# Private Functions
# =============================================================================


def _undefined_reason(xs: list[float], ys: list[float]) -> str:
    """Decide whether the paired samples can define a correlation.

    Args:
        xs: First score series
        ys: Second score series (same length as ``xs``)

    Returns:
        Why the correlation is undefined; empty string when it is defined.

    Raises:
        ValueError: The two score series have different lengths
    """
    if len(xs) != len(ys):
        raise ValueError(f"score series lengths differ: {len(xs)} != {len(ys)}")

    n = len(xs)
    if n < MIN_PAIRED_SAMPLES:
        return f"only {n} paired samples, fewer than {MIN_PAIRED_SAMPLES}"

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_ss = float(((x - x.mean()) ** 2).sum())
    y_ss = float(((y - y.mean()) ** 2).sum())
    if x_ss <= 0.0 or y_ss <= 0.0:
        return "score variance is 0 on one side"

    return ""


def _scipy_pearson(xs: list[float], ys: list[float]) -> tuple[float | None, float | None, str]:
    """Compute the Pearson correlation and its two-sided p-value.

    Args:
        xs: First score series
        ys: Second score series (same length as ``xs``)

    Returns:
        Tuple ``(correlation, p_value, undefined_reason)``. The correlation lies
        in [-1.0, 1.0]; it is ``None`` when there are too few paired samples,
        when the variance is 0 on either side, or when the scipy call fails, and
        the reason is then a non-empty string. The p-value is ``None`` whenever
        the correlation is undefined.

    Raises:
        ValueError: The two score series have different lengths
    """
    reason = _undefined_reason(xs, ys)
    if reason:
        return None, None, reason

    try:
        result = stats.pearsonr(xs, ys)
    except Exception as e:
        return None, None, f"correlation computation failed: {e}"

    return float(result.statistic), float(result.pvalue), ""


# =============================================================================
# Functions
# =============================================================================


def pearson_correlation(scores_a: list[float], scores_b: list[float]) -> float | None:
    """Compute the Pearson correlation of two equal-length score series.

    Args:
        scores_a: First score series
        scores_b: Second score series

    Returns:
        The correlation in [-1.0, 1.0], or ``None`` when there are too few paired
        samples, the variance is 0 on either side, or the computation fails.

    Raises:
        ValueError: The two score series have different lengths
    """
    value, _, reason = _scipy_pearson(list(scores_a), list(scores_b))
    if value is None:
        logger.warning(f"{reason}; the correlation is undefined")
    return value


def evaluate_consistency(
    list_a: list[Any],
    list_b: list[Any],
) -> dict[str, Any]:
    """Pair two score sets on their id intersection and assess their consistency.

    Pairing iterates ``list_a`` in its original order and keeps only the ids
    present on both sides, so the relative ordering of the two lists does not
    matter. Only ``id`` and ``score`` are read; metadata keys such as
    ``output_model`` / ``context`` / ``judge_model`` do not affect the result.

    Args:
        list_a: First score set, elements shaped like
            ``{"id": int, "output_model": str, "context": str, "judge_model": str,
            "score": float}``
        list_b: Second score set, same format

    Returns:
        Dict with the following keys:
        - ``pearson_r``: Pearson correlation, ``0.0`` when undefined
        - ``defined``: whether the correlation can be defined
        - ``undefined_reason``: why the correlation is undefined (empty when defined)
        - ``num_pairs``: number of successfully paired samples
        - ``num_only_a`` / ``num_only_b``: ids seen on only one side
        - ``unpaired_ids``: ids that failed to pair (sorted)
        - ``mean_a`` / ``mean_b``: side means over the paired samples
        - ``pearson_p``: two-sided p-value, ``None`` when the correlation is undefined

    Examples:
        >>> evaluate_consistency(
        ...     [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}],
        ...     [{"id": 2, "score": 0.3}, {"id": 1, "score": 0.8}],
        ... )["num_pairs"]
        2
    """
    scores_a, skipped_a = extract_scores(list_a)
    scores_b, skipped_b = extract_scores(list_b)

    if skipped_a or skipped_b:
        logger.warning(f"skipped invalid entries: {skipped_a} in list_a, {skipped_b} in list_b")

    paired_ids = [entry_id for entry_id in scores_a if entry_id in scores_b]
    unpaired_ids = sorted(set(scores_a) ^ set(scores_b))
    if unpaired_ids:
        logger.warning(f"{len(unpaired_ids)} ids are unpaired (one side only): {unpaired_ids}")

    xs = [scores_a[entry_id] for entry_id in paired_ids]
    ys = [scores_b[entry_id] for entry_id in paired_ids]

    pearson_r, p_value, undefined_reason = _scipy_pearson(xs, ys)
    defined = pearson_r is not None

    if not defined:
        logger.warning(f"{undefined_reason}; Pearson r is undefined and recorded as 0.0")

    return {
        "pearson_r": float(pearson_r) if defined else 0.0,
        "pearson_p": p_value,
        "defined": defined,
        "undefined_reason": undefined_reason,
        "num_pairs": len(paired_ids),
        "num_only_a": len(set(scores_a) - set(scores_b)),
        "num_only_b": len(set(scores_b) - set(scores_a)),
        "unpaired_ids": unpaired_ids,
        "mean_a": float(np.mean(xs)) if xs else 0.0,
        "mean_b": float(np.mean(ys)) if ys else 0.0,
    }
