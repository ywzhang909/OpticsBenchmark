"""
Optis Benchmark - ROUGE Scorer
"""

from __future__ import annotations

from typing import Any

from src.algorithm.rouge_eval_utils import compute_rouge

# =============================================================================
# Classes
# =============================================================================


class ROGUEScorer:
    """Scorer that computes ROUGE metrics."""

    @staticmethod
    def calculate_all(
        predicted: str,
        reference: str,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute the requested ROUGE metrics for one prediction/reference pair."""
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]
        return compute_rouge(predicted, reference, metrics=metrics)
