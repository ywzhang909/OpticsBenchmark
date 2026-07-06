from __future__ import annotations

from typing import Any

try:
    from src.algorithm.rouge_eval_utils import compute_rouge
except ImportError:
    pass


class ROGUEScorer:
    @staticmethod
    def calculate_all(predicted: str, reference: str, metrics: list[str] | None = None) -> dict[str, Any]:
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]
        return compute_rouge(predicted, reference, metrics=metrics)
