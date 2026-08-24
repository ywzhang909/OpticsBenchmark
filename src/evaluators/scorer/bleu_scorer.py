"""
Optis Benchmark - BLEU Scorer
"""

from __future__ import annotations

from src.algorithm.bleu_eval_utils import compute_bleu

# =============================================================================
# Classes
# =============================================================================


class BLEUScorer:
    """Scorer that computes BLEU for a single prediction."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str],
    ) -> dict[str, float]:
        """Compute BLEU score against gold answers."""
        result = compute_bleu(pred_answer, gold_answers)
        return {
            "bleu": result["bleu"],
            "brevity_penalty": result["brevity_penalty"],
        }
