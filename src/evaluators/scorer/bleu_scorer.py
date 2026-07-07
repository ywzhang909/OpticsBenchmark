from __future__ import annotations

from src.algorithm.bleu_eval_utils import compute_bleu


class BLEUScorer:
    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str],
    ) -> dict[str, float]:
        result = compute_bleu(pred_answer, gold_answers)
        return {
            "bleu": result["bleu"],
            "brevity_penalty": result["brevity_penalty"],
        }
