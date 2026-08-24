"""
Optis Benchmark - BERTScore Scorer
"""

from __future__ import annotations

from src.algorithm.bert_score_eval_utils import compute_bert_score, compute_bert_score_batch


class BERTScoreScorer:
    """Scorer that computes BERTScore precision / recall / F1."""

    @classmethod
    def calculate_batch(
        cls,
        pred_answers: list[str],
        gold_answers: list[str],
        model_name: str = "roberta-large",
    ) -> dict[str, float]:
        """Compute BERTScore metrics for batches of predictions."""
        result = compute_bert_score_batch(pred_answers, gold_answers, model_name=model_name)
        return {
            "bertScore_precision": result["precision"],
            "bertScore_recall": result["recall"],
            "bertScore_f1": result["f1"],
        }

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str] | str,
        model_name: str = "roberta-large",
    ) -> dict[str, float]:
        """Compute BERTScore metrics for a single prediction against references."""
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        if not isinstance(pred_answer, str):
            pred_answer = str(pred_answer)
        result = compute_bert_score(pred_answer, gold_answers, model_name=model_name)
        return {
            "bertScore_precision": result["precision"],
            "bertScore_recall": result["recall"],
            "bertScore_f1": result["f1"],
        }
