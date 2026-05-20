# bertScore_eval_utils.py
"""
BERTScore Evaluation Utilities Module

This module provides functionality to compute BERTScore between a predicted
answer and one or more reference answers, returning precision, recall, and F1.
"""

from bert_score import score as bert_score

_MODEL_TYPE = "albert-base-v2"


def compute_bertscore(pred_answer: str, gold_answer: list[str]) -> dict[str, float]:
    """Compute BERTScore between a prediction and multiple reference answers.

    When multiple references are provided, the best F1 score (and its
    corresponding precision and recall) across all references is returned.

    Args:
        pred_answer: The model output text string.
        gold_answer: A list of reference text strings.

    Returns:
        dict with keys 'precision', 'recall', 'f1' containing the best scores.
    """
    candidates = [pred_answer] * len(gold_answer)
    P, R, F1 = bert_score(candidates, gold_answer, lang='en', verbose=False, model_type=_MODEL_TYPE)

    best_idx = F1.argmax().item()
    return {
        'precision': round(P[best_idx].item(), 4),
        'recall': round(R[best_idx].item(), 4),
        'f1': round(F1[best_idx].item(), 4),
    }
