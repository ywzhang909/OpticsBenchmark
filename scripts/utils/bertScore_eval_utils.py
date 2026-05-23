# bertScore_eval_utils.py
"""
BERTScore Evaluation Utilities Module

This module provides functionality to compute BERTScore between a predicted
answer and one or more reference answers, returning precision, recall, and F1.
"""

from bert_score import score as bert_score


def _validate_input(pred_answer: str, gold_answer: list) -> dict | None:
    """Validate inputs and return early if degenerate.

    Returns a zero-score dict if prediction is empty, or None if inputs are valid.
    """
    if not gold_answer:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not pred_answer or not pred_answer.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return None


def compute_bert_score(
    pred_answer: str,
    gold_answer: list[str],
    model_name: str = "albert-base-v2",
) -> dict[str, float]:
    """Compute BERTScore between a prediction and multiple reference answers.

    When multiple references are provided, the best F1 score (and its
    corresponding precision and recall) across all references is returned.

    Args:
        pred_answer: The model output text string.
        gold_answer: A list of reference text strings.
        model_name: HuggingFace model name (default: albert-base-v2).

    Returns:
        dict with keys 'precision', 'recall', 'f1' containing the best scores.
    """
    early = _validate_input(pred_answer, gold_answer)
    if early is not None:
        return early

    candidates = [pred_answer] * len(gold_answer)
    P, R, F1 = bert_score(
        candidates, gold_answer, lang="en", verbose=False, model_type=model_name
    )

    best_idx = F1.argmax().item()
    return {
        "precision": round(P[best_idx].item(), 4),
        "recall": round(R[best_idx].item(), 4),
        "f1": round(F1[best_idx].item(), 4),
    }
