# bertScore_eval_utils.py
"""
BERTScore Evaluation Utilities Module

This module provides functionality to compute BERTScore between a predicted
answer and one or more reference answers, returning precision, recall, and F1.
"""

from bert_score import BERTScorer

_scorer_cache: dict[str, BERTScorer] = {}


def _get_scorer(model_name: str = "roberta-large") -> BERTScorer:
    """Get or create a cached BERTScorer instance for the given model."""
    if model_name not in _scorer_cache:
        _scorer_cache[model_name] = BERTScorer(lang="en", model_type=model_name)
    return _scorer_cache[model_name]


def _validate_input(pred_answer: str, gold_answer: list) -> dict | None:
    """Validate inputs and return early if degenerate.

    Returns a zero-score dict if prediction is empty, or None if inputs are valid.
    """
    if not gold_answer:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not pred_answer or not pred_answer.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return None


def compute_bert_score_batch(
    pred_answers: list[str],
    gold_answers: list[str],
    model_name: str = "roberta-large",
) -> dict[str, float]:
    """Compute BERTScore for a batch of (pred, gold) pairs in one model invocation.

    Args:
        pred_answers: List of predicted texts.
        gold_answers: List of reference texts (same length as pred_answers).
        model_name: HuggingFace model name (default: roberta-large).

    Returns:
        dict with keys 'precision', 'recall', 'f1' averaged over all pairs.
    """
    if not pred_answers or not gold_answers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    scorer = _get_scorer(model_name)
    P, R, F1 = scorer.score(pred_answers, gold_answers)
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }


def compute_bert_score(
    pred_answer: str,
    gold_answer: list[str],
    model_name: str = "roberta-large",
) -> dict[str, float]:
    """Compute BERTScore between a prediction and multiple reference answers.

    When multiple references are provided, the best F1 score (and its
    corresponding precision and recall) across all references is returned.

    Args:
        pred_answer: The model output text string.
        gold_answer: A list of reference text strings.
        model_name: HuggingFace model name (default: roberta-large).

    Returns:
        dict with keys 'precision', 'recall', 'f1' containing the best scores.
    """
    early = _validate_input(pred_answer, gold_answer)
    if early is not None:
        return early

    scorer = _get_scorer(model_name)
    candidates = [pred_answer] * len(gold_answer)
    P, R, F1 = scorer.score(candidates, gold_answer)

    best_idx = F1.argmax().item()
    return {
        "precision": round(P[best_idx].item(), 4),
        "recall": round(R[best_idx].item(), 4),
        "f1": round(F1[best_idx].item(), 4),
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="BERTScore Evaluation")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    parser.add_argument("--gold", type=str, required=True, nargs="+", help="Gold/reference text(s)")
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-large",
        help="HuggingFace model name (default: roberta-large)",
    )
    args = parser.parse_args()

    try:
        result = compute_bert_score(args.pred, args.gold, model_name=args.model)
    except Exception as e:
        result = {"error": str(e)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
