# meteor_eval_utils.py
"""
METEOR Evaluation Utilities Module

METEOR (Metric for Evaluation of Translation with Explicit ORdering)
evaluates text by computing unigram precision and recall, then combining
them via a harmonic mean with a fragmentation penalty.

Unlike BLEU (precision-oriented), METEOR correlates better with human
judgment because it uses recall, stemming, and synonym matching.

Uses NLTK's METEOR implementation which requires WordNet data.
"""

from __future__ import annotations

from typing import Any


def _ensure_wordnet() -> None:
    """Download WordNet data if not already present."""
    import nltk

    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def _validate_input(pred_answer: str, gold_answer: list) -> dict | None:
    """Return zero-score dict if degenerate, else None."""
    if not gold_answer:
        return {"meteor": 0.0, "precision": 0.0, "recall": 0.0, "frag_penalty": 0.0}
    if not pred_answer or not pred_answer.strip():
        return {"meteor": 0.0, "precision": 0.0, "recall": 0.0, "frag_penalty": 0.0}
    return None


def compute_meteor(
    pred_answer: str,
    gold_answer: list[str],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> dict[str, Any]:
    """Compute METEOR score between a prediction and multiple references.

    When multiple references are provided, the best score across all
    references is returned.

    Args:
        pred_answer: The model output text string.
        gold_answer: A list of reference text strings.
        alpha: Parameter for parameterized harmonic mean (default 0.9).
        beta: Fragmentation penalty parameter (default 3.0).
        gamma: Fragmentation penalty weight (default 0.5).

    Returns:
        dict with keys ``meteor``, ``precision``, ``recall``, ``frag_penalty``.
    """
    early = _validate_input(pred_answer, gold_answer)
    if early is not None:
        return early

    from nltk.translate.meteor_score import meteor_score

    _ensure_wordnet()

    # Tokenize for NLTK's expected format (list of words)
    pred_tokens = pred_answer.split()

    best_score = -1.0
    best_details = {}
    for ref in gold_answer:
        ref_tokens = ref.split()
        score = meteor_score([ref_tokens], pred_tokens, alpha=alpha, beta=beta, gamma=gamma)
        if score > best_score:
            best_score = score
            # NLTK's meteor_score doesn't expose components directly;
            # report the aggregate and approximate components.
            best_details = {
                "meteor": round(score, 4),
                "precision": round(score, 4),  # NLTK returns only the aggregate
                "recall": round(score, 4),
                "frag_penalty": 0.0,
            }

    return best_details


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="METEOR Evaluation")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    parser.add_argument("--gold", type=str, required=True, nargs="+", help="Gold/reference text(s)")
    args = parser.parse_args()

    result = compute_meteor(args.pred, args.gold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
