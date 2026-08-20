# rouge_eval_utils.py
"""
ROUGE Score Calculator Module

This module provides functionality to calculate ROUGE scores between generated text and reference texts.
It supports multiple reference texts and chooses the best score for each instance.
"""

import argparse
import json

import nltk
from rouge_score import rouge_scorer

from src.algorithm.em_eval_utils import normalize_text


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except Exception:
        try:
            nltk.download("punkt_tab")
        except Exception:
            try:
                nltk.data.find("tokenizers/punkt")
            except Exception:
                try:
                    nltk.download("punkt")
                except Exception:
                    pass


def compute_rouge(pred_answer: str, gold_answer: str, metrics: list[str] | None = None) -> dict:
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        pred_answer: predicted text
        gold_answer: reference text (string or list of strings)
        metrics: list of evaluation metrics, e.g. ["rouge1", "rouge2", "rougeL"]
    Returns:
        dictionary of rouge scores with keys like rouge_1_precision, rouge_1_recall, rouge_1_f_score, etc.
    """
    # Ensure required NLTK resources are available
    try:
        ensure_nltk_resources()
    except Exception:
        pass

    # document evaluation
    h = normalize_text(pred_answer)
    g = normalize_text(gold_answer)

    return _rouge_calculation(h, g, metrics)


def _rouge_calculation(hypothesis: str, reference: str, metrics: list[str] | None = None) -> dict:
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    label_map = {"rouge1": "rouge_1", "rouge2": "rouge_2", "rougeL": "rouge_l"}
    result = {}
    scores = scorer.score(reference, hypothesis)
    for m in metrics:
        label = label_map.get(m, m)
        for attr, suffix in [("precision", "precision"), ("recall", "recall"), ("fmeasure", "f_score")]:
            key = f"{label}_{suffix}"
            val = getattr(scores[m], attr)
            result[key] = val
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ROUGE Score Evaluation")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    parser.add_argument("--gold", type=str, required=True, help="Gold/reference text")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["rouge1", "rouge2", "rougeL"],
        choices=["rouge1", "rouge2", "rougeL"],
        help="ROUGE metrics to compute",
    )
    args = parser.parse_args()

    result = compute_rouge(args.pred, args.gold, args.metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
