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


def ensure_nltk_resources():
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


DEFAULT_METRICS = ["rouge1", "rouge2", "rougeL"]


def compute_rouge(pred_answer, gold_answer, metrics=None):
    """Main function for rouge scoring.

    Accepts a single reference or a list of references; when multiple
    references are given, the best score across references is chosen.

    Args:
        pred_answer: predicted text
        gold_answer: reference text (string or list of strings)
        metrics: list of evaluation metrics, e.g. ["rouge1", "rouge2", "rougeL"].
            When omitted, the function returns a single best ROUGE-L F-score
            float; when provided, it returns a per-metric breakdown dict.
    Returns:
        float (best ROUGE-L F-score) when ``metrics`` is None, otherwise a
        dictionary of rouge scores with keys like rouge_1_precision,
        rouge_1_recall, rouge_1_f_score, etc.
    """
    # Ensure required NLTK resources are available
    try:
        ensure_nltk_resources()
    except Exception:
        pass

    references = gold_answer if isinstance(gold_answer, list) else [gold_answer]

    if metrics is None:
        return _best_rouge_l_f(pred_answer, references)
    return _rouge_calculation(pred_answer, references, metrics)


def _best_rouge_l_f(pred_answer, references):
    """Return the best ROUGE-L F-score across all references."""
    ensure_nltk_resources()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    hypothesis = normalize_text(pred_answer)
    best = 0.0
    for reference in references:
        score = scorer.score(normalize_text(reference), hypothesis)["rougeL"].fmeasure
        best = max(best, score)
    return best


def _rouge_calculation(hypothese, references, metrics=None):
    metrics = metrics or DEFAULT_METRICS
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    LABEL_MAP = {"rouge1": "rouge_1", "rouge2": "rouge_2", "rougeL": "rouge_l"}
    hypothesis = normalize_text(hypothese)
    attrs = [("precision", "precision"), ("recall", "recall"), ("fmeasure", "f_score")]

    result: dict[str, float] = {}
    for reference in references:
        scores = scorer.score(normalize_text(reference), hypothesis)
        for m in metrics:
            label = LABEL_MAP.get(m, m)
            for attr, suffix in attrs:
                key = f"{label}_{suffix}"
                val = getattr(scores[m], attr)
                result[key] = max(result.get(key, 0.0), val)
    return result


def main():
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
