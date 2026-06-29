# rouge_eval_utils.py
"""
ROUGE Score Calculator Module

This module provides functionality to calculate ROUGE scores between generated text and reference texts.
It supports multiple reference texts and chooses the best score for each instance.
"""

import nltk
from rouge_score import rouge_scorer


def ensure_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")


def compute_rouge(pred_answer, gold_answer, metrics=None):
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
    ensure_nltk_resources()

    # document evaluation
    h = pred_answer.lower()
    gold_list = [g.lower() for g in (gold_answer if isinstance(gold_answer, list) else [gold_answer])]

    return _rouge_calculation(h, gold_list, metrics)


def _rouge_calculation(hypotheses, references, metrics=None):
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    LABEL_MAP = {"rouge1": "rouge_1", "rouge2": "rouge_2", "rougeL": "rouge_l"}
    result = {}
    for ref in references:
        scores = scorer.score(ref, hypotheses)
        for m in metrics:
            label = LABEL_MAP.get(m, m)
            for attr, suffix in [("precision", "precision"), ("recall", "recall"), ("fmeasure", "f_score")]:
                key = f"{label}_{suffix}"
                val = getattr(scores[m], attr)
                result[key] = max(result.get(key, 0), val)
    return result
