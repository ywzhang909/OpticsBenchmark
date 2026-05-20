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
            # Fallback to older punkt tokenizer
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")


def compute_rouge(pred_answer, gold_answer):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    # Ensure required NLTK resources are available
    ensure_nltk_resources()

    def _rouge_calculation(hypotheses, references, metrics=None):

        if metrics is None:
            metrics = ["rougeL"]
        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        rouge_score = 0
        reference_idx = None
        for idx, ref in enumerate(references):
            score = scorer.score(ref, hypotheses)
            if score["rougeL"].fmeasure > rouge_score:
                rouge_score = score["rougeL"].fmeasure
                reference_idx = idx

        return rouge_score, reference_idx

    # sentence evaluation
    # h = '\n'.join(nltk.sent_tokenize(pred_answer.lower()))
    # r1 = '\n'.join(nltk.sent_tokenize(gold_answer.lower()))

    # document evaluation
    h = pred_answer.lower()
    r1 = [g.lower() for g in gold_answer]
    rouge_score, reference_idx = _rouge_calculation(h, r1)

    return rouge_score, reference_idx

pred_answer = "The cat is on the mat."
gold_answer = ["The cat is on the mat.", "The cat sat on the mat."]

rouge_score, reference_idx = compute_rouge(pred_answer, gold_answer)
