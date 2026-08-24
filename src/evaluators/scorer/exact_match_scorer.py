"""
Optis Benchmark - Exact Match Scorer
"""

from __future__ import annotations

from collections import Counter

from src.algorithm.em_eval_utils import compute_exact_match, record_doi_punctuation

# =============================================================================
# Classes
# =============================================================================


class ExactMatchScorer:
    """Scorer that computes normalized exact match between two strings."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answer: str,
        entry_name: str = "",
    ) -> float:
        """Compute exact match score, with special handling for DOI entries."""
        em = float(compute_exact_match(pred_answer, gold_answer))
        if entry_name.lower() == "doi":
            pred_doi_punct = record_doi_punctuation(pred_answer)
            gold_doi_punct = record_doi_punctuation(gold_answer)
            for punct, pred_indices in pred_doi_punct.items():
                gold_indices = gold_doi_punct.get(punct, [])
                if Counter(pred_indices) != Counter(gold_indices):
                    em = 0.0
                    break
        return em
