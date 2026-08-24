"""
Optis Benchmark - Citation Scorer
"""

from __future__ import annotations

from src.algorithm.citation_eval_utils import compute_citation_f1


class CitationScorer:
    """Scorer that computes NLI-based citation recall / precision / F1."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        citations: list[dict],
        at_most_citations: int | None = None,
    ) -> dict[str, float]:
        """Compute citation metrics for a prediction with cited sources."""
        result = compute_citation_f1(pred_answer, citations, at_most_citations)
        return {
            "citation_rec": result["citation_rec"],
            "citation_prec": result["citation_prec"],
            "citation_f1": result["citation_f1"],
        }
