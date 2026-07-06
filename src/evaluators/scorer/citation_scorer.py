from __future__ import annotations

try:
    from src.algorithm.citation_eval_utils import compute_citation_f1
except ImportError:
    pass


class CitationScorer:
    @classmethod
    def calculate(
        cls,
        question: str,
        pred_answer: str,
        citations: list[dict],
        at_most_citations: int | None = None,
    ) -> dict[str, float]:
        result = compute_citation_f1(question, pred_answer, citations, at_most_citations)
        return {
            "citation_rec": result["citation_rec"],
            "citation_prec": result["citation_prec"],
            "citation_f1": result["citation_f1"],
        }
