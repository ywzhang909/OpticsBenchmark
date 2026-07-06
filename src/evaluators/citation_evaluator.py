from __future__ import annotations

import time
from typing import Any

from src.module import AggregatedResults, EvaluationResult

from .base import BaseEvaluator
from .helpers import _try_parse_json, normalize_dict_key


class CitationEvaluator(BaseEvaluator):
    """
    Evaluator for citation and retrieval tasks.

    Calculates precision, recall, and F1 for retrieved papers.
    """

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        """Evaluate citation accuracy."""
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            expected = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            expected = normalize_dict_key(expected)

            pred_papers = self._extract_papers(predicted)
            expected_papers = self._extract_papers(expected)

            metrics = self._calculate_retrieval_metrics(pred_papers, expected_papers)

            citation_accuracy = self._calculate_citation_accuracy(pred_papers, expected_papers)
            metrics["citation_accuracy"] = citation_accuracy

            score = (
                0.3 * metrics.get("recall", 0)
                + 0.3 * metrics.get("precision", 0)
                + 0.4 * metrics.get("citation_accuracy", 0)
            )
            metrics["composite_score"] = score

            return EvaluationResult(
                task_id=task_id,
                metrics=metrics,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Error in CitationEvaluator for task {task_id}: {e}")
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate citation evaluation results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
        )

    def _extract_papers(self, data: Any) -> set:
        papers = set()

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    paper_id = item.get("doi") or item.get("title", "").lower()
                    if paper_id:
                        papers.add(paper_id)
                elif isinstance(item, str):
                    papers.add(item.lower())
        elif isinstance(data, dict):
            if "papers" in data:
                return self._extract_papers(data["papers"])

        return papers

    def _calculate_retrieval_metrics(
        self,
        predicted: set,
        expected: set,
    ) -> dict[str, float]:
        if not predicted and not expected:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not expected:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = len(predicted & expected)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_correct": tp,
        }

    def _calculate_citation_accuracy(
        self,
        predicted: set,
        expected: set,
    ) -> float:
        if not expected:
            return 1.0 if not predicted else 0.0

        exact_matches = len(predicted & expected)

        partial_matches = 0
        for pred in predicted - expected:
            for exp in expected - predicted:
                if self._title_similarity(pred, exp) > 0.8:
                    partial_matches += 1
                    break

        return (exact_matches + partial_matches * 0.5) / len(expected) if expected else 1.0

    @staticmethod
    def _title_similarity(title1: str, title2: str) -> float:
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
