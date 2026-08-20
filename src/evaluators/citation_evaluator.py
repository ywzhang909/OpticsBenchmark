from __future__ import annotations

import json
import time
from typing import Any

from src.algorithm.citation_eval_utils import _get_citation_model, _get_citation_tokenizer, unload_citation_model
from src.module import AggregatedResults, EvaluationResult
from src.utils import logger

from .base import BaseEvaluator
from .scorer import CitationScorer


class CitationEvaluator(BaseEvaluator):
    async def setup(self) -> None:
        """预加载 Citation NLI 模型。"""
        _get_citation_model()
        _get_citation_tokenizer()

    async def teardown(self) -> None:
        """释放 Citation NLI 模型，回收 GPU 显存。"""
        unload_citation_model()

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        start_time = time.time()

        try:
            predicted = str(predicted_output)
            expected = str(expected_output)

            metrics = CitationScorer.calculate(predicted, expected)

            return EvaluationResult(
                task_id=task_id,
                metrics=metrics,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error in CitationEvaluator for task {task_id}: {e}")
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
        )

    @staticmethod
    def _extract_papers(data: Any) -> set[str]:
        """Extract paper identifiers from various input formats."""
        papers: set[str] = set()
        if isinstance(data, dict):
            if "papers" in data:
                for paper in data["papers"]:
                    if isinstance(paper, dict):
                        if "doi" in paper:
                            papers.add(paper["doi"])
                        elif "title" in paper:
                            papers.add(paper["title"].lower())
                    elif isinstance(paper, str):
                        papers.add(paper)
            else:
                for value in data.values():
                    if isinstance(value, str):
                        papers.add(value.lower())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "doi" in item:
                        papers.add(item["doi"])
                    elif "title" in item:
                        papers.add(item["title"].lower())
                elif isinstance(item, str):
                    papers.add(item)
        return papers

    @staticmethod
    def _calculate_retrieval_metrics(predicted: set[str], reference: set[str]) -> dict[str, float]:
        """Calculate precision, recall, F1 for paper retrieval."""
        if not predicted and not reference:
            return {"num_correct": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not predicted or not reference:
            return {"num_correct": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        correct = predicted & reference
        num_correct = len(correct)
        precision = num_correct / len(predicted)
        recall = num_correct / len(reference)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"num_correct": num_correct, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def _title_similarity(title1: str, title2: str) -> float:
        """Calculate word-level Jaccard similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    @staticmethod
    def _calculate_citation_accuracy(predicted: set[str], reference: set[str]) -> float:
        """Calculate citation accuracy considering exact and partial matches."""
        if not reference:
            return 1.0
        exact_matches = predicted & reference
        remaining_ref = reference - exact_matches
        remaining_pred = predicted - exact_matches
        partial = 0
        for ref_item in remaining_ref:
            for pred_item in remaining_pred:
                if CitationEvaluator._title_similarity(pred_item, ref_item) > 0.8:
                    partial += 1
                    break
        return (len(exact_matches) + partial) / len(reference)


