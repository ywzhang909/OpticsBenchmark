from __future__ import annotations

import json
import time
from typing import Any

from src.module import AggregatedResults, EvaluationResult

from .base import BaseEvaluator
from .helpers import _try_parse_json
from .scorer import CitationScorer


class CitationEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            expected = _try_parse_json(expected_output)

            pred_answer = self._to_text(predicted)
            citations = self._to_citations(expected)

            metrics = CitationScorer.calculate(pred_answer, citations)

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
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
        )

    @staticmethod
    def _to_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            return " ".join(
                item.get("title", "") if isinstance(item, dict) else str(item)
                for item in data
            )
        if isinstance(data, dict):
            papers = data.get("papers", data)
            if isinstance(papers, list):
                return " ".join(
                    p.get("title", "") if isinstance(p, dict) else str(p)
                    for p in papers
                )
            return json.dumps(data)
        return str(data)

    @staticmethod
    def _to_citations(data: Any) -> list[dict]:
        if isinstance(data, str):
            return []
        if isinstance(data, list):
            return [
                {"title": item.get("title", ""), "text": json.dumps(item)}
                if isinstance(item, dict) else {"title": str(item), "text": str(item)}
                for item in data
            ]
        if isinstance(data, dict):
            papers = data.get("papers")
            if isinstance(papers, list):
                return [
                    {"title": p.get("title", ""), "text": json.dumps(p)}
                    if isinstance(p, dict) else {"title": str(p), "text": str(p)}
                    for p in papers
                ]
            return [{"title": str(k), "text": str(v)} for k, v in data.items()]
        return []
