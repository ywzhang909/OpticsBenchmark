from __future__ import annotations

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


