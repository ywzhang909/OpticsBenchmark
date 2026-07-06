from __future__ import annotations

import time
from collections import Counter
from typing import Any

from src.module import AggregatedResults, EvaluationResult

from .base import BaseEvaluator
from .helpers import _try_parse_json, normalize_dict_key, normalize_text
from .scorer import ExactMatchScorer


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator that checks for exact matches across JSON entries."""

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate exact match by comparing parsed JSON entries."""
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            expected = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            expected = normalize_dict_key(expected)

            scores: list[float] = []
            info_names = normalize_dict_key(self.config.get("info_names", list(expected.keys())))
            for entry_name, pred_value in predicted.items():
                if entry_name not in info_names:
                    continue
                gold_value = expected.get(entry_name)
                if gold_value is None:
                    continue
                if isinstance(pred_value, str) and isinstance(gold_value, str):
                    score = ExactMatchScorer.calculate(pred_value, gold_value, entry_name)
                    scores.append(score)
                elif isinstance(pred_value, list) and isinstance(gold_value, list):
                    pred_norm = [normalize_text(p) for p in pred_value]
                    gold_norm = [normalize_text(g) for g in gold_value]
                    scores.append(1.0 if Counter(pred_norm) == Counter(gold_norm) else 0.0)

            avg_score = sum(scores) / len(scores) if scores else 0.0

            return EvaluationResult(
                task_id=task_id,
                metrics={"exact_match_avg": avg_score, "num_entries": len(scores)},
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Error in ExactMatchEvaluator for task {task_id}: {e}")
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate exact match results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
        )
