from __future__ import annotations

import time
from typing import Any

from src.module import AggregatedResults, EvaluationResult

from .base import BaseEvaluator
from .helpers import _get_sentence_embedder, _try_parse_json, normalize_dict_key, sentenceMatch
from .scorer import ROGUEScorer


class RougeEvaluator(BaseEvaluator):
    """
    Evaluator for text generation tasks using ROUGE metrics.

    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores via ROGUEScorer.
    Supports both direct string input and structured dict fields (via info_names config).
    """

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        """Evaluate using ROUGE metrics."""
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            reference = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            reference = normalize_dict_key(reference)

            if isinstance(predicted, dict) and isinstance(reference, dict):
                info_names = normalize_dict_key(
                    self.config.get("info_names", list(reference.keys()))
                )
                all_rouge_metrics: dict[str, list[float]] = {}
                match_model = self.config.get("hungarian_match", {}).get("model", "BAAI/bge-m3")
                for entry_name in info_names:
                    pred_values = predicted.get(entry_name)
                    gold_values = reference.get(entry_name)
                    if pred_values is None or gold_values is None:
                        continue
                    if not isinstance(pred_values, (str, list)) or not isinstance(gold_values, (str, list)):
                        continue
                    if not isinstance(pred_values, list):
                        pred_values = [pred_values]
                    if not isinstance(gold_values, list):
                        gold_values = [gold_values]
                    assignments = sentenceMatch(pred_values, gold_values, embedder=_get_sentence_embedder(match_model))
                    for pred_idx, gold_idx in assignments:
                        rouge_metric = ROGUEScorer.calculate_all(pred_values[pred_idx], gold_values[gold_idx], self.config.get("metrics", None))
                        for metric_name, score in rouge_metric.items():
                            all_rouge_metrics.setdefault(metric_name, []).append(score)

                metrics = {
                    name: sum(scores) / len(scores)
                    for name, scores in all_rouge_metrics.items()
                }

            elif isinstance(predicted, str) and isinstance(reference, str):
                metrics = ROGUEScorer.calculate_all(predicted, reference)
            else:
                metrics = {}

            return EvaluationResult(
                task_id=task_id,
                metrics=metrics,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Error in RougeEvaluator for task {task_id}: {e}")
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate ROUGE evaluation results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
        )
