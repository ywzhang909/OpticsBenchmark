"""
Optis Benchmark - BertScore Evaluator

Evaluates text generation quality using BERTScore, with Hungarian
sentence matching for structured (dict) outputs.
"""

from __future__ import annotations

import time
from typing import Any

from src.algorithm.bert_score_eval_utils import _get_scorer, unload_bert_scorer
from src.module import AggregatedResults, EvaluationResult
from src.utils import logger

from .base import BaseEvaluator
from .helpers import (
    _get_sentence_embedder,
    _try_parse_json,
    normalize_dict_key,
    sentence_match,
    unload_sentence_embedder,
)
from .scorer import BERTScoreScorer

# =============================================================================
# Classes
# =============================================================================


class BertScoreEvaluator(BaseEvaluator):
    """
    Evaluator for text generation tasks using BERTScore.

    Computes BERTScore precision, recall, and F1 via BERTScoreScorer.
    Supports both direct string input and structured dict fields (via info_names config).
    """

    async def setup(self) -> None:
        """预加载 BERTScorer 和 SentenceEmbedder 模型。"""
        model_name = self.config.get("model_name", "roberta-large")
        match_model = self.config.get("hungarian_match", {}).get("model", "BAAI/bge-m3")
        _get_scorer(model_name)
        _get_sentence_embedder(match_model)

    async def teardown(self) -> None:
        """释放 BERTScorer 和 SentenceEmbedder 模型，回收 GPU 显存。"""
        model_name = self.config.get("model_name", "roberta-large")
        match_model = self.config.get("hungarian_match", {}).get("model", "BAAI/bge-m3")
        unload_bert_scorer(model_name)
        unload_sentence_embedder(match_model)

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        """Evaluate using BERTScore metrics."""
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
                model_name = self.config.get("model_name", "roberta-large")
                match_model = self.config.get("hungarian_match", {}).get("model", "BAAI/bge-m3")
                all_preds: list[str] = []
                all_golds: list[str] = []
                for entry_name in info_names:
                    pred_values = predicted.get(entry_name)
                    gold_values = reference.get(entry_name)
                    if pred_values is None or gold_values is None:
                        continue
                    if not isinstance(pred_values, (str, list)) or not isinstance(
                        gold_values, (str, list)
                    ):
                        continue
                    if not isinstance(pred_values, list):
                        pred_values = [pred_values]
                    if not isinstance(gold_values, list):
                        gold_values = [gold_values]
                    assignments = sentence_match(
                        pred_values,
                        gold_values,
                        embedder=_get_sentence_embedder(match_model),
                    )
                    for pred_idx, gold_idx in assignments:
                        all_preds.append(pred_values[pred_idx])
                        all_golds.append(gold_values[gold_idx])

                metrics = (
                    BERTScoreScorer.calculate_batch(all_preds, all_golds, model_name)
                    if all_preds
                    else {}
                )

            elif isinstance(predicted, str) and isinstance(reference, str):
                metrics = BERTScoreScorer.calculate(predicted, [reference])
            elif isinstance(predicted, str) and isinstance(reference, list):
                metrics = BERTScoreScorer.calculate(predicted, reference)
            else:
                metrics = {}

            return EvaluationResult(
                task_id=task_id,
                metrics=metrics,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error in BertScoreEvaluator for task {task_id}: {e}")
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate BERTScore evaluation results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
        )
