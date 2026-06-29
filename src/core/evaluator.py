"""
OptiS Benchmark - Evaluator Module

This module defines the evaluation logic for optical design tasks
and research-related tasks (paper review, summarization, etc.).
"""

from __future__ import annotations

import json
import math
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
# =============================================================================
# Paper extraction evaluation utilities (from scripts/utils/)
# =============================================================================

try:
    from scripts.utils.bertScore_eval_utils import compute_bert_score
    from scripts.utils.bleu_eval_utils import compute_bleu
    from scripts.utils.cider_eval_utils import compute_cider
    from scripts.utils.citation_eval_utils import compute_citation_f1
    from scripts.utils.edit_distance_utils import (
        levenshtein_distance,
        normalized_edit_similarity,
        word_edit_similarity,
        word_error_rate,
    )
    from scripts.utils.em_eval_utils import (
        compute_exact_match,
        normalize_text,
        record_doi_punctuation,
    )
    from scripts.utils.hungarian_algorithm_utils import hungarian_match
    from scripts.utils.jaccard_similarity_utils import (
        char_ngram_jaccard,
        dice_coefficient,
        jaccard_similarity,
    )
    from scripts.utils.meteor_eval_utils import compute_meteor
    from scripts.utils.perplexity_eval_utils import compute_perplexity
    from scripts.utils.sentence_similarity_utils import SentenceEmbedder
    from scripts.utils.rouge_eval_utils import compute_rouge
except ImportError:
    pass

# =============================================================================
# Helper
# =============================================================================


def _try_parse_json(data: Any) -> Any:
    """Try to parse a JSON string, falling back to the original value."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def normalize_dict_key(data: dict) -> dict:
    """Normalize text content via ``normalize_text``.

    Applies :func:`normalize_text` to every string encountered:
    - ``str`` → normalized string
    - ``list`` → normalized strings inside (non-str items kept as-is)
    - ``dict`` → normalized keys **and** string values (non-str kept as-is)

    Returns:
        Data of the same type with string content normalized.
    """
    if isinstance(data, dict):
        return {
            normalize_text(k) if isinstance(k, str) else k:
            v
            for k, v in data.items()
        }
    return data


def sentenceMatch(
    pred_sentences: list[str],
    gold_sentences: list[str],
    model_name: str = "BAAI/bge-m3",
) -> list[tuple[int, int]]:
    """Match predicted sentences to gold sentences via Hungarian algorithm.

    Encodes both sentence lists with a transformer embedder, computes
    pairwise cosine similarity matrix, then solves the optimal one-to-one
    assignment.

    Args:
        pred_sentences: List of predicted sentences.
        gold_sentences: List of gold/reference sentences.
        model_name: HuggingFace model name for the embedder.

    Returns:
        List of (pred_idx, gold_idx) assignment tuples.
    """
    embedder = SentenceEmbedder(model_name=model_name)
    pred_embs = embedder.encode(pred_sentences)
    gold_embs = embedder.encode(gold_sentences)
    import numpy as np

    sim_matrix = np.dot(pred_embs, gold_embs.T).astype(np.float32)
    assignments, _ = hungarian_match(sim_matrix)
    return assignments


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationResult:
    """Result of evaluating a single task."""

    task_id: str
    success: bool
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_time: float = 0.0
    cost: float = 0.0
    latency: float = 0.0


@dataclass
class AggregatedResults:
    """Aggregated results across multiple tasks."""

    total_tasks: int
    successful_tasks: int
    success_rate: float
    avg_score: float
    avg_execution_time: float
    total_cost: float
    metrics_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    per_task_results: list[EvaluationResult] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Report from validating evaluation results."""

    valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class ModelComparisonResult:
    """Result from comparing two models."""

    model_a_name: str
    model_b_name: str
    mean_a: float
    mean_b: float
    difference: float
    t_statistic: float
    p_value: float
    significant: bool
    winner: str


# =============================================================================
# Base Classes
# =============================================================================


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.

    Evaluators are responsible for scoring agent outputs
    against expected outputs or ground truth.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize evaluator with configuration."""
        self.config = config

    @abstractmethod
    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.

        Args:
            task_id: Unique identifier for the task
            predicted_output: The agent's predicted output
            expected_output: The expected/ground truth output
            metadata: Additional task metadata

        Returns:
            EvaluationResult with score and details
        """
        pass

    @abstractmethod
    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """
        Aggregate results across multiple evaluations.

        Args:
            results: List of individual evaluation results

        Returns:
            AggregatedResults with summary statistics
        """
        pass


# =============================================================================
# Core Evaluators
# =============================================================================


class MetricBasedEvaluator(BaseEvaluator):
    """
    Evaluator that computes numeric metrics for optical design tasks.

    This evaluator is used for tasks where success can be measured
    by specific optical performance metrics (MTF, spot size, etc.).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.scoring_method = config.get("scoring_method", "metric_based")
        self.metrics_config = config.get("metrics", [])
        self.success_criteria = config.get("success_criteria", [])

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate using numeric metrics."""
        start_time = time.time()

        try:
            # Parse outputs
            predicted = _try_parse_json(predicted_output)

            expected = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            expected = normalize_dict_key(expected)

            # Compute metrics
            metrics = self._compute_metrics(predicted, expected, metadata)

            # Check success criteria
            success, score = self._check_success_criteria(metrics)

            return EvaluationResult(
                task_id=task_id,
                success=success,
                score=score,
                metrics=metrics,
                details={
                    "predicted": predicted,
                    "expected": expected,
                    "criteria_met": self._get_criteria_status(metrics),
                },
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _compute_metrics(
        self,
        predicted: dict[str, Any],
        expected: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Compute metrics based on configuration."""
        metrics = {}

        for metric_cfg in self.metrics_config:
            metric_name = metric_cfg.get("name")
            metric_type = metric_cfg.get("type", "numeric")

            if metric_type == "numeric":
                pred_val = predicted.get(metric_name, 0)
                metrics[metric_name] = float(pred_val)
            elif metric_type == "binary":
                pred_val = predicted.get(metric_name, False)
                metrics[metric_name] = 1.0 if pred_val else 0.0
            elif metric_type == "llm_judge":
                metrics[metric_name] = predicted.get(metric_name, 0.5)

        return metrics

    def _check_success_criteria(
        self,
        metrics: dict[str, float],
    ) -> tuple[bool, float]:
        """Check if metrics meet success criteria."""
        if not self.success_criteria:
            return True, 1.0

        all_met = True
        scores = []

        for criterion in self.success_criteria:
            metric_name = criterion.get("metric")
            operator = criterion.get("operator", ">=")
            target_value = criterion.get("value")

            actual_value = metrics.get(metric_name)
            if actual_value is None:
                all_met = False
                scores.append(0.0)
                continue

            met = self._compare(actual_value, operator, target_value)
            scores.append(1.0 if met else 0.0)
            if not met:
                all_met = False

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return all_met, avg_score

    @staticmethod
    def _compare(actual: float, operator: str, target: Any) -> bool:
        """Compare actual value with target using operator."""
        if isinstance(target, bool):
            return bool(actual) == target

        target = float(target)
        if operator == ">=":
            return actual >= target
        elif operator == "<=":
            return actual <= target
        elif operator == ">":
            return actual > target
        elif operator == "<":
            return actual < target
        elif operator == "==":
            return actual == target
        elif operator == "!=":
            return actual != target
        return False

    def _get_criteria_status(
        self,
        metrics: dict[str, float],
    ) -> dict[str, bool]:
        """Get status of each criterion."""
        status = {}
        for criterion in self.success_criteria:
            metric_name = criterion.get("metric")
            operator = criterion.get("operator", ">=")
            target_value = criterion.get("value")
            actual_value = metrics.get(metric_name)

            if actual_value is not None:
                status[metric_name] = self._compare(actual_value, operator, target_value)
            else:
                status[metric_name] = False
        return status

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate results across all tasks."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        total_cost = sum(r.cost for r in results)
        total_time = sum(r.execution_time for r in results)

        # Aggregate metrics
        metrics_summary = {}
        if results and results[0].metrics:
            metric_names = results[0].metrics.keys()
            for name in metric_names:
                values = [r.metrics.get(name, 0) for r in results]
                metrics_summary[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": self._std(values),
                }

        return AggregatedResults(
            total_tasks=total,
            successful_tasks=successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_score=sum(r.score for r in results) / total if total > 0 else 0.0,
            avg_execution_time=total_time / total if total > 0 else 0.0,
            total_cost=total_cost,
            metrics_summary=metrics_summary,
            per_task_results=results,
        )

    @staticmethod
    def _std(values: list[float]) -> float:
        """Compute standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)


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
            # Parse outputs as JSON
            predicted = _try_parse_json(predicted_output)

            expected = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            expected = normalize_dict_key(expected)

            # Iterate over entries and compute exact match scores
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
                success=avg_score >= 0.5,
                score=avg_score,
                metrics={"exact_match_avg": avg_score, "num_entries": len(scores)},
                details={
                    "scores": scores,
                    "num_entries": len(scores),
                },
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate exact match results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)

        return AggregatedResults(
            total_tasks=total,
            successful_tasks=successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_score=sum(r.score for r in results) / total if total > 0 else 0.0,
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            total_cost=sum(r.cost for r in results),
            per_task_results=results,
        )


class PartialMatchEvaluator(BaseEvaluator):
    """Evaluator that checks for partial matches with thresholds."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.8)

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Check for partial match."""
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            expected = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            expected = normalize_dict_key(expected)

            if isinstance(predicted, str) and isinstance(expected, str):
                score = self._string_similarity(predicted, expected)
            elif isinstance(predicted, dict) and isinstance(expected, dict):
                score = self._dict_similarity(predicted, expected)
            else:
                score = 1.0 if predicted == expected else 0.0

            success = score >= self.threshold

            return EvaluationResult(
                task_id=task_id,
                success=success,
                score=score,
                details={"threshold": self.threshold},
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using Jaccard index."""
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _dict_similarity(self, d1: dict, d2: dict) -> float:
        """Compute dictionary similarity."""
        all_keys = set(d1.keys()) | set(d2.keys())
        if not all_keys:
            return 1.0

        matches = 0
        for key in all_keys:
            if d1.get(key) == d2.get(key):
                matches += 1

        return matches / len(all_keys)

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate partial match results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)

        return AggregatedResults(
            total_tasks=total,
            successful_tasks=successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_score=sum(r.score for r in results) / total if total > 0 else 0.0,
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            total_cost=sum(r.cost for r in results),
            per_task_results=results,
        )


class SummarizationEvaluator(MetricBasedEvaluator):
    """
    Evaluator for summarization tasks using ROUGE metrics.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.weight_rouge_1 = config.get("weight_rouge_1", 0.2)
        self.weight_rouge_2 = config.get("weight_rouge_2", 0.3)
        self.weight_rouge_l = config.get("weight_rouge_l", 0.5)

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate summarization using ROUGE metrics."""
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            reference = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            reference = normalize_dict_key(reference)

            if isinstance(predicted, dict):
                field = predicted.get("text") or predicted.get("summary") or predicted.get("output")
                predicted = str(field) if field else str(predicted)
            elif not isinstance(predicted, str):
                predicted = str(predicted)

            if isinstance(reference, dict):
                field = reference.get("text") or reference.get("summary") or reference.get("output")
                reference = str(field) if field else str(reference)
            elif not isinstance(reference, str):
                reference = str(reference)

            # Calculate ROUGE metrics
            rouge_metrics = ROGUEScorer.calculate_all(predicted, reference)

            # Calculate weighted score
            score = (
                self.weight_rouge_1 * rouge_metrics.get("rouge_1_f_score", 0)
                + self.weight_rouge_2 * rouge_metrics.get("rouge_2_f_score", 0)
                + self.weight_rouge_l * rouge_metrics.get("rouge_l_f_score", 0)
            )

            # Calculate content coverage
            pred_words = set(predicted.lower().split())
            ref_words = set(reference.lower().split())
            if ref_words:
                content_coverage = len(pred_words & ref_words) / len(ref_words)
            else:
                content_coverage = 0.0

            metrics = {
                "rouge_1": rouge_metrics.get("rouge_1_f_score", 0),
                "rouge_2": rouge_metrics.get("rouge_2_f_score", 0),
                "rouge_l": rouge_metrics.get("rouge_l_f_score", 0),
                "content_coverage": content_coverage,
                "composite_score": score,
            }

            # Check success criteria
            success, _ = self._check_success_criteria(metrics)

            return EvaluationResult(
                task_id=task_id,
                success=success,
                score=score,
                metrics=metrics,
                details={
                    "predicted": predicted[:500],  # Truncate for storage
                    "reference": reference[:500],
                    "rouge_details": rouge_metrics,
                },
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e),
                execution_time=time.time() - start_time,
            )


# =============================================================================
# Citation/Retrieval Evaluator
# =============================================================================


class CitationEvaluator(MetricBasedEvaluator):
    """
    Evaluator for citation and retrieval tasks.

    Calculates precision, recall, and F1 for retrieved papers.
    """

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate citation accuracy."""
        start_time = time.time()

        try:
            # Parse outputs
            if isinstance(predicted_output, str):
                predicted = json.loads(predicted_output)
            else:
                predicted = predicted_output

            if isinstance(expected_output, str):
                expected = json.loads(expected_output)
            else:
                expected = expected_output

            predicted = normalize_dict_key(predicted)
            expected = normalize_dict_key(expected)

            # Extract paper lists
            pred_papers = self._extract_papers(predicted)
            expected_papers = self._extract_papers(expected)

            # Calculate metrics
            metrics = self._calculate_retrieval_metrics(pred_papers, expected_papers)

            # Calculate citation accuracy
            citation_accuracy = self._calculate_citation_accuracy(pred_papers, expected_papers)
            metrics["citation_accuracy"] = citation_accuracy

            # Composite score
            score = (
                0.3 * metrics.get("recall", 0)
                + 0.3 * metrics.get("precision", 0)
                + 0.4 * metrics.get("citation_accuracy", 0)
            )
            metrics["composite_score"] = score

            # Check success
            success, _ = self._check_success_criteria(metrics)

            return EvaluationResult(
                task_id=task_id,
                success=success,
                score=score,
                metrics=metrics,
                details={
                    "num_predicted": len(pred_papers),
                    "num_expected": len(expected_papers),
                    "correct_papers": list(pred_papers & expected_papers),
                },
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def _extract_papers(self, data: Any) -> set:
        """Extract paper identifiers from data."""
        papers = set()

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Use DOI or title as identifier
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
        """Calculate retrieval metrics (precision, recall, F1)."""
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
        """Calculate how accurately papers are cited."""
        if not expected:
            return 1.0 if not predicted else 0.0

        # Check for exact matches
        exact_matches = len(predicted & expected)

        # Check for partial matches (title similarity)
        partial_matches = 0
        for pred in predicted - expected:
            for exp in expected - predicted:
                if self._title_similarity(pred, exp) > 0.8:
                    partial_matches += 1
                    break

        return (exact_matches + partial_matches * 0.5) / len(expected) if expected else 1.0

    @staticmethod
    def _title_similarity(title1: str, title2: str) -> float:
        """Calculate title similarity."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class RougeEvaluator(BaseEvaluator):
    """
    Evaluator for text generation tasks using ROUGE metrics.

    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores via ROGUEScorer.
    """

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate using ROUGE metrics."""
        start_time = time.time()

        try:
            predicted = _try_parse_json(predicted_output)
            reference = _try_parse_json(expected_output)

            predicted = normalize_dict_key(predicted)
            references = normalize_dict_key(reference)

            info_names = normalize_dict_key(self.config.get("info_names", list(references.keys())))


            for entry_name, pred_value in predicted.items():
                if entry_name not in info_names:
                    continue
                gold_value = references.get(entry_name)
                if gold_value is None:
                    continue
                if isinstance(gold_value, list) and isinstance(pred_value, list) and len(gold_value) > 1:
                    assignments = sentenceMatch(pred_value, gold_value)
                    for pred_idx, gold_idx in assignments:
                        matched_pred = pred_value[pred_idx]
                        matched_gold = gold_value[gold_idx]
                        rouge_metrics = ROGUEScorer.calculate_all(matched_pred, [matched_gold])

            score = (
                rouge_metrics.get("rouge_1_f_score", 0)
                + rouge_metrics.get("rouge_2_f_score", 0)
                + rouge_metrics.get("rouge_l_f_score", 0)
            ) / 3

            metrics = {
                "rouge_1": rouge_metrics.get("rouge_1_f_score", 0),
                "rouge_2": rouge_metrics.get("rouge_2_f_score", 0),
                "rouge_l": rouge_metrics.get("rouge_l_f_score", 0),
                "rouge_1_precision": rouge_metrics.get("rouge_1_precision", 0),
                "rouge_1_recall": rouge_metrics.get("rouge_1_recall", 0),
                "rouge_2_precision": rouge_metrics.get("rouge_2_precision", 0),
                "rouge_2_recall": rouge_metrics.get("rouge_2_recall", 0),
                "rouge_l_precision": rouge_metrics.get("rouge_l_precision", 0),
                "rouge_l_recall": rouge_metrics.get("rouge_l_recall", 0),
            }

            success = score >= 0.5

            return EvaluationResult(
                task_id=task_id,
                success=success,
                score=score,
                metrics=metrics,
                details={
                    "predicted": predicted[:500],
                    "reference": reference[:500],
                    "rouge_details": rouge_metrics,
                },
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return EvaluationResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate ROUGE evaluation results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)

        return AggregatedResults(
            total_tasks=total,
            successful_tasks=successful,
            success_rate=successful / total if total > 0 else 0.0,
            avg_score=sum(r.score for r in results) / total if total > 0 else 0.0,
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            total_cost=sum(r.cost for r in results),
            per_task_results=results,
        )


# =============================================================================
# Scorer Classes (wrapping scripts/utils/ functions)
# =============================================================================


class ROGUEScorer:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scorer.

    Delegates to ``scripts.utils.rouge_eval_utils.compute_rouge``.
    """

    @staticmethod
    def calculate_all(predicted: str, reference: str, metrics: list[str] | None = None) -> dict[str, float]:
        """Calculate ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, etc.).

        Args:
            predicted: Predicted text
            reference: Reference text
            metrics: List of metrics, e.g. ["rouge1", "rouge2", "rougeL"]

        Returns:
            Dict with keys like rouge_1_precision, rouge_1_recall, rouge_1_f_score, etc.
        """
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]
        return compute_rouge(predicted, reference, metrics=metrics)



class ExactMatchScorer:
    """Exact-match scorer, wrapping ``scripts.utils.em_eval_utils``.

    Computes normalized exact match with special DOI punctuation handling.
    """

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answer: str,
        entry_name: str = "",
    ) -> float:
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


class BLEUScorer:
    """BLEU score scorer, wrapping ``scripts.utils.bleu_eval_utils``."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str],
    ) -> dict[str, float]:
        result = compute_bleu(pred_answer, gold_answers)
        return {
            "bleu": result["bleu"],
            "brevity_penalty": result["brevity_penalty"],
        }





class BERTScoreScorer:
    """BERTScore scorer, wrapping ``scripts.utils.bertScore_eval_utils``."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str],
        model_name: str = "microsoft/deberta-xlarge-mnli",
    ) -> dict[str, float]:
        result = compute_bert_score(pred_answer, gold_answers, model_name=model_name)
        return {
            "bertScore_precision": result["precision"],
            "bertScore_recall": result["recall"],
            "bertScore_f1": result["f1"],
        }


class PerplexityScorer:
    """Perplexity scorer, wrapping ``scripts.utils.perplexity_eval_utils``."""

    @classmethod
    def calculate(
        cls,
        text: str,
        model_name: str = "gpt2",
        max_length: int = 1024,
        stride: int = 512,
    ) -> dict[str, float]:
        result = compute_perplexity(text, model_name=model_name, max_length=max_length, stride=stride)
        return {
            "perplexity": result["perplexity"],
            "avg_log_likelihood": result["avg_log_likelihood"],
        }


class MeteorScorer:
    """METEOR scorer, wrapping ``scripts.utils.meteor_eval_utils``."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str],
    ) -> dict[str, float]:
        return {"meteor": compute_meteor(pred_answer, gold_answers)["meteor"]}


class CiderScorer:
    """CIDEr scorer, wrapping ``scripts.utils.cider_eval_utils``."""

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str],
    ) -> dict[str, float]:
        result = compute_cider(pred_answer, gold_answers)
        return {
            "cider": result["cider"],
        }


class SentenceSimilarityScorer:
    """Sentence-similarity scorer, wrapping ``scripts.utils.sentence_similarity_utils``.

    Uses a transformer embedder (default: BAAI/bge-m3) to encode sentences,
    then computes pairwise cosine similarity with Hungarian matching for
    multi-sentence fields.
    """

    @classmethod
    def calculate_similarity_matrix(
        cls,
        pred_sentences: list[str],
        gold_sentences: list[str],
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
    ) -> dict[str, float]:
        embedder = SentenceEmbedder(model_name=model_name, batch_size=batch_size)
        pred_embs = embedder.encode(pred_sentences)
        gold_embs = embedder.encode(gold_sentences)
        import numpy as np

        sim_matrix = np.dot(pred_embs, gold_embs.T).astype(np.float32)
        assignments, total_score = hungarian_match(sim_matrix)
        n = max(len(pred_sentences), len(gold_sentences))
        return {
            "hungarian_total_score": total_score,
            "hungarian_mean_score": total_score / n if n > 0 else 0.0,
            "num_assignments": len(assignments),
        }

    @classmethod
    def calculate_best_match(
        cls,
        pred_sentences: list[str],
        gold_sentences: list[str],
        model_name: str = "BAAI/bge-m3",
    ) -> dict[str, float]:
        embedder = SentenceEmbedder(model_name=model_name)
        pred_embs = embedder.encode(pred_sentences)
        gold_embs = embedder.encode(gold_sentences)
        import numpy as np

        sim_matrix = np.dot(pred_embs, gold_embs.T).astype(np.float32)
        best = float(sim_matrix.max()) if sim_matrix.size > 0 else 0.0
        return {"best_similarity": best}


class CitationScorer:
    """Citation-scoring scorer, wrapping ``scripts.utils.citation_eval_utils``.

    Computes citation precision, recall, and F1 using an NLI-based
    AutoAIS model.
    """

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


# =============================================================================
# Composite Score Calculator
# =============================================================================


class CompositeScore:
    """
    Calculate composite scores from multiple evaluation dimensions.
    """

    DEFAULT_WEIGHTS = {
        "success_rate": 0.30,
        "performance": 0.35,
        "efficiency": 0.20,
        "cost": 0.15,
    }

    @classmethod
    def calculate(
        cls,
        results: list[EvaluationResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Calculate composite score from results.

        Args:
            results: List of evaluation results
            weights: Custom weights for each dimension

        Returns:
            Dict with composite score and component scores
        """
        weights = weights or cls.DEFAULT_WEIGHTS

        success_score = cls.success_rate_score(results)
        performance_score = cls.performance_score(results)
        efficiency_score = cls.efficiency_score(results)
        cost_score = cls.cost_score(results)

        composite = (
            weights.get("success_rate", 0.30) * success_score
            + weights.get("performance", 0.35) * performance_score
            + weights.get("efficiency", 0.20) * efficiency_score
            + weights.get("cost", 0.15) * cost_score
        )

        return {
            "composite_score": composite,
            "success_rate": success_score,
            "performance": performance_score,
            "efficiency": efficiency_score,
            "cost": cost_score,
        }

    @staticmethod
    def success_rate_score(results: list[EvaluationResult]) -> float:
        """Calculate success rate score."""
        if not results:
            return 0.0
        successful = sum(1 for r in results if r.success)
        return successful / len(results)

    @staticmethod
    def performance_score(results: list[EvaluationResult]) -> float:
        """Calculate performance score (average score)."""
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    @staticmethod
    def efficiency_score(results: list[EvaluationResult]) -> float:
        """
        Calculate efficiency score based on execution time.

        Lower is better, so we invert it.
        """
        if not results:
            return 0.0

        avg_time = sum(r.execution_time for r in results) / len(results)

        # Normalize: assume 60s is baseline (score=0.5)
        # 0s -> 1.0, 120s -> 0.0
        if avg_time <= 0:
            return 1.0

        score = max(0, min(1, 1 - avg_time / 120))
        return score

    @staticmethod
    def cost_score(results: list[EvaluationResult]) -> float:
        """
        Calculate cost efficiency score.

        Lower cost is better, so we invert it.
        """
        total_cost = sum(r.cost for r in results)

        if total_cost <= 0:
            return 1.0

        # Normalize: assume $1 is baseline (score=0.5)
        score = max(0, min(1, 1 - total_cost / 2))
        return score


# =============================================================================
# Result Analyzer
# =============================================================================


class ResultAnalyzer:
    """
    Analyze evaluation results with statistical methods.
    """

    @staticmethod
    def compute_statistics(results: list[EvaluationResult]) -> dict[str, Any]:
        """
        Compute statistical metrics from results.

        Args:
            results: List of evaluation results

        Returns:
            Dict with statistical metrics
        """
        if not results:
            return {}

        scores = [r.score for r in results]
        latencies = [r.execution_time for r in results]

        # Sort for percentile calculations
        sorted_scores = sorted(scores)
        sorted_latencies = sorted(latencies)

        n = len(scores)

        def percentile(values: list, p: float) -> float:
            """Calculate percentile."""
            if not values:
                return 0.0
            k = (len(values) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return values[int(k)]
            return values[f] * (c - k) + values[c] * (k - f)

        return {
            # Score statistics
            "num_tasks": n,
            "mean_score": sum(scores) / n,
            "median_score": percentile(sorted_scores, 0.5),
            "std_score": ResultAnalyzer._std(scores),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "p25_score": percentile(sorted_scores, 0.25),
            "p75_score": percentile(sorted_scores, 0.75),
            # Latency statistics
            "mean_latency": sum(latencies) / n,
            "median_latency": percentile(sorted_latencies, 0.5),
            "p95_latency": percentile(sorted_latencies, 0.95),
            "max_latency": max(sorted_latencies) if sorted_latencies else 0.0,
            # Success rate
            "success_count": sum(1 for r in results if r.success),
            "success_rate": sum(1 for r in results if r.success) / n,
            # Cost statistics
            "total_cost": sum(r.cost for r in results),
            "avg_cost": sum(r.cost for r in results) / n,
        }

    @staticmethod
    def _std(values: list[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def compare_models(
        results_a: list[EvaluationResult],
        results_b: list[EvaluationResult],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> ModelComparisonResult:
        """
        Compare two models using statistical tests.

        Args:
            results_a: Results from model A
            results_b: Results from model B
            model_a_name: Name of model A
            model_b_name: Name of model B

        Returns:
            ModelComparisonResult with statistical comparison
        """
        scores_a = [r.score for r in results_a]
        scores_b = [r.score for r in results_b]

        mean_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
        mean_b = sum(scores_b) / len(scores_b) if scores_b else 0.0

        # Welch's t-test (unequal variances)
        var_a = ResultAnalyzer._std(scores_a) ** 2
        var_b = ResultAnalyzer._std(scores_b) ** 2

        n_a, n_b = len(scores_a), len(scores_b)

        if var_a / n_a + var_b / n_b == 0:
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat = (mean_a - mean_b) / math.sqrt(var_a / n_a + var_b / n_b)

            # Approximate degrees of freedom
            num = (var_a / n_a + var_b / n_b) ** 2
            denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
            num / denom if denom > 0 else min(n_a, n_b) - 1

            # Approximate p-value using normal distribution
            p_value = 2 * (1 - ResultAnalyzer._normal_cdf(abs(t_stat)))

        return ModelComparisonResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            mean_a=mean_a,
            mean_b=mean_b,
            difference=mean_a - mean_b,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < 0.05,
            winner="A" if mean_a > mean_b else "B",
        )

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =============================================================================
# Error Analyzer
# =============================================================================


class ErrorAnalyzer:
    """
    Categorize and analyze errors in evaluation results.
    """

    @staticmethod
    def categorize_errors(results: list[EvaluationResult]) -> dict[str, int]:
        """
        Categorize errors by type.

        Args:
            results: List of evaluation results

        Returns:
            Dict mapping error category to count
        """
        categories = {
            "timeout": 0,
            "api_error": 0,
            "invalid_output": 0,
            "quality_below_threshold": 0,
            "success": 0,
            "other": 0,
        }

        for result in results:
            if result.success:
                categories["success"] += 1
            else:
                category = ErrorAnalyzer._classify_error(result)
                categories[category] = categories.get(category, 0) + 1

        return categories

    @staticmethod
    def _classify_error(result: EvaluationResult) -> str:
        """Classify a single error."""
        if not result.error:
            if result.score < 0.5:
                return "quality_below_threshold"
            return "other"

        error_lower = result.error.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "api" in error_lower or "rate limit" in error_lower:
            return "api_error"
        elif "invalid" in error_lower or "parse" in error_lower:
            return "invalid_output"
        elif "quality" in error_lower or "threshold" in error_lower:
            return "quality_below_threshold"
        else:
            return "other"

    @staticmethod
    def get_error_details(results: list[EvaluationResult]) -> list[dict]:
        """Get detailed error information for failed results."""
        errors = []

        for result in results:
            if not result.success and result.error:
                errors.append(
                    {
                        "task_id": result.task_id,
                        "error": result.error,
                        "category": ErrorAnalyzer._classify_error(result),
                        "score": result.score,
                        "execution_time": result.execution_time,
                    }
                )

        return sorted(errors, key=lambda x: x["execution_time"], reverse=True)


# =============================================================================
# Evaluation QA
# =============================================================================


class EvaluationQA:
    """
    Quality assurance for evaluation results.
    """

    @staticmethod
    def validate_results(results: list[EvaluationResult]) -> ValidationReport:
        """
        Validate results for data integrity issues.

        Args:
            results: List of evaluation results

        Returns:
            ValidationReport with issues found
        """
        issues = []

        for i, result in enumerate(results):
            # Check task_id
            if not result.task_id:
                issues.append(f"Result {i}: Missing task_id")

            # Check for negative values
            if result.execution_time < 0:
                issues.append(f"Result {i}: Negative execution_time")
            if result.cost < 0:
                issues.append(f"Result {i}: Negative cost")

            # Check for impossible values
            if result.score < 0 or result.score > 1:
                issues.append(f"Result {i}: Score {result.score} out of range [0, 1]")

            # Check for NaN/Inf
            if math.isnan(result.score) or math.isinf(result.score):
                issues.append(f"Result {i}: Invalid score (NaN/Inf)")

        # Check for statistical outliers
        latencies = [r.execution_time for r in results if r.execution_time >= 0]
        if latencies:
            mean_lat = sum(latencies) / len(latencies)
            std_lat = ResultAnalyzer._std(latencies)
            outliers = [
                (i, r)
                for i, r in enumerate(results)
                if abs(r.execution_time - mean_lat) > 3 * std_lat
            ]
            if outliers:
                issues.append(f"Found {len(outliers)} latency outliers (>3 std from mean)")

        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
        )

    @staticmethod
    def check_consistency(
        results: list[EvaluationResult],
        duplicate_runs: int = 3,
    ) -> dict[str, Any]:
        """
        Check consistency of repeated runs.

        Args:
            results: List of evaluation results
            duplicate_runs: Expected number of duplicate runs per task

        Returns:
            Dict with consistency metrics
        """
        # Group by task_id
        by_task: dict[str, list[EvaluationResult]] = {}
        for result in results:
            task_id = result.task_id
            if task_id not in by_task:
                by_task[task_id] = []
            by_task[task_id].append(result)

        # Check variance per task
        inconsistencies = []
        for task_id, task_results in by_task.items():
            if len(task_results) > 1:
                scores = [r.score for r in task_results]
                std = ResultAnalyzer._std(scores)
                if std > 0.1:  # Threshold for acceptable variance
                    inconsistencies.append(
                        {
                            "task_id": task_id,
                            "num_runs": len(task_results),
                            "std": std,
                            "scores": scores,
                        }
                    )

        return {
            "num_tasks": len(by_task),
            "total_runs": len(results),
            "avg_runs_per_task": len(results) / len(by_task) if by_task else 0,
            "inconsistent_tasks": len(inconsistencies),
            "inconsistency_details": inconsistencies,
        }


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """
    Generate evaluation reports in various formats.
    """

    @staticmethod
    def generate_html_report(
        results: list[EvaluationResult],
        model_name: str,
        task_name: str,
        aggregated: AggregatedResults | None = None,
    ) -> str:
        """
        Generate HTML evaluation report.

        Args:
            results: List of evaluation results
            model_name: Name of the model
            task_name: Name of the task
            aggregated: Pre-aggregated results (optional)

        Returns:
            HTML report string
        """
        stats = ResultAnalyzer.compute_statistics(results)

        # Get composite score
        CompositeScore.calculate(results)

        # Categorize errors
        errors = ErrorAnalyzer.categorize_errors(results)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OptiS Benchmark Report - {model_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; margin-bottom: 8px; }}
        h2 {{ color: #4a5568; margin-bottom: 16px; font-size: 1.2rem; }}
        .meta {{ color: #718096; margin-bottom: 24px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-card.secondary {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-card.tertiary {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .stat-value {{ font-size: 2.5rem; font-weight: bold; margin-bottom: 4px; }}
        .stat-label {{ font-size: 0.85rem; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f7fafc; font-weight: 600; color: #4a5568; }}
        .error-summary {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
        .error-item {{ background: #fff5f5; border: 1px solid #feb2b2; padding: 12px; border-radius: 8px; text-align: center; }}
        .error-item.success {{ background: #f0fff4; border-color: #9ae6b4; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>OptiS Benchmark Report</h1>
            <div class="meta">
                <strong>Model:</strong> {model_name} |
                <strong>Task:</strong> {task_name} |
                <strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>

        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get("success_rate", 0) * 100:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card secondary">
                    <div class="stat-value">{stats.get("mean_score", 0) * 100:.1f}%</div>
                    <div class="stat-label">Avg Score</div>
                </div>
                <div class="stat-card tertiary">
                    <div class="stat-value">{stats.get("mean_latency", 0):.1f}s</div>
                    <div class="stat-label">Avg Latency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.get("total_cost", 0):.4f}</div>
                    <div class="stat-label">Total Cost</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Score Distribution</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Mean</td><td>{stats.get("mean_score", 0):.4f}</td></tr>
                <tr><td>Median</td><td>{stats.get("median_score", 0):.4f}</td></tr>
                <tr><td>Std Dev</td><td>{stats.get("std_score", 0):.4f}</td></tr>
                <tr><td>Min</td><td>{stats.get("min_score", 0):.4f}</td></tr>
                <tr><td>Max</td><td>{stats.get("max_score", 0):.4f}</td></tr>
                <tr><td>P25</td><td>{stats.get("p25_score", 0):.4f}</td></tr>
                <tr><td>P75</td><td>{stats.get("p75_score", 0):.4f}</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Error Summary</h2>
            <div class="error-summary">
                <div class="error-item success"><strong>{errors.get("success", 0)}</strong><br>Success</div>
                <div class="error-item"><strong>{errors.get("timeout", 0)}</strong><br>Timeout</div>
                <div class="error-item"><strong>{errors.get("api_error", 0)}</strong><br>API Error</div>
                <div class="error-item"><strong>{errors.get("quality_below_threshold", 0)}</strong><br>Low Quality</div>
                <div class="error-item"><strong>{errors.get("other", 0)}</strong><br>Other</div>
            </div>
        </div>
    </div>
</body>
</html>"""

        return html

    @staticmethod
    def generate_markdown_report(
        results: list[EvaluationResult],
        model_name: str,
        task_name: str,
    ) -> str:
        """Generate Markdown evaluation report."""
        stats = ResultAnalyzer.compute_statistics(results)
        composite = CompositeScore.calculate(results)
        errors = ErrorAnalyzer.categorize_errors(results)

        md = f"""# OptiS Benchmark Report

## Overview

| Field | Value |
|-------|-------|
| Model | {model_name} |
| Task | {task_name} |
| Generated | {time.strftime("%Y-%m-%d %H:%M:%S")} |

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | {stats.get("num_tasks", 0)} |
| Success Rate | {stats.get("success_rate", 0) * 100:.1f}% |
| Average Score | {stats.get("mean_score", 0) * 100:.1f}% |
| Composite Score | {composite.get("composite_score", 0) * 100:.1f}% |
| Total Cost | ${stats.get("total_cost", 0):.4f} |
| Avg Latency | {stats.get("mean_latency", 0):.1f}s |

## Score Distribution

| Statistic | Value |
|-----------|-------|
| Mean | {stats.get("mean_score", 0):.4f} |
| Median | {stats.get("median_score", 0):.4f} |
| Std Dev | {stats.get("std_score", 0):.4f} |
| Min | {stats.get("min_score", 0):.4f} |
| Max | {stats.get("max_score", 0):.4f} |

## Error Summary

| Category | Count |
|----------|-------|
| Success | {errors.get("success", 0)} |
| Timeout | {errors.get("timeout", 0)} |
| API Error | {errors.get("api_error", 0)} |
| Low Quality | {errors.get("quality_below_threshold", 0)} |
| Other | {errors.get("other", 0)} |

---

*Report generated by OptiS Benchmark*
"""

        return md

# =============================================================================
# Factory Function
# =============================================================================


def create_evaluator(config: dict[str, Any]) -> list[BaseEvaluator]:
    """
    Factory function to create evaluator instances based on eval_metrics config.

    Args:
        config: Evaluator configuration containing eval_metrics dict

    Returns:
        List of configured evaluator instances
    """
    eval_metrics = config.get("eval_metrics", "")

    if eval_metrics == "":
        print("Error: 'eval_metrics' is empty or not configured in the evaluation config.")
        sys.exit(1)

    EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
        "exact_match": ExactMatchEvaluator,
        "partial_match": PartialMatchEvaluator,
        "metric_based": MetricBasedEvaluator,
        "summarization": SummarizationEvaluator,
        "rouge": RougeEvaluator,
        "citation": CitationEvaluator,
        "retrieval": CitationEvaluator
    }

    evaluators: list[BaseEvaluator] = []
    for eval_type, eval_cfg in eval_metrics.items():
        if eval_type in EVALUATOR_MAP:
            cls = EVALUATOR_MAP[eval_type]
            evaluators.append(cls(eval_cfg if eval_cfg else {}))
        else:
            print(f"Warning: Unknown evaluator type '{eval_type}', skipping.")

    if not evaluators:
        print("Error: No valid evaluators created from 'eval_metrics' configuration.")
        sys.exit(1)

    return evaluators
