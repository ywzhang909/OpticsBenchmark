"""
OptiS Benchmark - Evaluator Module

This module defines the evaluation logic for optical design tasks
and research-related tasks (paper review, summarization, etc.).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import logger

# =============================================================================
# Paper extraction evaluation utilities (from scripts/utils/)
# =============================================================================
try:
    from src.algorithm.bertScore_eval_utils import compute_bert_score, compute_bert_score_batch
    from src.algorithm.bleu_eval_utils import compute_bleu
    from src.algorithm.citation_eval_utils import compute_citation_f1
    from src.algorithm.em_eval_utils import (
        compute_exact_match,
        normalize_text,
        record_doi_punctuation,
    )
    from src.algorithm.hungarian_algorithm_utils import hungarian_match
    from src.algorithm.sentence_similarity_utils import SentenceEmbedder
    from src.algorithm.rouge_eval_utils import compute_rouge
except ImportError:
    pass

# Global singleton for SentenceEmbedder — loaded once, reused across all evaluators.
_SENTENCE_EMBEDDER: SentenceEmbedder | None = None

def _get_sentence_embedder(model_name: str = "BAAI/bge-m3") -> SentenceEmbedder:
    global _SENTENCE_EMBEDDER
    if _SENTENCE_EMBEDDER is None:
        _SENTENCE_EMBEDDER = SentenceEmbedder(model_name=model_name)
    return _SENTENCE_EMBEDDER

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
    embedder: SentenceEmbedder | None = None,
) -> list[tuple[int, int]]:
    """Match predicted sentences to gold sentences via Hungarian algorithm.

    Encodes both sentence lists with a transformer embedder, computes
    pairwise cosine similarity matrix, then solves the optimal one-to-one
    assignment.

    Args:
        pred_sentences: List of predicted sentences.
        gold_sentences: List of gold/reference sentences.
        model_name: HuggingFace model name for the embedder (used
            only when ``embedder`` is not provided).
        embedder: Pre-created SentenceEmbedder instance to reuse.
            If ``None``, one is created from ``model_name``.

    Returns:
        List of (pred_idx, gold_idx) assignment tuples.
    """
    if embedder is None:
        embedder = _get_sentence_embedder(model_name)
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
    metrics: dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0

@dataclass
class AggregatedResults:
    """Aggregated results across multiple tasks."""

    total_tasks: int
    metrics_summary: dict[str, float] = field(default_factory=dict)
    avg_execution_time: float = 0.0
    per_task_results: list[EvaluationResult] = field(default_factory=list)

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

    @staticmethod
    def _avg_metrics(results: list[EvaluationResult]) -> dict[str, float]:
        if not results:
            return {}
        names = set()
        for r in results:
            names.update(r.metrics.keys())
        summary = {}
        for name in names:
            values = [r.metrics.get(name, 0) for r in results]
            summary[name] = sum(values) / len(values)
        return summary


# =============================================================================
# Core Evaluators
# =============================================================================
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
                metrics={"exact_match_avg": avg_score, "num_entries": len(scores)},
                execution_time=time.time() - start_time,
            )
        except Exception as e:
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
        metadata: dict[str, Any] | None = None,
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

class BertScoreEvaluator(BaseEvaluator):
    """
    Evaluator for text generation tasks using BERTScore.

    Computes BERTScore precision, recall, and F1 via BERTScoreScorer.
    Supports both direct string input and structured dict fields (via info_names config).
    """

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
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
                    if not isinstance(pred_values, (str, list)) or not isinstance(gold_values, (str, list)):
                        continue
                    if not isinstance(pred_values, list):
                        pred_values = [pred_values]
                    if not isinstance(gold_values, list):
                        gold_values = [gold_values]
                    assignments = sentenceMatch(pred_values, gold_values, embedder=_get_sentence_embedder(match_model))
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

# =============================================================================
# Citation/Retrieval Evaluator
# =============================================================================
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
            # Parse outputs
            predicted = _try_parse_json(predicted_output)
            expected = _try_parse_json(expected_output)

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
        """Aggregate citation evaluation results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=sum(r.execution_time for r in results) / total if total > 0 else 0.0,
            per_task_results=results,
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

# =============================================================================
# Scorer Classes (wrapping scripts/utils/ functions)
# =============================================================================
class ROGUEScorer:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scorer.

    Delegates to ``src.algorithm.rouge_eval_utils.compute_rouge``.
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
    """Exact-match scorer, wrapping ``src.algorithm.em_eval_utils``.

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
    """BLEU score scorer, wrapping ``src.algorithm.bleu_eval_utils``."""

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
    """BERTScore scorer, wrapping ``src.algorithm.bertScore_eval_utils``."""

    @classmethod
    def calculate_batch(
        cls,
        pred_answers: list[str],
        gold_answers: list[str],
        model_name: str = "roberta-large",
    ) -> dict[str, float]:
        result = compute_bert_score_batch(pred_answers, gold_answers, model_name=model_name)
        return {
            "bertScore_precision": result["precision"],
            "bertScore_recall": result["recall"],
            "bertScore_f1": result["f1"],
        }

    @classmethod
    def calculate(
        cls,
        pred_answer: str,
        gold_answers: list[str] | str,
        model_name: str = "roberta-large",
    ) -> dict[str, float]:
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        if not isinstance(pred_answer, str):
            pred_answer = str(pred_answer)
        result = compute_bert_score(pred_answer, gold_answers, model_name=model_name)
        return {
            "bertScore_precision": result["precision"],
            "bertScore_recall": result["recall"],
            "bertScore_f1": result["f1"],
        }

class CitationScorer:
    """Citation-scoring scorer, wrapping ``src.algorithm.citation_eval_utils``.

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
        logger.error("'eval_metrics' is empty or not configured in the evaluation config.")
        return []

    EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
        "exact_match": ExactMatchEvaluator,
        "rouge": RougeEvaluator,
        "bert_score": BertScoreEvaluator,
        "citation": CitationEvaluator,
    }

    evaluators: list[BaseEvaluator] = []
    for eval_type, eval_cfg in eval_metrics.items():
        if eval_type in EVALUATOR_MAP:
            cls = EVALUATOR_MAP[eval_type]
            evaluators.append(cls(eval_cfg if eval_cfg else {}))
        else:
            logger.warning(f"Unknown evaluator type '{eval_type}', skipping.")

    if not evaluators:
        logger.error("No valid evaluators created from 'eval_metrics' configuration.")
        return []

    return evaluators
