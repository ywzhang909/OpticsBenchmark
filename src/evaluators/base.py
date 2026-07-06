from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.module import AggregatedResults, EvaluationResult


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
