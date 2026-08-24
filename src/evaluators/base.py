"""
Optis Benchmark - Base Evaluator

Defines the BaseEvaluator interface shared by all evaluators.
"""

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

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize evaluator with configuration."""
        self.config = config

    async def setup(self) -> None:
        """加载此评估器所需的 GPU 模型。子类可重写。"""
        pass

    async def teardown(self) -> None:
        """释放 GPU 模型。子类可重写。"""
        pass

    @abstractmethod
    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.

        Args:
            task_id: Unique identifier for the task
            predicted_output: The agent's predicted output
            expected_output: The expected/ground truth output

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
