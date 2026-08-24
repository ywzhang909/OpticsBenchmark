"""
Optis Benchmark - Base Evaluator Tests

Tests for the ExactMatchEvaluator.
"""

import json

import pytest

from src.evaluators import ExactMatchEvaluator
from src.module import EvaluationResult

# =============================================================================
# Classes
# =============================================================================


class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator."""

    @pytest.mark.asyncio
    async def test_exact_match_success(self, exact_evaluator: ExactMatchEvaluator):
        """Test exact match when outputs are identical."""
        output = {"result": "test", "value": 42}

        result = await exact_evaluator.evaluate(
            task_id="test_001",
            predicted_output=output,
            expected_output=output,
        )

        # Only string entries are scored ("value": 42 is skipped)
        assert result.metrics["exact_match_avg"] == 1.0
        assert result.metrics["num_entries"] == 1

    @pytest.mark.asyncio
    async def test_exact_match_failure(self, exact_evaluator: ExactMatchEvaluator):
        """Test exact match when outputs differ."""
        predicted = {"result": "test1"}
        expected = {"result": "test2"}

        result = await exact_evaluator.evaluate(
            task_id="test_002",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.metrics["exact_match_avg"] == 0.0
        assert result.metrics["num_entries"] == 1

    @pytest.mark.asyncio
    async def test_exact_match_json_string(self, exact_evaluator: ExactMatchEvaluator):
        """Test exact match with JSON string inputs."""
        text = json.dumps({"result": "exact match string"})

        result = await exact_evaluator.evaluate(
            task_id="test_003",
            predicted_output=text,
            expected_output=text,
        )

        assert result.metrics["exact_match_avg"] == 1.0
        assert result.metrics["num_entries"] == 1

    @pytest.mark.asyncio
    async def test_exact_match_aggregate(self, exact_evaluator: ExactMatchEvaluator):
        """Test aggregation of exact match results."""
        results = [
            EvaluationResult(
                task_id="task_001",
                metrics={"exact_match_avg": 1.0},
                execution_time=5.0,
            ),
            EvaluationResult(
                task_id="task_002",
                metrics={"exact_match_avg": 0.0},
                execution_time=3.0,
            ),
            EvaluationResult(
                task_id="task_003",
                metrics={"exact_match_avg": 1.0},
                execution_time=4.0,
            ),
        ]

        aggregated = await exact_evaluator.aggregate(results)

        assert aggregated.total_tasks == 3
        assert aggregated.metrics_summary["exact_match_avg"] == pytest.approx(2 / 3)
        assert aggregated.avg_execution_time == pytest.approx(4.0)
