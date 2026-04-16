"""
OptiS Benchmark - Base Evaluator Tests

Tests for MetricBasedEvaluator, ExactMatchEvaluator, and PartialMatchEvaluator.
"""

import pytest
import asyncio
from src.core.evaluator import (
    MetricBasedEvaluator,
    ExactMatchEvaluator,
    PartialMatchEvaluator,
    EvaluationResult,
)


# =============================================================================
# MetricBasedEvaluator Tests
# =============================================================================


class TestMetricBasedEvaluator:
    """Tests for MetricBasedEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_success(
        self,
        metric_evaluator: MetricBasedEvaluator,
    ):
        """Test successful evaluation."""
        predicted = {"mtf": 0.85, "spot_size": 0.008, "distortion": 0.02}
        expected = {"mtf": 0.9, "spot_size": 0.005, "distortion": 0.01}

        result = await metric_evaluator.evaluate(
            task_id="test_001",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert isinstance(result, EvaluationResult)
        assert result.task_id == "test_001"
        assert result.success is True
        assert result.score == 1.0
        assert "mtf" in result.metrics
        assert "spot_size" in result.metrics

    @pytest.mark.asyncio
    async def test_evaluate_failure(
        self,
        metric_evaluator: MetricBasedEvaluator,
    ):
        """Test failed evaluation (criteria not met)."""
        predicted = {"mtf": 0.5, "spot_size": 0.05, "distortion": 0.1}
        expected = {"mtf": 0.9, "spot_size": 0.005, "distortion": 0.01}

        result = await metric_evaluator.evaluate(
            task_id="test_002",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert isinstance(result, EvaluationResult)
        assert result.success is False
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_evaluate_with_json_string(
        self,
        metric_evaluator: MetricBasedEvaluator,
    ):
        """Test evaluation with JSON string inputs."""
        import json

        predicted = json.dumps({"mtf": 0.85, "spot_size": 0.008})
        expected = json.dumps({"mtf": 0.9, "spot_size": 0.005})

        result = await metric_evaluator.evaluate(
            task_id="test_003",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.success is True
        assert "mtf" in result.metrics

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_metrics(
        self,
        metric_evaluator: MetricBasedEvaluator,
    ):
        """Test handling of missing metrics."""
        predicted = {"mtf": 0.85}  # Missing spot_size
        expected = {"mtf": 0.9, "spot_size": 0.005}

        result = await metric_evaluator.evaluate(
            task_id="test_004",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.success is False
        assert "spot_size" in result.metrics

    @pytest.mark.asyncio
    async def test_aggregate_results(
        self,
        metric_evaluator: MetricBasedEvaluator,
    ):
        """Test result aggregation."""
        results = [
            EvaluationResult(
                task_id="task_001",
                success=True,
                score=1.0,
                metrics={"mtf": 0.9, "spot_size": 0.005},
                execution_time=10.0,
                cost=0.05,
            ),
            EvaluationResult(
                task_id="task_002",
                success=False,
                score=0.5,
                metrics={"mtf": 0.5, "spot_size": 0.02},
                execution_time=12.0,
                cost=0.06,
            ),
        ]

        aggregated = await metric_evaluator.aggregate(results)

        assert aggregated.total_tasks == 2
        assert aggregated.successful_tasks == 1
        assert aggregated.success_rate == 0.5
        assert aggregated.avg_score == 0.75
        assert aggregated.avg_execution_time == 11.0
        assert "mtf" in aggregated.metrics_summary

    def test_compare_operators(self):
        """Test comparison operators."""
        assert MetricBasedEvaluator._compare(5.0, ">=", 4.0) is True
        assert MetricBasedEvaluator._compare(5.0, ">=", 5.0) is True
        assert MetricBasedEvaluator._compare(5.0, ">=", 6.0) is False

        assert MetricBasedEvaluator._compare(5.0, "<=", 6.0) is True
        assert MetricBasedEvaluator._compare(5.0, "<=", 5.0) is True
        assert MetricBasedEvaluator._compare(5.0, "<=", 4.0) is False

        assert MetricBasedEvaluator._compare(5.0, ">", 4.0) is True
        assert MetricBasedEvaluator._compare(5.0, "<", 6.0) is True
        assert MetricBasedEvaluator._compare(5.0, "==", 5.0) is True
        assert MetricBasedEvaluator._compare(5.0, "!=", 6.0) is True


# =============================================================================
# ExactMatchEvaluator Tests
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

        assert result.success is True
        assert result.score == 1.0
        assert result.details["exact_match"] is True

    @pytest.mark.asyncio
    async def test_exact_match_failure(self, exact_evaluator: ExactMatchEvaluator):
        """Test exact match when outputs differ."""
        predicted = {"result": "test1", "value": 42}
        expected = {"result": "test2", "value": 42}

        result = await exact_evaluator.evaluate(
            task_id="test_002",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.success is False
        assert result.score == 0.0
        assert result.details["exact_match"] is False

    @pytest.mark.asyncio
    async def test_exact_match_string(self, exact_evaluator: ExactMatchEvaluator):
        """Test exact match with strings."""
        text = "exact match string"

        result = await exact_evaluator.evaluate(
            task_id="test_003",
            predicted_output=text,
            expected_output=text,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_exact_match_aggregate(self, exact_evaluator: ExactMatchEvaluator):
        """Test aggregation of exact match results."""
        results = [
            EvaluationResult(task_id="task_001", success=True, score=1.0, execution_time=5.0),
            EvaluationResult(task_id="task_002", success=False, score=0.0, execution_time=3.0),
            EvaluationResult(task_id="task_003", success=True, score=1.0, execution_time=4.0),
        ]

        aggregated = await exact_evaluator.aggregate(results)

        assert aggregated.total_tasks == 3
        assert aggregated.successful_tasks == 2
        assert aggregated.success_rate == pytest.approx(2 / 3)
        assert aggregated.avg_score == pytest.approx(2 / 3)


# =============================================================================
# PartialMatchEvaluator Tests
# =============================================================================


class TestPartialMatchEvaluator:
    """Tests for PartialMatchEvaluator."""

    @pytest.mark.asyncio
    async def test_partial_match_high_similarity(
        self,
        partial_evaluator: PartialMatchEvaluator,
    ):
        """Test partial match with high similarity."""
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy dog"

        result = await partial_evaluator.evaluate(
            task_id="test_001",
            predicted_output=text1,
            expected_output=text2,
        )

        assert result.success is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_partial_match_medium_similarity(
        self,
        partial_evaluator: PartialMatchEvaluator,
    ):
        """Test partial match with medium similarity."""
        text1 = "the quick brown fox"
        text2 = "the quick red fox"

        result = await partial_evaluator.evaluate(
            task_id="test_002",
            predicted_output=text1,
            expected_output=text2,
        )

        assert 0 < result.score < 1
        assert result.score >= partial_evaluator.threshold

    @pytest.mark.asyncio
    async def test_partial_match_low_similarity(
        self,
        partial_evaluator: PartialMatchEvaluator,
    ):
        """Test partial match with low similarity."""
        text1 = "hello world"
        text2 = "goodbye world"

        result = await partial_evaluator.evaluate(
            task_id="test_003",
            predicted_output=text1,
            expected_output=text2,
        )

        assert result.score < partial_evaluator.threshold
        assert result.success is False

    @pytest.mark.asyncio
    async def test_partial_match_dict(self):
        """Test partial match with dictionaries."""
        evaluator = PartialMatchEvaluator({"threshold": 0.6})

        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"a": 1, "b": 2, "d": 4}

        result = await evaluator.evaluate(
            task_id="test_004",
            predicted_output=dict1,
            expected_output=dict2,
        )

        assert result.score == pytest.approx(2 / 4)  # 2 matches out of 4 keys

    def test_string_similarity_jaccard(self):
        """Test Jaccard similarity calculation."""
        s1 = "the quick brown fox"
        s2 = "the quick red fox"

        similarity = PartialMatchEvaluator({})._string_similarity(s1, s2)

        # Both have: the, quick, fox - 3 common words
        # Union: the, quick, brown, fox, red - 5 words
        # Jaccard = 3/5 = 0.6
        assert similarity == pytest.approx(0.6, rel=0.01)

    def test_string_similarity_empty(self):
        """Test Jaccard similarity with empty strings."""
        evaluator = PartialMatchEvaluator({})

        assert evaluator._string_similarity("", "") == 1.0
        assert evaluator._string_similarity("hello", "") == 0.0
        assert evaluator._string_similarity("", "world") == 0.0

    def test_dict_similarity(self):
        """Test dictionary similarity calculation."""
        evaluator = PartialMatchEvaluator({})

        d1 = {"a": 1, "b": 2}
        d2 = {"a": 1, "c": 3}

        similarity = evaluator._dict_similarity(d1, d2)

        # 1 match (a), 3 total unique keys
        assert similarity == pytest.approx(1 / 3)

    def test_dict_similarity_empty(self):
        """Test dictionary similarity with empty dicts."""
        evaluator = PartialMatchEvaluator({})

        assert evaluator._dict_similarity({}, {}) == 1.0
        assert evaluator._dict_similarity({"a": 1}, {}) == 0.0
