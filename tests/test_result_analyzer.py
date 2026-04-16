"""
OptiS Benchmark - Result Analyzer Tests

Tests for ResultAnalyzer, ErrorAnalyzer, CompositeScore, and EvaluationQA.
"""

import pytest
import math
from src.core.evaluator import (
    ResultAnalyzer,
    ErrorAnalyzer,
    CompositeScore,
    EvaluationQA,
    EvaluationResult,
    ModelComparisonResult,
    ValidationReport,
)


class TestResultAnalyzer:
    """Tests for ResultAnalyzer."""

    def test_compute_statistics_empty(self):
        """Test statistics computation with empty results."""
        stats = ResultAnalyzer.compute_statistics([])

        assert stats == {}

    def test_compute_statistics_single_result(self, sample_results: list[EvaluationResult]):
        """Test statistics with single result."""
        stats = ResultAnalyzer.compute_statistics([sample_results[0]])

        assert stats["num_tasks"] == 1
        assert stats["mean_score"] == 0.95
        assert stats["median_score"] == 0.95
        assert stats["success_count"] == 1
        assert stats["success_rate"] == 1.0

    def test_compute_statistics_multiple_results(self, sample_results: list[EvaluationResult]):
        """Test statistics with multiple results."""
        stats = ResultAnalyzer.compute_statistics(sample_results)

        assert stats["num_tasks"] == 4
        assert stats["mean_score"] == pytest.approx((0.95 + 0.88 + 0.45 + 0.92) / 4)
        assert stats["median_score"] == pytest.approx(0.9)
        assert stats["min_score"] == 0.45
        assert stats["max_score"] == 0.95
        assert stats["success_count"] == 3
        assert stats["success_rate"] == pytest.approx(3 / 4)

    def test_compute_statistics_percentiles(self):
        """Test percentile calculations."""
        results = [
            EvaluationResult(task_id=f"task_{i}", success=True, score=i / 10, execution_time=1.0)
            for i in range(1, 11)  # Scores: 0.1, 0.2, ..., 1.0
        ]

        stats = ResultAnalyzer.compute_statistics(results)

        assert stats["p25_score"] == pytest.approx(0.25)
        assert stats["p75_score"] == pytest.approx(0.75)

    def test_compute_statistics_latency(self, sample_results: list[EvaluationResult]):
        """Test latency statistics."""
        stats = ResultAnalyzer.compute_statistics(sample_results)

        expected_latency = (10.5 + 12.3 + 8.7 + 11.2) / 4
        assert stats["mean_latency"] == pytest.approx(expected_latency)
        assert stats["median_latency"] > 0

    def test_compute_statistics_cost(self, sample_results: list[EvaluationResult]):
        """Test cost statistics."""
        stats = ResultAnalyzer.compute_statistics(sample_results)

        total_cost = 0.05 + 0.06 + 0.04 + 0.055
        assert stats["total_cost"] == pytest.approx(total_cost)
        assert stats["avg_cost"] == pytest.approx(total_cost / 4)

    def test_compare_models_identical(self, sample_results: list[EvaluationResult]):
        """Test model comparison with identical performance."""
        # Create identical results
        results_a = sample_results
        results_b = [
            EvaluationResult(
                task_id=r.task_id,
                success=r.success,
                score=r.score,
                execution_time=r.execution_time,
                cost=r.cost,
            )
            for r in sample_results
        ]

        comparison = ResultAnalyzer.compare_models(results_a, results_b, "Model A", "Model B")

        assert comparison.winner == "A"  # A > B when equal (or tie-breaker)
        assert comparison.difference == pytest.approx(0.0, abs=1e-6)

    def test_compare_models_different(self, sample_results: list[EvaluationResult]):
        """Test model comparison with different performance."""
        results_a = sample_results  # avg ~0.8
        results_b = [
            EvaluationResult(
                task_id=r.task_id,
                success=r.success,
                score=r.score * 0.5,  # Lower scores
                execution_time=r.execution_time,
                cost=r.cost,
            )
            for r in sample_results
        ]

        comparison = ResultAnalyzer.compare_models(results_a, results_b, "Model A", "Model B")

        assert comparison.mean_a > comparison.mean_b
        assert comparison.winner == "A"
        assert comparison.difference > 0

    def test_compare_models_statistical_significance(self):
        """Test statistical significance detection."""
        # Create clearly different results
        results_a = [
            EvaluationResult(task_id=f"task_{i}", success=True, score=0.9, execution_time=1.0)
            for i in range(10)
        ]
        results_b = [
            EvaluationResult(task_id=f"task_{i}", success=True, score=0.5, execution_time=1.0)
            for i in range(10)
        ]

        comparison = ResultAnalyzer.compare_models(results_a, results_b, "A", "B")

        assert comparison.significant is True
        assert comparison.p_value < 0.05

    def test_normal_cdf(self):
        """Test normal CDF approximation."""
        # CDF(0) = 0.5
        assert ResultAnalyzer._normal_cdf(0) == pytest.approx(0.5, abs=0.01)

        # CDF(1.96) ≈ 0.975
        assert ResultAnalyzer._normal_cdf(1.96) == pytest.approx(0.975, abs=0.01)

        # CDF(-1.96) ≈ 0.025
        assert ResultAnalyzer._normal_cdf(-1.96) == pytest.approx(0.025, abs=0.01)


class TestErrorAnalyzer:
    """Tests for ErrorAnalyzer."""

    def test_categorize_errors_all_success(self, sample_results: list[EvaluationResult]):
        """Test error categorization with all successes."""
        successful_results = [r for r in sample_results if r.success]

        categories = ErrorAnalyzer.categorize_errors(successful_results)

        assert categories["success"] == len(successful_results)
        assert categories["timeout"] == 0
        assert categories["api_error"] == 0

    def test_categorize_errors_with_errors(self):
        """Test error categorization with various errors."""
        results = [
            EvaluationResult(task_id="1", success=True, score=1.0, execution_time=1.0),
            EvaluationResult(
                task_id="2", success=False, score=0.0, error="Timeout error", execution_time=1.0
            ),
            EvaluationResult(
                task_id="3",
                success=False,
                score=0.0,
                error="API rate limit exceeded",
                execution_time=1.0,
            ),
            EvaluationResult(task_id="4", success=False, score=0.3, execution_time=1.0),
        ]

        categories = ErrorAnalyzer.categorize_errors(results)

        assert categories["success"] == 1
        assert categories["timeout"] == 1
        assert categories["api_error"] == 1
        assert categories["quality_below_threshold"] == 1

    def test_classify_error_timeout(self):
        """Test error classification for timeout."""
        result = EvaluationResult(
            task_id="test",
            success=False,
            score=0.0,
            error="Request timed out after 60s",
            execution_time=60.0,
        )

        category = ErrorAnalyzer._classify_error(result)

        assert category == "timeout"

    def test_classify_error_api(self):
        """Test error classification for API errors."""
        result = EvaluationResult(
            task_id="test",
            success=False,
            score=0.0,
            error="API error: rate limit exceeded",
            execution_time=1.0,
        )

        category = ErrorAnalyzer._classify_error(result)

        assert category == "api_error"

    def test_classify_error_quality(self):
        """Test error classification for quality issues."""
        result = EvaluationResult(
            task_id="test",
            success=False,
            score=0.3,
            error=None,
            execution_time=1.0,
        )

        category = ErrorAnalyzer._classify_error(result)

        assert category == "quality_below_threshold"

    def test_get_error_details(self):
        """Test getting detailed error information."""
        results = [
            EvaluationResult(
                task_id="task_1",
                success=False,
                score=0.2,
                error="Timeout error",
                execution_time=60.0,
            ),
            EvaluationResult(
                task_id="task_2",
                success=False,
                score=0.4,
                error="API error",
                execution_time=5.0,
            ),
            EvaluationResult(
                task_id="task_3",
                success=True,
                score=0.9,
                execution_time=10.0,
            ),
        ]

        details = ErrorAnalyzer.get_error_details(results)

        assert len(details) == 2
        # Sorted by execution time descending
        assert details[0]["task_id"] == "task_1"
        assert details[0]["category"] == "timeout"


class TestCompositeScore:
    """Tests for CompositeScore."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = CompositeScore.DEFAULT_WEIGHTS

        assert weights["success_rate"] == 0.30
        assert weights["performance"] == 0.35
        assert weights["efficiency"] == 0.20
        assert weights["cost"] == 0.15

    def test_success_rate_score(self, sample_results: list[EvaluationResult]):
        """Test success rate calculation."""
        score = CompositeScore.success_rate_score(sample_results)

        assert score == pytest.approx(3 / 4)

    def test_success_rate_score_empty(self):
        """Test success rate with empty results."""
        score = CompositeScore.success_rate_score([])

        assert score == 0.0

    def test_performance_score(self, sample_results: list[EvaluationResult]):
        """Test performance score calculation."""
        score = CompositeScore.performance_score(sample_results)

        expected = (0.95 + 0.88 + 0.45 + 0.92) / 4
        assert score == pytest.approx(expected)

    def test_efficiency_score(self, sample_results: list[EvaluationResult]):
        """Test efficiency score calculation."""
        score = CompositeScore.efficiency_score(sample_results)

        # Based on average execution time
        avg_time = (10.5 + 12.3 + 8.7 + 11.2) / 4
        expected = max(0, min(1, 1 - avg_time / 120))
        assert score == pytest.approx(expected)

    def test_cost_score(self, sample_results: list[EvaluationResult]):
        """Test cost score calculation."""
        score = CompositeScore.cost_score(sample_results)

        total_cost = 0.05 + 0.06 + 0.04 + 0.055
        expected = max(0, min(1, 1 - total_cost / 2))
        assert score == pytest.approx(expected)

    def test_calculate_composite(self, sample_results: list[EvaluationResult]):
        """Test full composite score calculation."""
        result = CompositeScore.calculate(sample_results)

        assert "composite_score" in result
        assert "success_rate" in result
        assert "performance" in result
        assert "efficiency" in result
        assert "cost" in result

        # All scores should be in [0, 1]
        for key, value in result.items():
            assert 0 <= value <= 1, f"{key} = {value} out of range"

    def test_calculate_with_custom_weights(self, sample_results: list[EvaluationResult]):
        """Test composite score with custom weights."""
        custom_weights = {
            "success_rate": 0.5,
            "performance": 0.3,
            "efficiency": 0.1,
            "cost": 0.1,
        }

        result = CompositeScore.calculate(sample_results, weights=custom_weights)

        assert "composite_score" in result


class TestEvaluationQA:
    """Tests for EvaluationQA."""

    def test_validate_results_valid(self, sample_results: list[EvaluationResult]):
        """Test validation with valid results."""
        report = EvaluationQA.validate_results(sample_results)

        assert report.valid is True
        assert len(report.issues) == 0

    def test_validate_results_missing_task_id(self):
        """Test validation with missing task ID."""
        results = [
            EvaluationResult(
                task_id="",  # Empty task ID
                success=True,
                score=0.9,
                execution_time=1.0,
            )
        ]

        report = EvaluationQA.validate_results(results)

        assert report.valid is False
        assert len(report.issues) > 0
        assert any("task_id" in issue.lower() for issue in report.issues)

    def test_validate_results_negative_values(self):
        """Test validation with negative values."""
        results = [
            EvaluationResult(
                task_id="test",
                success=True,
                score=0.9,
                execution_time=-5.0,  # Negative time
                cost=0.05,
            )
        ]

        report = EvaluationQA.validate_results(results)

        assert report.valid is False
        assert any("negative" in issue.lower() for issue in report.issues)

    def test_validate_results_score_out_of_range(self):
        """Test validation with score out of range."""
        results = [
            EvaluationResult(
                task_id="test",
                success=True,
                score=1.5,  # Out of range
                execution_time=1.0,
            )
        ]

        report = EvaluationQA.validate_results(results)

        assert report.valid is False
        assert any("range" in issue.lower() for issue in report.issues)

    def test_validate_results_nan_inf(self):
        """Test validation with NaN/Inf values."""
        results = [
            EvaluationResult(
                task_id="test",
                success=True,
                score=float("nan"),
                execution_time=1.0,
            )
        ]

        report = EvaluationQA.validate_results(results)

        assert report.valid is False
        assert any("nan" in issue.lower() or "inf" in issue.lower() for issue in report.issues)

    def test_check_consistency_consistent(self, sample_results: list[EvaluationResult]):
        """Test consistency check with consistent results."""
        # Duplicate some results
        results = (
            sample_results
            + [
                EvaluationResult(
                    task_id=r.task_id,
                    success=r.success,
                    score=r.score + 0.01,  # Slightly different
                    execution_time=r.execution_time,
                    cost=r.cost,
                )
                for r in sample_results[:2]
            ]
        )

        consistency = EvaluationQA.check_consistency(results, duplicate_runs=2)

        assert consistency["num_tasks"] == 4
        assert consistency["total_runs"] == 6

    def test_check_consistency_inconsistent(self):
        """Test consistency check with inconsistent results."""
        results = [
            EvaluationResult(task_id="task_1", success=True, score=0.9, execution_time=1.0),
            EvaluationResult(
                task_id="task_1", success=True, score=0.3, execution_time=1.0
            ),  # Very different
        ]

        consistency = EvaluationQA.check_consistency(results, duplicate_runs=2)

        assert consistency["inconsistent_tasks"] > 0
        assert len(consistency["inconsistency_details"]) > 0
