"""
OptiS Benchmark - Integration Tests

End-to-end integration tests for the evaluation system.
"""

import pytest
import asyncio
from src.core.evaluator import (
    MetricBasedEvaluator,
    SummarizationEvaluator,
    CitationEvaluator,
    CompositeScore,
    ResultAnalyzer,
    create_evaluator,
    EvaluationResult,
)


class TestEvaluatorFactory:
    """Tests for evaluator factory function."""

    def test_create_metric_based_evaluator(self):
        """Test creating metric-based evaluator."""
        config = {"scoring_method": "metric_based"}

        evaluator = create_evaluator(config)

        assert isinstance(evaluator, MetricBasedEvaluator)

    def test_create_exact_match_evaluator(self):
        """Test creating exact match evaluator."""
        from src.core.evaluator import ExactMatchEvaluator

        config = {"scoring_method": "exact_match"}

        evaluator = create_evaluator(config)

        assert isinstance(evaluator, ExactMatchEvaluator)

    def test_create_summarization_evaluator(self):
        """Test creating summarization evaluator."""
        config = {"scoring_method": "summarization"}

        evaluator = create_evaluator(config)

        assert isinstance(evaluator, SummarizationEvaluator)

    def test_create_citation_evaluator(self):
        """Test creating citation evaluator."""
        config = {"scoring_method": "citation"}

        evaluator = create_evaluator(config)

        assert isinstance(evaluator, CitationEvaluator)

    def test_create_rouge_alias(self):
        """Test that 'rouge' is accepted as alias for summarization."""
        config = {"scoring_method": "rouge"}

        evaluator = create_evaluator(config)

        assert isinstance(evaluator, SummarizationEvaluator)

    def test_create_retrieval_alias(self):
        """Test that 'retrieval' is accepted as alias for citation."""
        config = {"scoring_method": "retrieval"}

        evaluator = create_evaluator(config)

        assert isinstance(evaluator, CitationEvaluator)

    def test_create_invalid_scoring_method(self):
        """Test that invalid scoring method raises error."""
        config = {"scoring_method": "invalid_method"}

        with pytest.raises(ValueError, match="Unknown scoring method"):
            create_evaluator(config)


class TestEndToEndLensDesign:
    """End-to-end tests for lens design evaluation."""

    @pytest.mark.asyncio
    async def test_lens_design_evaluation(self):
        """Test full lens design evaluation workflow."""
        config = {
            "scoring_method": "metric_based",
            "metrics": [
                {"name": "mtf", "type": "numeric"},
                {"name": "spot_size", "type": "numeric"},
                {"name": "distortion", "type": "numeric"},
            ],
            "success_criteria": [
                {"metric": "mtf", "operator": ">=", "value": 0.8},
                {"metric": "spot_size", "operator": "<=", "value": 0.01},
            ],
        }

        evaluator = MetricBasedEvaluator(config)

        # Simulate multiple lens design tasks
        tasks = [
            (
                "lens_001",
                {"mtf": 0.92, "spot_size": 0.005, "distortion": 0.002},
                {"mtf": 0.9, "spot_size": 0.005, "distortion": 0.001},
            ),
            (
                "lens_002",
                {"mtf": 0.75, "spot_size": 0.015, "distortion": 0.005},
                {"mtf": 0.9, "spot_size": 0.005, "distortion": 0.001},
            ),
            (
                "lens_003",
                {"mtf": 0.88, "spot_size": 0.008, "distortion": 0.003},
                {"mtf": 0.85, "spot_size": 0.01, "distortion": 0.002},
            ),
        ]

        results = []
        for task_id, predicted, expected in tasks:
            result = await evaluator.evaluate(task_id, predicted, expected)
            results.append(result)

        # Aggregate results
        aggregated = await evaluator.aggregate(results)

        # Verify results
        assert aggregated.total_tasks == 3
        assert aggregated.successful_tasks == 2  # lens_001 and lens_003 pass criteria
        assert aggregated.avg_score > 0.5

        # Check composite score
        composite = CompositeScore.calculate(results)
        assert composite["composite_score"] > 0


class TestEndToEndSummarization:
    """End-to-end tests for summarization evaluation."""

    @pytest.mark.asyncio
    async def test_summarization_evaluation(self):
        """Test full summarization evaluation workflow."""
        config = {
            "scoring_method": "summarization",
            "weight_rouge_1": 0.2,
            "weight_rouge_2": 0.3,
            "weight_rouge_l": 0.5,
        }

        evaluator = SummarizationEvaluator(config)

        summaries = [
            (
                "sum_001",
                "Deep learning has revolutionized optical design. Neural networks can optimize lens parameters faster than traditional methods.",
                "Deep learning techniques have transformed optical design. AI can optimize lens parameters more efficiently than traditional approaches.",
            ),
            (
                "sum_002",
                "Machine learning helps with optical system optimization.",
                "Machine learning assists in optimizing optical systems through iterative refinement and automated parameter tuning.",
            ),
        ]

        results = []
        for task_id, predicted, expected in summaries:
            result = await evaluator.evaluate(task_id, predicted, expected)
            results.append(result)

        # Check ROUGE metrics are calculated
        for result in results:
            assert "rouge_1" in result.metrics
            assert "rouge_2" in result.metrics
            assert "rouge_l" in result.metrics

        # Statistics
        stats = ResultAnalyzer.compute_statistics(results)
        assert stats["num_tasks"] == 2
        assert stats["mean_score"] > 0


class TestEndToEndPaperRetrieval:
    """End-to-end tests for paper retrieval evaluation."""

    @pytest.mark.asyncio
    async def test_paper_retrieval_evaluation(self):
        """Test full paper retrieval evaluation workflow."""
        config = {"scoring_method": "citation"}

        evaluator = CitationEvaluator(config)

        retrieval_tasks = [
            (
                "retrieval_001",
                {
                    "papers": [
                        {"doi": "10.1234/optics.2023.001", "title": "DL for Lens Design"},
                        {"doi": "10.1234/optics.2023.002", "title": "Neural Networks in Optics"},
                    ]
                },
                {
                    "papers": [
                        {"doi": "10.1234/optics.2023.001", "title": "DL for Lens Design"},
                        {"doi": "10.1234/optics.2023.003", "title": "ML for Optical Systems"},
                    ]
                },
            ),
            (
                "retrieval_002",
                {
                    "papers": [
                        {"doi": "10.1234/optics.2023.005", "title": "Camera Design"},
                    ]
                },
                {
                    "papers": [
                        {"doi": "10.1234/optics.2023.006", "title": "Photography Optics"},
                    ]
                },
            ),
        ]

        results = []
        for task_id, predicted, expected in retrieval_tasks:
            result = await evaluator.evaluate(task_id, predicted, expected)
            results.append(result)

        # Check retrieval metrics
        for result in results:
            assert "precision" in result.metrics
            assert "recall" in result.metrics
            assert "f1" in result.metrics

        # First task should have better recall (1/2 match)
        assert results[0].metrics["recall"] > results[1].metrics["recall"]


class TestEndToEndModelComparison:
    """End-to-end tests for model comparison."""

    @pytest.mark.asyncio
    async def test_model_comparison_workflow(self):
        """Test full model comparison workflow."""
        from src.core.evaluator import ResultAnalyzer

        # Simulate results from two different models
        model_a_results = [
            EvaluationResult(
                task_id=f"task_{i}",
                success=True,
                score=0.85 + (i % 3) * 0.05,
                execution_time=10.0 + i,
                cost=0.05 + i * 0.01,
            )
            for i in range(5)
        ]

        model_b_results = [
            EvaluationResult(
                task_id=f"task_{i}",
                success=True,
                score=0.80 + (i % 3) * 0.03,
                execution_time=12.0 + i,
                cost=0.04 + i * 0.01,
            )
            for i in range(5)
        ]

        # Compare models
        comparison = ResultAnalyzer.compare_models(
            model_a_results,
            model_b_results,
            "Model A (GPT-4)",
            "Model B (Claude-3)",
        )

        # Model A should have higher average score
        assert comparison.mean_a > comparison.mean_b
        assert comparison.winner == "A"

        # Get individual statistics
        stats_a = ResultAnalyzer.compute_statistics(model_a_results)
        stats_b = ResultAnalyzer.compute_statistics(model_b_results)

        assert stats_a["mean_score"] > stats_b["mean_score"]

        # Calculate composite scores
        composite_a = CompositeScore.calculate(model_a_results)
        composite_b = CompositeScore.calculate(model_b_results)

        assert composite_a["composite_score"] > composite_b["composite_score"]


class TestErrorHandling:
    """Test error handling in evaluation workflows."""

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self):
        """Test handling of invalid JSON input."""
        evaluator = MetricBasedEvaluator(
            {
                "scoring_method": "metric_based",
                "metrics": [{"name": "score", "type": "numeric"}],
            }
        )

        result = await evaluator.evaluate(
            task_id="test",
            predicted_output="{ invalid json }",
            expected_output='{"score": 100}',
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_metrics_handling(self):
        """Test handling of missing metrics."""
        evaluator = MetricBasedEvaluator(
            {
                "scoring_method": "metric_based",
                "metrics": [
                    {"name": "mtf", "type": "numeric"},
                    {"name": "missing_metric", "type": "numeric"},
                ],
            }
        )

        result = await evaluator.evaluate(
            task_id="test",
            predicted_output={"mtf": 0.9},  # Missing missing_metric
            expected_output={"mtf": 0.85, "missing_metric": 0.5},
        )

        # Should still produce a result with partial evaluation
        assert "mtf" in result.metrics
        assert "missing_metric" in result.metrics  # Added with default value

    @pytest.mark.asyncio
    async def test_empty_string_handling(self):
        """Test handling of empty string inputs."""
        evaluator = SummarizationEvaluator(
            {
                "scoring_method": "summarization",
            }
        )

        result = await evaluator.evaluate(
            task_id="test",
            predicted_output="",
            expected_output="This is a reference summary.",
        )

        # Should handle gracefully with low score
        assert result.score >= 0
        assert result.error is None
