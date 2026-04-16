"""
OptiS Benchmark - Test Configuration

Pytest fixtures and configuration for evaluation tests.
"""

import pytest
import asyncio
from typing import Any
from dataclasses import dataclass, field

from src.core.evaluator import (
    EvaluationResult,
    MetricBasedEvaluator,
    ExactMatchEvaluator,
    PartialMatchEvaluator,
    SummarizationEvaluator,
    CitationEvaluator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_metric_config() -> dict[str, Any]:
    """Sample metric-based evaluator configuration."""
    return {
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


@pytest.fixture
def sample_summarization_config() -> dict[str, Any]:
    """Sample summarization evaluator configuration."""
    return {
        "scoring_method": "summarization",
        "weight_rouge_1": 0.2,
        "weight_rouge_2": 0.3,
        "weight_rouge_l": 0.5,
    }


@pytest.fixture
def sample_citation_config() -> dict[str, Any]:
    """Sample citation evaluator configuration."""
    return {
        "scoring_method": "citation",
    }


@pytest.fixture
def metric_evaluator(sample_metric_config: dict) -> MetricBasedEvaluator:
    """Create metric-based evaluator instance."""
    return MetricBasedEvaluator(sample_metric_config)


@pytest.fixture
def exact_evaluator() -> ExactMatchEvaluator:
    """Create exact match evaluator instance."""
    return ExactMatchEvaluator({})


@pytest.fixture
def partial_evaluator() -> PartialMatchEvaluator:
    """Create partial match evaluator instance."""
    return PartialMatchEvaluator({"threshold": 0.8})


@pytest.fixture
def summarization_evaluator(sample_summarization_config: dict) -> SummarizationEvaluator:
    """Create summarization evaluator instance."""
    return SummarizationEvaluator(sample_summarization_config)


@pytest.fixture
def citation_evaluator(sample_citation_config: dict) -> CitationEvaluator:
    """Create citation evaluator instance."""
    return CitationEvaluator(sample_citation_config)


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """Create sample evaluation results for testing."""
    return [
        EvaluationResult(
            task_id="task_001",
            success=True,
            score=0.95,
            metrics={"mtf": 0.95, "spot_size": 0.005},
            execution_time=10.5,
            cost=0.05,
        ),
        EvaluationResult(
            task_id="task_002",
            success=True,
            score=0.88,
            metrics={"mtf": 0.88, "spot_size": 0.008},
            execution_time=12.3,
            cost=0.06,
        ),
        EvaluationResult(
            task_id="task_003",
            success=False,
            score=0.45,
            metrics={"mtf": 0.45, "spot_size": 0.02},
            execution_time=8.7,
            cost=0.04,
        ),
        EvaluationResult(
            task_id="task_004",
            success=True,
            score=0.92,
            metrics={"mtf": 0.92, "spot_size": 0.006},
            execution_time=11.2,
            cost=0.055,
        ),
    ]


@pytest.fixture
def sample_paper_retrieval() -> tuple[dict, dict]:
    """Sample paper retrieval data."""
    predicted = {
        "papers": [
            {"doi": "10.1234/optics.2023.001", "title": "Deep learning for lens design"},
            {"doi": "10.1234/optics.2023.002", "title": "Neural networks in optical systems"},
            {"doi": "10.1234/optics.2023.003", "title": "AI-assisted optical design"},
        ]
    }
    expected = {
        "papers": [
            {"doi": "10.1234/optics.2023.001", "title": "Deep learning for lens design"},
            {"doi": "10.1234/optics.2023.004", "title": "Machine learning optics"},
            {"doi": "10.1234/optics.2023.005", "title": "Computational photography"},
        ]
    }
    return predicted, expected


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockTask:
    """Mock task for testing."""

    task_id: str
    instruction: str
    expected_output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
