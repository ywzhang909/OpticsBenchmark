"""
OptiS Benchmark - Test Configuration

Pytest fixtures and configuration for evaluation tests.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.core.evaluator import (
    BertScoreEvaluator,
    CitationEvaluator,
    EvaluationResult,
    ExactMatchEvaluator,
    RougeEvaluator,
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
def sample_results() -> list[EvaluationResult]:
    """Create sample evaluation results for testing."""
    return [
        EvaluationResult(
            task_id="task_001",
            metrics={"mtf": 0.95, "spot_size": 0.005},
            execution_time=10.5,
        ),
        EvaluationResult(
            task_id="task_002",
            metrics={"mtf": 0.88, "spot_size": 0.008},
            execution_time=12.3,
        ),
        EvaluationResult(
            task_id="task_003",
            metrics={"mtf": 0.45, "spot_size": 0.02},
            execution_time=8.7,
        ),
        EvaluationResult(
            task_id="task_004",
            metrics={"mtf": 0.92, "spot_size": 0.006},
            execution_time=11.2,
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
# Composite Scorer fixtures
# =============================================================================


@pytest.fixture
def sample_composite_config() -> dict[str, Any]:
    """Sample composite scoring configuration."""
    from src.core.composite_scorer import CompositeScoreConfig

    return CompositeScoreConfig.default_optical().to_dict()


@pytest.fixture
def sample_composite_scorer():
    """Create a composite scorer instance."""
    from src.core.composite_scorer import CompositeScorer

    return CompositeScorer()


@pytest.fixture
def mock_judge_scores() -> dict[str, float]:
    """Sample judge scores for testing blend logic."""
    return {
        "optical_accuracy": 0.8,
        "metric_correctness": 0.75,
        "output_completeness": 0.9,
        "citation_accuracy": 0.7,
        "reasoning_quality": 0.85,
        "robustness": 0.6,
        "efficiency": 0.95,
        "reproducibility": 1.0,
    }


@pytest.fixture
def mock_static_scores() -> dict[str, float]:
    """Sample static (automated) metric scores."""
    return {
        "optical_accuracy": 0.85,
        "metric_correctness": 0.8,
        "output_completeness": 0.95,
        "citation_accuracy": 0.75,
        "reasoning_quality": 0.7,
        "robustness": 0.65,
        "efficiency": 0.9,
        "reproducibility": 0.95,
    }


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockTask:
    """Mock task for testing."""

    task_id: str
    prompt: str
    expected_output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
