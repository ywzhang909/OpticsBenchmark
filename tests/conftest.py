"""
OptiS Benchmark - Test Configuration

Pytest fixtures and configuration for evaluation tests.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.evaluators import (
    BertScoreEvaluator,
    CitationEvaluator,
    ExactMatchEvaluator,
    RougeEvaluator,
    RubricBasedEvaluator,
)
from src.module import EvaluationResult

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
    return {
        "dimensions": [
            {"name": "optical_accuracy", "weight": 0.25},
            {"name": "metric_correctness", "weight": 0.20},
        ],
        "static_weight": 0.7,
        "llm_judge_weight": 0.3,
    }


@pytest.fixture
def sample_composite_scorer():
    """Create a composite scorer instance (stub — module not yet migrated)."""
    return None


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


# =============================================================================
# RubricBasedEvaluator fixtures
# =============================================================================

VLLM_BASE_URL = "https://impecunious909.asia/vllm/v1"
VLLM_MODEL = "qwen"
VLLM_API_KEY = "sk-11235813"


@pytest.fixture
def rubric_judge_config() -> dict[str, Any]:
    """Judge config pointing at a vLLM endpoint (OpenAI-compatible).

    Uses ``raw_http: true`` because the endpoint's Cloudflare WAF blocks
    the official OpenAI Python library's User-Agent.
    """
    return {
        "provider": "openai",
        "model": VLLM_MODEL,
        "api_base": VLLM_BASE_URL,
        "api_key": VLLM_API_KEY,
        "temperature": 0.0,
        "raw_http": True,
    }


@pytest.fixture
def rubric_evaluator_offline() -> RubricBasedEvaluator:
    """RubricBasedEvaluator with no LLM callable (offline mode)."""
    return RubricBasedEvaluator({})


@pytest.fixture
def rubric_evaluator_online(
    rubric_judge_config: dict[str, Any],
) -> RubricBasedEvaluator:
    """RubricBasedEvaluator configured with the live vLLM endpoint."""
    return RubricBasedEvaluator({"judge_config": rubric_judge_config})


@pytest.fixture
def rubric_sample_data() -> dict[str, Any]:
    """Realistic paper info extraction sample for rubric evaluation."""
    return {
        "predicted": {
            "ten keywords": "diffractive optics, meta-lens, "
            "wavefront shaping, computational imaging, PSF",
            "objective": "Design a meta-lens for wide-field "
            "imaging in the visible spectrum",
            "novelty": "Inverse-design algorithm for meta-lens",
            "method": "FDTD simulations with adjoint optimization",
            "performance metrics": "Efficiency: 85%, Strehl: 0.92, FOV: 60°",
        },
        "expected": {
            "ten keywords": "meta-lens, diffractive optics, "
            "wavefront engineering, computational imaging, PSF",
            "objective": "Design and optimize a meta-lens for "
            "wide-field imaging in visible spectrum",
            "novelty": "Novel inverse-design for meta-lens",
            "method": "FDTD with adjoint topology optimization",
            "performance metrics": "Efficiency: 85%, Strehl: 0.95, FOV: 60°",
        },
    }
