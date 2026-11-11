"""
Module Result - Evaluation result data models

Defines the core data structures of the evaluation pipeline:
- EvaluationResult: evaluation result of a single task
- AggregatedResults: aggregated results across multiple tasks
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
