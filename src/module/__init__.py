"""
Module Package - Evaluation result data model package

Exports the core data structures of the evaluation pipeline for use by
evaluators and upper-level modules.
"""

from .result import AggregatedResults, EvaluationResult

__all__ = [
    "AggregatedResults",
    "EvaluationResult",
]
