"""
Module Package - 评估结果数据模型包

导出评估流程的核心数据结构，供 evaluators 及上层模块使用。
"""

from .result import AggregatedResults, EvaluationResult

__all__ = [
    "AggregatedResults",
    "EvaluationResult",
]
