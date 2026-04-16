"""
OptiS Benchmark - Core Module

This module exports the main classes and functions for the benchmark.
"""

from .agent import (
    AgentConfig,
    AgentProvider,
    AgentResponse,
    BaseAgent,
    Message,
    OpenAIAgent,
    AnthropicAgent,
    ToolCall,
    create_agent,
)

from .evaluator import (
    AggregatedResults,
    BaseEvaluator,
    EvaluationResult,
    ExactMatchEvaluator,
    MetricBasedEvaluator,
    PartialMatchEvaluator,
    create_evaluator,
)

from .runner import (
    EvaluationRunner,
    RunnerConfig,
    TaskConfig,
    TaskInstance,
    run_evaluation,
)

__all__ = [
    # Agent
    "AgentConfig",
    "AgentProvider", 
    "AgentResponse",
    "BaseAgent",
    "Message",
    "OpenAIAgent",
    "AnthropicAgent",
    "ToolCall",
    "create_agent",
    # Evaluator
    "AggregatedResults",
    "BaseEvaluator",
    "EvaluationResult",
    "ExactMatchEvaluator",
    "MetricBasedEvaluator",
    "PartialMatchEvaluator",
    "create_evaluator",
    # Runner
    "EvaluationRunner",
    "RunnerConfig",
    "TaskConfig",
    "TaskInstance",
    "run_evaluation",
]
