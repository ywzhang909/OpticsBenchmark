"""
OptiS Benchmark - Core Module

This module exports the main classes and functions for the benchmark.
"""

from .agent import (
    AgentConfig,
    AgentProvider,
    AgentResponse,
    AnthropicAgent,
    BaseAgent,
    Message,
    OpenAIAgent,
    ToolCall,
    create_agent,
)
from .composite_scorer import (
    CompositeScoreConfig,
    CompositeScorer,
    CoverageReport,
    DimensionCoverage,
    ScoreReport,
    VerificationCatch,
    build_coverage_report,
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
from .llm_judge import DEFAULT_RUBRICS, JudgePromptBuilder, JudgeResult, LLMJudge, Rubric
from .runner import (
    AgentOutput,
    AgentRunner,
    RunnerConfig,
    TaskConfig,
    TaskInstance,
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
    # Composite scoring
    "CompositeScoreConfig",
    "CompositeScorer",
    "CoverageReport",
    "DimensionCoverage",
    "ScoreReport",
    "VerificationCatch",
    "build_coverage_report",
    # LLM Judge
    "DEFAULT_RUBRICS",
    "JudgePromptBuilder",
    "JudgeResult",
    "LLMJudge",
    "Rubric",
    # Runner
    "AgentOutput",
    "AgentRunner",
    "RunnerConfig",
    "TaskConfig",
    "TaskInstance",
]
