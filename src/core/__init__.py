"""
OptiS Benchmark - Core Module

This module exports the main classes and functions for the benchmark.
"""

from .agent import (
    AgentConfig,
    AgentOutput,
    AgentProvider,
    AnthropicAgent,
    BaseAgent,
    Message,
    OpenAIAgent,
    ToolCall,
    create_agent,
)
from .llm_judge import DEFAULT_RUBRICS, JudgePromptBuilder, JudgeResult, LLMJudge, Rubric
from .llm_runner import LLMOutput, LLMPredRunner, LLMRunnerConfig
from .runner import (
    AgentRunner,
    RunnerConfig,
    TaskConfig,
    TaskInstance,
)

__all__ = [
    # Agent
    "AgentConfig",
    "AgentProvider",
    "AgentOutput",
    "BaseAgent",
    "Message",
    "OpenAIAgent",
    "AnthropicAgent",
    "ToolCall",
    "create_agent",
    # Evaluator
    "AggregatedResults",
    "BaseEvaluator",
    "BertScoreEvaluator",
    "CitationEvaluator",
    "EvaluationResult",
    "ExactMatchEvaluator",
    "RougeEvaluator",
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
    # LLM Runner
    "LLMOutput",
    "LLMRunnerConfig",
    "LLMPredRunner",
    # Runner
    "AgentOutput",
    "AgentRunner",
    "RunnerConfig",
    "TaskConfig",
    "TaskInstance",
]
