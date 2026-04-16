"""
OptiS Benchmark - Package Root

This module makes the src directory a proper Python package.
"""

__version__ = "1.0.0"
__author__ = "OptiS Benchmark Contributors"

from .core import (
    AgentConfig,
    BaseAgent,
    BaseEvaluator,
    EvaluationResult,
    EvaluationRunner,
    create_agent,
    create_evaluator,
    run_evaluation,
)

from .environments import (
    BaseEnvironment,
    EnvironmentConfig,
    LocalEnvironment,
    ZOSAPIEnvironment,
)

from .utils import (
    setup_logger,
    get_logger,
    JSONLParser,
    YAMLParser,
    ConfigParser,
    ResultsParser,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentConfig",
    "BaseAgent",
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationRunner",
    "create_agent",
    "create_evaluator",
    "run_evaluation",
    # Environments
    "BaseEnvironment",
    "EnvironmentConfig",
    "LocalEnvironment",
    "ZOSAPIEnvironment",
    # Utils
    "setup_logger",
    "get_logger",
    "JSONLParser",
    "YAMLParser",
    "ConfigParser",
    "ResultsParser",
]
