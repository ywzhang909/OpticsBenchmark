"""
Optis Benchmark - Package Root

This module makes the src directory a proper Python package.
"""

__version__ = "1.0.0"
__author__ = "Optis Benchmark Contributors"

from .core import (
    AgentRunner,
    RunnerConfig,
    TaskConfig,
    TaskInstance,
)
from .environments import (
    BaseEnvironment,
    EnvironmentConfig,
    LocalEnvironment,
    ZOSAPIEnvironment,
)
from .utils import (
    ConfigParser,
    JSONLParser,
    ResultsParser,
    YAMLParser,
    get_logger,
    setup_logger,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseEvaluator",
    "EvaluationResult",
    "AgentRunner",
    "create_evaluator",
    "RunnerConfig",
    "TaskConfig",
    "TaskInstance",
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
