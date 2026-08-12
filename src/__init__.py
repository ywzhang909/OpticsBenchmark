"""
Optis Benchmark - Package Root

This module makes the src directory a proper Python package.
"""

__version__ = "1.0.0"
__author__ = "Optis Benchmark Contributors"

from .core import (
    AgentConfig,
    AgentOutput,
    AgentRunner,
    BaseAgent,
    create_agent,
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
    "AgentConfig",
    "AgentOutput",
    "BaseAgent",
    "BaseEvaluator",
    "EvaluationResult",
    "AgentRunner",
    "create_agent",
    "create_evaluator",
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
