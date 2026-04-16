"""
OptiS Benchmark - Environments Module

This module exports environment classes for optical design tasks.
"""

from .base_env import (
    BaseEnvironment,
    EnvironmentConfig,
    EnvironmentResponse,
    LocalEnvironment,
    create_environment,
)

from .zos_env import (
    ZOSAPIEnvironment,
    ZOSConnectionConfig,
)

__all__ = [
    # Base
    "BaseEnvironment",
    "EnvironmentConfig",
    "EnvironmentResponse",
    "LocalEnvironment",
    "create_environment",
    # ZOS-API
    "ZOSAPIEnvironment",
    "ZOSConnectionConfig",
]
