"""
Optis Benchmark - Logger Module

This module provides a globally unique logger instance for the benchmark.
Auto-initialized on import with sensible defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import loguru
from loguru import logger as _logger

# =============================================================================
# Constants
# =============================================================================

_CONSOLE_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <4}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

_initialized = False


def _init_default() -> None:
    """Initialize logger with default console handler (called once on import)."""
    global _initialized
    if _initialized:
        return
    _logger.remove()
    _logger.add(
        sys.stderr,
        format=_CONSOLE_FMT,
        level="INFO",
        colorize=True,
    )
    _initialized = True


def setup_logger(
    log_file: str | Path | None = None,
    level: str = "INFO",
    console: bool = True,
    format: str | None = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    compression: str = "zip",
) -> None:
    """Reconfigure the global logger.

    Replaces all existing handlers with the specified configuration.

    Args:
        log_file: Path to log file. If None, only console logging is used.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to enable console (stderr) output.
        rotation: Log rotation size (file handler only).
        retention: Log retention period (file handler only).
        compression: Compression format for rotated logs (file handler only).
    """
    _logger.remove()

    if console:
        _logger.add(
            sys.stderr,
            format=format or _CONSOLE_FMT,
            level=level,
            colorize=True,
        )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_path,
            format=format or _FILE_FMT,
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=True,
        )

    global _initialized
    _initialized = True


def get_logger(name: str | None = None) -> loguru.Logger:
    """Get the global logger instance.

    This always returns the same singleton instance. When *name* is
    provided, the returned logger is bound with that name for context.

    Args:
        name: Optional context name (typically ``__name__``).

    Returns:
        The global logger instance.
    """
    if name:
        return _logger.bind(name=name)
    return _logger


# --- Auto-initialize on import ---
_init_default()

# Public singleton reference
logger = _logger

DEFAULT_FORMAT = _CONSOLE_FMT
