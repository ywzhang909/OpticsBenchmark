"""
OptiS Benchmark - Utils Module

This module exports utility functions and classes.
"""

from .generate_report import generate_html_report, generate_markdown_report, load_results
from .logger import DEFAULT_FORMAT, get_logger, logger, setup_logger
from .parser import (
    ConfigParser,
    JSONLParser,
    OpticalDataParser,
    ParsedLens,
    ResultsParser,
    YAMLParser,
)

__all__ = [
    # Logger
    "setup_logger",
    "get_logger",
    "DEFAULT_FORMAT",
    "logger",
    # Parser
    "JSONLParser",
    "YAMLParser",
    "ConfigParser",
    "ResultsParser",
    "OpticalDataParser",
    "ParsedLens",
    # Report
    "load_results",
    "generate_html_report",
    "generate_markdown_report",
]
