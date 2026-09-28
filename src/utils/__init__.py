"""
Optis Benchmark - Utils Module

This module exports utility functions and classes.
"""

from src.algorithm.bias import evaluate_judge_bias
from src.algorithm.consistency import evaluate_consistency, pearson_correlation

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
from .score_io import (
    cross_check_metadata,
    extract_scores,
    load_scores,
    records_by_id,
    records_by_key,
    summarize_records,
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
    # Score IO
    "load_scores",
    "extract_scores",
    "records_by_id",
    "records_by_key",
    "summarize_records",
    "cross_check_metadata",
    # Consistency (re-exported from src.algorithm.consistency)
    "evaluate_consistency",
    "pearson_correlation",
    # Bias (re-exported from src.algorithm.bias)
    "evaluate_judge_bias",
]
