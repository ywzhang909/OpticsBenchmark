"""
Optis Benchmark - Evaluators Module

Exports evaluator classes, scorers, and factory helpers.
"""

from .base import BaseEvaluator
from .bert_score_evaluator import BertScoreEvaluator
from .citation_evaluator import CitationEvaluator
from .exact_match_evaluator import ExactMatchEvaluator
from .factory import create_evaluator, sort_evaluators_by_priority
from .helpers import (
    _get_sentence_embedder,
    _try_parse_json,
    normalize_dict_key,
    sentence_match,
    unload_sentence_embedder,
)
from .rouge_evaluator import RougeEvaluator

__all__ = [
    "BaseEvaluator",
    "BertScoreEvaluator",
    "CitationEvaluator",
    "ExactMatchEvaluator",
    "RougeEvaluator",
    "create_evaluator",
    "sort_evaluators_by_priority",
    "_get_sentence_embedder",
    "_try_parse_json",
    "normalize_dict_key",
    "sentence_match",
    "unload_sentence_embedder",
]
