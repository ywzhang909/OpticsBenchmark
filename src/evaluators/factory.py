from __future__ import annotations

from typing import Any

from src.utils.logger import logger

from .base import BaseEvaluator
from .bert_score_evaluator import BertScoreEvaluator
from .citation_evaluator import CitationEvaluator
from .exact_match_evaluator import ExactMatchEvaluator
from .rouge_evaluator import RougeEvaluator

EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
    "exact_match": ExactMatchEvaluator,
    "rouge": RougeEvaluator,
    "bert_score": BertScoreEvaluator,
    "citation": CitationEvaluator,
}


def create_evaluator(config: dict[str, Any]) -> list[BaseEvaluator]:
    """
    Factory function to create evaluator instances based on eval_metrics config.

    Args:
        config: Evaluator configuration containing eval_metrics dict

    Returns:
        List of configured evaluator instances
    """
    eval_metrics = config.get("eval_metrics", "")

    if eval_metrics == "":
        logger.error("'eval_metrics' is empty or not configured in the evaluation config.")
        return []

    evaluators: list[BaseEvaluator] = []
    for eval_type, eval_cfg in eval_metrics.items():
        if eval_type in EVALUATOR_MAP:
            cls = EVALUATOR_MAP[eval_type]
            evaluators.append(cls(eval_cfg if eval_cfg else {}))
        else:
            logger.warning(f"Unknown evaluator type '{eval_type}', skipping.")

    if not evaluators:
        logger.error("No valid evaluators created from 'eval_metrics' configuration.")
        return []

    return evaluators
