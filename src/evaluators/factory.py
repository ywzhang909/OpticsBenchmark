from __future__ import annotations

from typing import Any

from src.utils import logger

from .base import BaseEvaluator
from .bert_score_evaluator import BertScoreEvaluator
from .citation_evaluator import CitationEvaluator
from .exact_match_evaluator import ExactMatchEvaluator
from .qualitative_evaluator import QualitativeEvaluator
from .rouge_evaluator import RougeEvaluator
from .rubric_based_evaluator import RubricBasedEvaluator

EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
    "exact_match": ExactMatchEvaluator,
    "rouge": RougeEvaluator,
    "bert_score": BertScoreEvaluator,
    "citation": CitationEvaluator,
    "qualitative": QualitativeEvaluator,
    "rubric_based": RubricBasedEvaluator,
}

# 评估器按 GPU 开销分类
GPU_INTENSIVE_EVALUATORS = {"citation", "bert_score"}
GPU_LIGHT_EVALUATORS = {"rouge"}
CPU_ONLY_EVALUATORS = {"exact_match", "qualitative", "rubric_based"}
ALL_GPU_EVALUATORS = GPU_INTENSIVE_EVALUATORS | GPU_LIGHT_EVALUATORS


def create_evaluator(config: dict[str, Any]) -> list[tuple[str, BaseEvaluator]]:
    """
    Factory function to create evaluator instances based on eval_metrics config.

    Args:
        config: Evaluator configuration containing eval_metrics dict

    Returns:
        List of (name, evaluator_instance) tuples
    """
    eval_metrics = config.get("eval_metrics", "")

    if eval_metrics == "":
        logger.error("'eval_metrics' is empty or not configured in the evaluation config.")
        return []

    evaluators: list[tuple[str, BaseEvaluator]] = []
    for eval_type, eval_cfg in eval_metrics.items():
        if eval_type in EVALUATOR_MAP:
            cls = EVALUATOR_MAP[eval_type]
            evaluators.append((eval_type, cls(eval_cfg if eval_cfg else {})))
        else:
            logger.warning(f"Unknown evaluator type '{eval_type}', skipping.")

    if not evaluators:
        logger.error("No valid evaluators created from 'eval_metrics' configuration.")
        return []

    return evaluators


def sort_evaluators_by_priority(
    named_evaluators: list[tuple[str, BaseEvaluator]],
    config: dict[str, Any],
) -> list[tuple[str, BaseEvaluator]]:
    """根据 YAML 配置中的 priority 字段对评估器排序。

    priority 值越小越先执行。未配置 priority 的评估器默认为 999。
    同一 priority 内保持原始顺序。

    YAML 配置示例:
        eval_metrics:
          citation:
            priority: 1
          bert_score:
            priority: 2
          rouge:
            priority: 3
          exact_match:
            priority: 4
    """
    eval_metrics_config = config.get("eval_metrics", {})

    def _get_priority(name: str) -> int:
        cfg = eval_metrics_config.get(name, {})
        if isinstance(cfg, dict):
            return cfg.get("priority", 999)
        return 999

    return sorted(named_evaluators, key=lambda x: _get_priority(x[0]))
