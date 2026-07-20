from __future__ import annotations

from typing import Any

from src.algorithm.em_eval_utils import normalize_text
from src.algorithm.hungarian_algorithm_utils import hungarian_match
from src.algorithm.model_registry import model_registry
from src.algorithm.sentence_similarity_utils import SentenceEmbedder

_EMBEDDER_PREFIX = "sentence_embedder"


def _get_sentence_embedder(model_name: str = "BAAI/bge-m3") -> SentenceEmbedder:
    """获取或创建缓存的 SentenceEmbedder 实例。"""
    key = f"{_EMBEDDER_PREFIX}:{model_name}"
    return model_registry.get_or_load(
        key,
        lambda: SentenceEmbedder(model_name=model_name),
    )


def unload_sentence_embedder(model_name: str = "BAAI/bge-m3") -> None:
    """显式卸载指定的 SentenceEmbedder，释放 GPU 显存。"""
    model_registry.unload(f"{_EMBEDDER_PREFIX}:{model_name}")


def _try_parse_json(data: Any) -> Any:
    import json

    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def normalize_dict_key(data: dict) -> dict:
    if isinstance(data, dict):
        return {
            normalize_text(k) if isinstance(k, str) else k: v
            for k, v in data.items()
        }
    return data


def sentenceMatch(
    pred_sentences: list[str],
    gold_sentences: list[str],
    model_name: str = "BAAI/bge-m3",
    embedder: SentenceEmbedder | None = None,
) -> list[tuple[int, int]]:
    if embedder is None:
        embedder = _get_sentence_embedder(model_name)
    pred_embs = embedder.encode(pred_sentences)
    gold_embs = embedder.encode(gold_sentences)
    import numpy as np

    sim_matrix = np.dot(pred_embs, gold_embs.T).astype(np.float32)
    assignments, _ = hungarian_match(sim_matrix)
    return assignments
