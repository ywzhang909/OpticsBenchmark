from __future__ import annotations

from typing import Any

from src.utils.logger import logger

try:
    from src.algorithm.em_eval_utils import normalize_text
    from src.algorithm.hungarian_algorithm_utils import hungarian_match
    from src.algorithm.sentence_similarity_utils import SentenceEmbedder
except ImportError:
    pass

_SENTENCE_EMBEDDER: SentenceEmbedder | None = None


def _get_sentence_embedder(model_name: str = "BAAI/bge-m3") -> SentenceEmbedder:
    global _SENTENCE_EMBEDDER
    if _SENTENCE_EMBEDDER is None:
        _SENTENCE_EMBEDDER = SentenceEmbedder(model_name=model_name)
    return _SENTENCE_EMBEDDER


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
