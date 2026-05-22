# sentence_similarity_utils.py
"""
Sentence Similarity Utilities Module

This module computes pairwise similarity matrices between two lists of
sentences using transformer-based embedding models. The default model is
BAAI/bge-m3, and it can be easily swapped to any other model by changing
the ``model_name`` parameter.
"""

from __future__ import annotations

import numpy as np


def _mean_pooling(embeddings, attention_mask):
    """Mean pooling on token-level embeddings weighted by attention mask."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class SentenceEmbedder:
    """Transformer-based sentence embedder.

    Encodes sentences into dense vector representations. The underlying
    model can be swapped simply by passing a different ``model_name``.

    Args:
        model_name: HuggingFace model name (default: ``"BAAI/bge-m3"``).
        device: Device to run inference on (``"cpu"``, ``"cuda"``, etc.).
            If ``None``, auto-detects CUDA.
        batch_size: Number of sentences to encode per batch.
        max_length: Maximum token length for truncation.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences into L2-normalized embeddings.

        Args:
            sentences: List of text strings to encode.

        Returns:
            Float32 array of shape ``(len(sentences), hidden_dim)`` with
            L2-normalized embeddings.
        """
        import torch

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            emb = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def compute_similarity_matrix(
    pred_sentences: list[str],
    gold_sentences: list[str],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix between two sentence lists.

    Args:
        pred_sentences: List of predicted/student sentences.
        gold_sentences: List of gold/reference sentences.
        model_name: HuggingFace model name for embeddings.
            Change this to any compatible model (e.g. ``"all-MiniLM-L6-v2"``
            via sentence-transformers format, ``"BAAI/bge-large-en-v1.5"``,
            etc.) to swap the embedding backbone.
        batch_size: Encoding batch size.
        device: Inference device (``None`` = auto-detect).

    Returns:
        Float32 array of shape ``(n, m)`` where element ``(i, j)`` is the
        cosine similarity between ``pred_sentences[i]`` and
        ``gold_sentences[j]``.
    """
    embedder = SentenceEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    pred_embs = embedder.encode(pred_sentences)
    gold_embs = embedder.encode(gold_sentences)

    return np.dot(pred_embs, gold_embs.T).astype(np.float32)
