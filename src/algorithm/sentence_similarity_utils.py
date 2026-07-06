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

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_tokenizer(model_name: str):
        """Load tokenizer with fallback for transformers 5.x compatibility."""
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except (ValueError, OSError):
            pass
        # Fallback: some older models (e.g. prajjwal1/bert-tiny) don't have
        # config.model_type needed by AutoTokenizer in transformers 5.x.
        from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer

        for cls in [BertTokenizer, AlbertTokenizer, RobertaTokenizer]:
            try:
                return cls.from_pretrained(model_name)
            except Exception:
                continue
        raise ImportError(
            f"Could not load tokenizer for model '{model_name}'. "
            "Try installing tiktoken or sentencepiece, or use a different model."
        )

    @staticmethod
    def _load_model(model_name: str):
        """Load model with fallback for transformers 5.x compatibility."""
        from transformers import AutoModel

        try:
            return AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except (ValueError, OSError):
            pass
        # Fallback: try specific model classes
        from transformers import BertModel, AlbertModel, RobertaModel

        for cls in [BertModel, AlbertModel, RobertaModel]:
            try:
                return cls.from_pretrained(model_name)
            except Exception:
                continue
        raise ImportError(
            f"Could not load model '{model_name}'. "
            "Try a different model name."
        )

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


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sentence Similarity Matrix")
    group_pred = parser.add_mutually_exclusive_group(required=True)
    group_pred.add_argument("--pred-sentences", type=str, nargs="+", help="Predicted sentences")
    group_pred.add_argument("--pred-file", type=str, help="File with one predicted sentence per line")
    group_gold = parser.add_mutually_exclusive_group(required=True)
    group_gold.add_argument("--gold-sentences", type=str, nargs="+", help="Gold/reference sentences")
    group_gold.add_argument("--gold-file", type=str, help="File with one gold sentence per line")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="HuggingFace embedding model (default: BAAI/bge-m3)",
    )
    parser.add_argument("--output", type=str, help="Path to save similarity matrix as JSON")
    args = parser.parse_args()

    if args.pred_file:
        with open(args.pred_file, "r", encoding="utf-8") as f:
            pred_sents = [line.rstrip("\n") for line in f if line.strip()]
    else:
        pred_sents = args.pred_sentences

    if args.gold_file:
        with open(args.gold_file, "r", encoding="utf-8") as f:
            gold_sents = [line.rstrip("\n") for line in f if line.strip()]
    else:
        gold_sents = args.gold_sentences

    try:
        sim_matrix = compute_similarity_matrix(pred_sents, gold_sents, model_name=args.model)
        result = {
            "pred_count": len(pred_sents),
            "gold_count": len(gold_sents),
            "matrix": sim_matrix.tolist(),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        print(text)
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
