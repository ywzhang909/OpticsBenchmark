# perplexity_eval_utils.py
"""
Perplexity Evaluation Utilities Module

Computes perplexity of generated text using a causal language model.
Lower perplexity indicates the text is more "expected" by the model,
which correlates with fluency and naturalness.

Uses GPT-2 by default (lightweight, fast). For domain-specific evaluation,
swap to a causal LM fine-tuned on scientific/ optics text.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _validate_input(text: str) -> str | None:
    """Return text stripped, or None if degenerate."""
    if not text or not text.strip():
        return None
    return text.strip()


def compute_perplexity(
    text: str,
    model_name: str = "gpt2",
    max_length: int = 1024,
    stride: int = 512,
    device: str | None = None,
) -> dict[str, Any]:
    """Compute perplexity of ``text`` using a causal LM.

    For long texts, a sliding-window approach is used (``stride`` controls
    the overlap) to avoid OOM.  The result is the exponential of the
    average negative log-likelihood over all tokens.

    Args:
        text: The generated text to evaluate.
        model_name: HuggingFace causal LM name (default ``"gpt2"``).
        max_length: Maximum context length for the model.
        stride: Overlap stride for sliding-window evaluation.
        device: Torch device (``None`` = auto-detect CUDA).

    Returns:
        dict with keys ``perplexity``, ``avg_log_likelihood``, ``num_tokens``,
        ``model_name``.
    """
    text = _validate_input(text)
    if text is None:
        return {
            "perplexity": float("inf"),
            "avg_log_likelihood": float("nan"),
            "num_tokens": 0,
            "model_name": model_name,
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    if seq_len <= max_length:
        # Single pass
        with torch.no_grad():
            outputs = model(
                encodings.input_ids.to(device),
                labels=encodings.input_ids.to(device),
            )
            neg_log_likelihood = outputs.loss.item()
        return {
            "perplexity": round(math.exp(neg_log_likelihood), 4),
            "avg_log_likelihood": round(-neg_log_likelihood, 4),
            "num_tokens": seq_len,
            "model_name": model_name,
        }

    # Sliding window for long sequences
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin:end].to(device)
        if input_ids.size(1) < 2:
            break
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss.item() * (end - begin)
            nlls.append(nll)
        prev_end = end

    total_nll = sum(nlls)
    avg_nll = total_nll / prev_end
    return {
        "perplexity": round(math.exp(avg_nll), 4),
        "avg_log_likelihood": round(-avg_nll, 4),
        "num_tokens": seq_len,
        "model_name": model_name,
    }
