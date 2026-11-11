"""
LLM Base - Abstract base class
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.utils import logger


def build_response_format(gold_answer_path: str | None) -> dict[str, Any] | None:
    """Build a response_format schema from gold_answer_path.

    Reads the gold answer JSON file, infers the type of the first record and
    generates a JSON schema.

    Args:
        gold_answer_path: Path to the gold answer JSON file.

    Returns:
        An OpenAI response_format dict, or None on failure.
    """
    if not gold_answer_path:
        return None
    gold_path_obj = Path(gold_answer_path)
    if not gold_path_obj.exists():
        return None
    with open(gold_path_obj, encoding="utf-8") as f:
        gold_data = json.load(f)
    if not isinstance(gold_data, list) or not gold_data:
        return None
    first = gold_data[0]
    payload = first.get("data", first)
    if not isinstance(payload, dict):
        return None

    def _infer_type(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "string"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string"}
        if isinstance(value, list):
            items = {"type": "string"}
            for item in value:
                if isinstance(item, (dict, list)):
                    items = _infer_type(item)
                    break
                t = _infer_type(item)
                if t != items:
                    items = {"type": "string"}
            return {"type": "array", "items": items}
        if isinstance(value, dict):
            props = {}
            required = []
            for k, v in value.items():
                props[k] = _infer_type(v)
                required.append(k)
            obj: dict[str, Any] = {
                "type": "object",
                "properties": props,
                "required": required,
            }
            obj["additionalProperties"] = False
            return obj
        return {"type": "string"}

    schema = _infer_type(payload)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "schema": schema,
            "strict": True,
        },
    }


# =============================================================================
# Classes
# =============================================================================

class BaseLLM(ABC):
    """Abstract base class for LLM models, defining a unified interface."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: Message list [{"role": "user", "content": "..."}].
                OpenAI-compatible models additionally accept the input keys
                "prompt" (text), "location" (path of a text file to inline)
                and "images" (image paths, sent as base64 data URLs for
                vision requests).
            provider: Provider instance
            **kwargs: Extra parameters (temperature, max_tokens, etc.)

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}
        """
        pass

    @staticmethod
    def _log_usage(
        usage: dict[str, Any],
        cost: float,
        latency: float,
    ) -> None:
        """Log token consumption information."""
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens") or input_tokens + output_tokens

        logger.info(
            f"\nToken Usage:\n"
            f"  Input:    {input_tokens:>8,} tokens\n"
            f"  Output:   {output_tokens:>8,} tokens\n"
            f"  Total:    {total_tokens:>8,} tokens\n"
            f"  Cost:     ${cost:.6f}\n"
            f"  Latency:  {latency:.2f}s"
        )

    @abstractmethod
    async def close(self, provider: Any) -> None:
        """Close the Provider connection."""
        pass
