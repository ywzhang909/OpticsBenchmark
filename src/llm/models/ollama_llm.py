"""
OllamaLLM - Local Ollama model class

Supports local Ollama HTTP API calls through OllamaProvider.
"""

from __future__ import annotations

import time
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.ollama_provider import OllamaProvider

# =============================================================================
# Classes
# =============================================================================


class OllamaLLM(BaseLLM):
    """Local Ollama model, supports OllamaProvider."""

    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request.

        Dispatch to the implementation matching the provider type; only
        OllamaProvider is supported.

        Args:
            messages: List of messages [{"role": "user", "content": "..."}]
            provider: Provider instance (must be an OllamaProvider)
            **kwargs: Extra parameters:
                - setup: API call parameter dict (temperature, max_tokens, etc.)
                - Other parameters passed through to the underlying API

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: When the provider type is not supported
        """
        if isinstance(provider, OllamaProvider):
            return await self._chat_ollama(messages, provider, **kwargs)
        raise ValueError(
            f"OllamaLLM does not support provider: {type(provider).__name__}, "
            f"only OllamaProvider is supported"
        )

    async def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        provider: OllamaProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": setup.get("temperature", 0.0),
                "num_predict": setup.get("max_completion_tokens", 4096),
            },
        }

        # Local Ollama models do not support response_format structured output
        # if setup.get("response_format", False):
        #     rf = build_response_format(kwargs.get("gold_answer_path"))
        #     if rf:
        #         request_body["format"] = rf["json_schema"]["schema"]

        try:
            response = await provider.client.post("/api/chat", json=request_body)
            response.raise_for_status()
            data = response.json()
            latency = time.time() - start_time

            content = data.get("message", {}).get("content", "")

            # Local Ollama models incur no API cost
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
            cost = 0.0
            self._log_usage(usage, cost, latency)

            return {
                "content": content,
                "usage": usage,
                "cost": cost,
                "latency": latency,
            }
        except Exception as e:
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def close(self, provider: Any) -> None:
        """Close the provider connection."""
        if isinstance(provider, OllamaProvider):
            await provider.close()
