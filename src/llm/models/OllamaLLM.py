"""
OllamaLLM - 本地 Ollama 模型调用类

支持通过 OllamaProvider 进行本地 Ollama HTTP API 调用。
"""

from __future__ import annotations

import time
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.OllamaProvider import OllamaProvider


class OllamaLLM(BaseLLM):
    """本地 Ollama 模型，支持 OllamaProvider。"""

    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(provider, OllamaProvider):
            return await self._chat_ollama(messages, provider, **kwargs)
        raise ValueError(
            f"OllamaLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 OllamaProvider"
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

        try:
            response = await provider.client.post("/api/chat", json=request_body)
            response.raise_for_status()
            data = response.json()
            latency = time.time() - start_time

            content = data.get("message", {}).get("content", "")

            # Ollama 本地模型无 API 费用
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }

            return {
                "content": content,
                "usage": usage,
                "cost": 0.0,
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
        if isinstance(provider, OllamaProvider):
            await provider.close()
