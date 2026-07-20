"""
QwenLLM - 通义千问模型调用类

支持通过 OpenAIProvider 进行 OpenAI 兼容 API 调用。
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from src.llm.base import BaseLLM
from src.llm.providers.OpenAIProvider import OpenAIProvider


class QwenLLM(BaseLLM):
    """通义千问模型，支持 OpenAIProvider。"""

    # 使用 max_tokens 的 API host
    _USE_MAX_TOKENS_HOSTS: set[str] = {"api.deepseek.com"}

    def __init__(self, model_name: str = "qwen3.5-plus"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(provider, OpenAIProvider):
            return await self._chat_openai(messages, provider, **kwargs)
        raise ValueError(
            f"QwenLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 OpenAIProvider"
        )

    async def _chat_openai(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})

        # 判断 token 参数名

        api_base = kwargs.get("api_base", "")
        host = urlparse(api_base).hostname if api_base else None
        tokens_param = (
            "max_tokens"
            if host and any(host.endswith(h) for h in self._USE_MAX_TOKENS_HOSTS)
            else "max_completion_tokens"
        )

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": setup.get("temperature", 0.0),
            tokens_param: setup.get("max_completion_tokens", 4096),
            "top_p": setup.get("top_p", 1.0),
        }

        # extra_body（如 enable_thinking）
        if setup.get("extra_body"):
            request_kwargs["extra_body"] = setup["extra_body"]

        # api_params 覆盖
        request_kwargs.update(setup.get("api_params", {}))

        try:
            response = await provider.client.chat.completions.create(**request_kwargs)
            latency = time.time() - start_time

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = response.usage.model_dump() if response.usage else {}
            cost = self._calculate_cost(usage)

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

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_1k = 0.0008
        output_cost_per_1k = 0.002

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        if isinstance(provider, OpenAIProvider):
            await provider.close()
