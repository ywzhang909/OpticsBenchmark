"""
GeminiLLM - Google Gemini 模型调用类

支持通过 GoogleProvider 进行 Google GenAI API 调用。
"""

from __future__ import annotations

import time
from typing import Any

from google.genai import types
from src.llm.base import BaseLLM
from src.llm.providers.GoogleProvider import GoogleProvider


class GeminiLLM(BaseLLM):
    """Google Gemini 模型，支持 GoogleProvider。"""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(provider, GoogleProvider):
            return await self._chat_google(messages, provider, **kwargs)
        raise ValueError(
            f"GeminiLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 GoogleProvider"
        )

    async def _chat_google(
        self,
        messages: list[dict[str, str]],
        provider: GoogleProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:

        start_time = time.time()
        setup = kwargs.get("setup", {})

        # 转换消息为 Gemini 格式
        contents = []
        system_instruction = ""

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        # 构建配置
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=setup.get("temperature"),
            top_p=setup.get("top_p"),
            top_k=setup.get("top_k"),
            max_output_tokens=setup.get("max_completion_tokens"),
            stop_sequences=setup.get("stop_sequences"),
            presence_penalty=setup.get("presence_penalty"),
            frequency_penalty=setup.get("frequency_penalty"),
            seed=setup.get("seed"),
            safety_settings=(
                [types.SafetySetting(**s) for s in setup["safety_settings"]]
                if setup.get("safety_settings")
                else None
            ),
            thinking_config=(
                types.ThinkingConfig(
                    include_thoughts=setup["thinking"].get("include_thoughts"),
                    thinking_budget=setup["thinking"].get("thinking_budget"),
                    thinking_level=setup["thinking"].get("thinking_level"),
                )
                if setup.get("thinking")
                else None
            ),
        )

        try:
            response = await provider.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            latency = time.time() - start_time

            content = response.text or ""

            usage = {
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
            }
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
        input_cost_per_1k = 0.00125
        output_cost_per_1k = 0.005

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        if isinstance(provider, GoogleProvider):
            await provider.close()
