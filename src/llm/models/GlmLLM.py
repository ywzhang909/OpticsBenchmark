"""
GlmLLM - 智谱 GLM 模型调用类

支持通过 OpenAIProvider 进行智谱 API 兼容调用。
Endpoint: https://open.bigmodel.cn/api/paas/v4
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM, build_response_format
from src.llm.providers.OpenAIProvider import OpenAIProvider


class GlmLLM(BaseLLM):
    """智谱 GLM 模型，支持 OpenAIProvider。"""

    def __init__(self, model_name: str = "glm-4-plus"):
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
            f"GlmLLM 不支持 provider: {type(provider).__name__}，"
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

        processed_messages = await self._process_messages(messages, provider)

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": setup.get("temperature", 0.7),
            "max_tokens": setup.get("max_tokens", 4096),
            "top_p": setup.get("top_p", 1.0),
        }

        if setup.get("response_format", False):
            rf = build_response_format(kwargs.get("gold_answer_path"))
            if rf:
                request_kwargs["response_format"] = rf
            else:
                request_kwargs["response_format"] = {"type": "json_object"}

        if setup.get("stop"):
            request_kwargs["stop"] = setup["stop"]

        if setup.get("presence_penalty") is not None:
            request_kwargs["presence_penalty"] = setup["presence_penalty"]

        if setup.get("frequency_penalty") is not None:
            request_kwargs["frequency_penalty"] = setup["frequency_penalty"]

        tools_config = setup.get("tools", {})
        if tools_config:
            tools = []
            if tools_config.get("web_search", False):
                tools.append({"type": "web_search"})
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = setup.get("tool_choice", "auto")

        try:
            response = await provider.client.chat.completions.create(**request_kwargs)
            latency = time.time() - start_time

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = response.usage.model_dump() if response.usage else {}
            cost = self._calculate_cost(usage)
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

    _ROLE_KEY_MAP: dict[str, tuple[str, str]] = {
        "prompt": ("user", "input_text"),
        "system": ("system", "input_text"),
        "developer": ("developer", "input_text"),
        "assistant": ("assistant", "input_text"),
    }

    _ROLE_ORDER = ["system", "developer", "user", "assistant"]

    async def _process_messages(
        self, messages: list[dict[str, str]], provider: OpenAIProvider
    ) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, str]]] = {}

        for message in messages:
            for key, value in message.items():
                if key in self._ROLE_KEY_MAP:
                    role, content_type = self._ROLE_KEY_MAP[key]
                    buckets.setdefault(role, []).append(
                        {"type": content_type, "text": value}
                    )
                elif key == "location":
                    file_object = await provider.client.files.create(
                        file=Path(value), purpose="file-extract"
                    )
                    buckets.setdefault("user", []).append(
                        {"type": "input_file", "file_id": file_object.id}
                    )
                elif key == "content" and "role" in message:
                    role = message["role"]
                    buckets.setdefault(role, []).append(
                        {"type": "input_text", "text": value}
                    )

        return [
            {"role": role, "content": content}
            for role in self._ROLE_ORDER
            if (content := buckets.get(role))
        ]

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_1k = 0.00007
        output_cost_per_1k = 0.00007

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        if isinstance(provider, OpenAIProvider):
            await provider.close()
