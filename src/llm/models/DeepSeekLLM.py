"""
DeepSeekLLM - DeepSeek 模型调用类

支持通过 OpenAIProvider 进行 OpenAI 兼容 API 调用。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.OpenAIProvider import OpenAIProvider


class DeepSeekLLM(BaseLLM):
    """DeepSeek 模型，支持 OpenAIProvider。"""

    def __init__(self, model_name: str = "deepseek-v4-pro"):
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
            f"DeepSeekLLM 不支持 provider: {type(provider).__name__}，"
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

        # 处理 messages
        processed_messages: list[dict[str, str]] = []
        for message in messages:
            for key, value in message.items():
                if key == "prompt":
                    processed_messages.append({"role": "user", "content": value})
                elif key == "location":
                    file_object = await provider.client.files.create(
                        file=Path(value),
                        purpose="file-extract"
                    )
                    processed_messages.append({"role": "system", "content": f'fileid://{file_object.id}'})

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "thinking": setup.get("thinking", "disabled"),
            "reasoning_effort": setup.get("reasoning_effort", "max"),
            "stream": setup.get("stream", False),
            "temperature": setup.get("temperature", 1.0),
            "max_tokens": setup.get("max_tokens", 4096),
            "top_p": setup.get("top_p", 1.0),
            "logprobs": setup.get("logprobs", False),
            "top_logprobs": setup.get("top_logprobs", 0),
        }

        if setup.get("response_format", False):
            request_kwargs["response_format"] = {"type": "json_object"}

        if setup.get("stop"):
            request_kwargs["stop"] = setup["stop"]

        tools_config = setup.get("tools", {})
        if tools_config:
            tools = []
            if tools_config.get("mcp_server", {}):
                tools.append({
                    "type": "mcp",
                    "server_label": tools_config["mcp_server"].get("server_label", None),
                    "server_description": tools_config["mcp_server"].get("server_description", None),
                    "server_url": tools_config["mcp_server"].get("server_url", None),
                    "require_approval": tools_config["mcp_server"].get("require_approval", None),
                })
            if tools_config.get("web_search", False):
                tools.append({"type": "web_search"})
            if tools_config.get("file_search", []):
                vector_store = await provider.client.vector_stores.create(name="knowledge_base")
                file_ids = []
                for file_url in tools_config["file_search"]:
                    file_path_obj = Path(file_url)
                    if file_path_obj.exists():
                        file_obj = await provider.client.files.create(
                            file=file_path_obj.open("rb"),
                            purpose="file-extract"
                        )
                        file_ids.append(file_obj.id)
                if file_ids:
                    await provider.client.vector_stores.file_batches.create_and_poll(
                        vector_store_id=vector_store.id,
                        file_ids=file_ids
                    )
                tools.append({
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                })
            if tools_config.get("tool_search", {}):
                custom_namespace = {
                    "type": "namespace",
                    "name": tools_config["tool_search"].get("name", "unknown"),
                    "description": tools_config["tool_search"].get("description", "unknown"),
                    "tools": tools,
                }
                request_kwargs["tools"] = [custom_namespace, {"type": "tool_search"}]
            else:
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

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_1k = 0.00014
        output_cost_per_1k = 0.00028

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        if isinstance(provider, OpenAIProvider):
            await provider.close()
