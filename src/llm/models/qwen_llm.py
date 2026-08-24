"""
QwenLLM - 通义千问模型调用类

支持通过 OpenAIProvider 进行 OpenAI 兼容 API 调用。
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.openai_provider import OpenAIProvider
from src.utils import logger

# =============================================================================
# Classes
# =============================================================================


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
        """发送聊天请求。

        根据 provider 类型分发到对应实现，仅支持 OpenAIProvider。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            provider: Provider 实例（须为 OpenAIProvider）
            **kwargs: 额外参数:
                - setup: API 调用参数字典（temperature、max_tokens 等）
                - 其他透传给底层 API 的参数

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: provider 类型不受支持时
        """
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

        # 处理 messages
        processed_messages: list[dict[str, str]] = []
        user_content: list[dict[str, str]] = []
        for message in messages:
            for key, value in message.items():
                if key == "prompt":
                    user_content.append({"type": "text", "text": value})
                elif key == "location":
                    # file_object = await provider.client.files.create(
                    #     file=Path(value),
                    #     purpose="file-extract"
                    # )
                    with open(value, "rb") as f:
                        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
                    user_content.append({
                        "type": "file",
                        "file": {
                            "file_data": f"data:application/pdf;base64,{pdf_base64}",
                            "file_name": Path(value).name,
                    }})
        processed_messages.append({"role": "user", "content": user_content})
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_completion_tokens": setup.get("max_completion_tokens", 4096),
            "logprobs": setup.get("logprobs", False),
            "top_logprobs": setup.get("top_logprobs", 0),
            "parallel_tool_calls": setup.get("parallel_tool_calls", False)
        }

        if setup.get("response_format", False):
            request_kwargs["response_format"] = {"type": "json_object"}

        extra_body: dict[str, Any] = {
            "repetition_penalty": setup.get("repetition_penalty", 1.0),
            "enable_thinking": setup.get("enable_thinking", False),
            "preserve_thinking": setup.get("preserve_thinking", False),
            "enable_search": setup.get("enable_search", False),
            "clear_thinking": setup.get("clear_thinking", False),
        }

        if extra_body["enable_thinking"]:
            extra_body["thinking_budget"] = setup.get("thinking_budget", 4096)

        if setup.get("reasoning_effort") and "thinking_budget" in extra_body:
            raise ValueError("reasoning_effort 与 thinking_budget 不支持同时设置")

        if setup.get("reasoning_effort"):
            extra_body["reasoning_effort"] = setup["reasoning_effort"]

        if setup.get("temperature"):
            request_kwargs["temperature"] = setup["temperature"]

        if setup.get("top_p"):
            request_kwargs["top_p"] = setup["top_p"]

        if setup.get("presence_penalty"):
            request_kwargs["presence_penalty"] = setup["presence_penalty"]

        if setup.get("stop"):
            request_kwargs["stop"] = setup["stop"]

        tools_config = setup.get("tools", {})
        if tools_config:
            tools = []
            if tools_config.get("mcp_server", {}):
                tools.append({
                    "type": "mcp",
                    "server_label": tools_config["mcp_server"].get("server_label", None),
                    "server_description": tools_config["mcp_server"].get(
                        "server_description", None
                    ),
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

        # api_params 覆盖
        request_kwargs.update({"extra_body": extra_body})

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
            logger.error(f"Error in QwenLLM: {e}")
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
        """关闭 Provider 连接。"""
        if isinstance(provider, OpenAIProvider):
            await provider.close()
