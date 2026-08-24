"""
GlmLLM - 智谱 GLM 模型调用类

支持通过 OpenAIProvider 进行智谱 API 兼容调用（OpenAI SDK）。
Endpoint: https://open.bigmodel.cn/api/paas/v4

setup 参数与 OpenAI Chat Completions 兼容：
  - max_tokens             → max_tokens
  - response_format: true  → response_format (json_object)
  - gold_answer_path       → response_format (json_schema via _build_structured_output)
  - tools / tool_choice    → tools / tool_choice
  - frequency/presence_penalty, logit_bias, metadata, n, store 等均透传
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.openai_provider import OpenAIProvider
from src.utils.general import _dict_to_response_format

# =============================================================================
# Classes
# =============================================================================


class GlmLLM(BaseLLM):
    """智谱 GLM 模型，支持 OpenAIProvider（OpenAI SDK 兼容）。"""

    def __init__(self, model_name: str = "glm-4.5-air"):
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
                - gold_answer_path: gold answer JSON 路径，用于结构化输出
                - 其他透传给底层 API 的参数

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: provider 类型不受支持时
        """
        if not isinstance(provider, OpenAIProvider):
            raise ValueError(
                f"GlmLLM 不支持 provider: {type(provider).__name__}，"
                f"仅支持 OpenAIProvider"
            )

        setup = kwargs.get("setup", {})
        gold_answer_path = kwargs.get("gold_answer_path", None)
        if gold_answer_path:
            text_format = self._build_structured_output(gold_answer_path)
            if text_format:
                setup["text_format"] = text_format

        return await self._chat_openai(messages, provider, setup)

    async def _chat_openai(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
        setup: dict[str, Any],
    ) -> dict[str, Any]:
        start_time = time.time()

        processed_messages = await self._process_messages(messages, provider)

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": setup.get("temperature", 0.0),
            "top_p": setup.get("top_p", 1.0),
            "max_tokens": setup.get("max_tokens", 4096),
            "frequency_penalty": setup.get("frequency_penalty", 0.0),
            "presence_penalty": setup.get("presence_penalty", 0.0),
            "logit_bias": setup.get("logit_bias", None),
            "metadata": setup.get("metadata", None),
            "n": setup.get("n", 1),
            "parallel_tool_calls": setup.get("parallel_tool_calls", True),
            "reasoning_effort": setup.get("reasoning_effort", None),
            "service_tier": setup.get("service_tier", "auto"),
            "stop": setup.get("stop", None),
            "store": setup.get("store", False),
            "web_search_options": setup.get("web_search_options", None),
            "logprobs": setup.get("logprobs", False),
            "top_logprobs": setup.get("top_logprobs", 0),
        }

        if setup.get("response_format", False):
            request_kwargs["response_format"] = {"type": "json_object"}

        if setup.get("text_format", False):
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": setup["text_format"]["format"]["schema"],
            }

        tools_config = setup.get("tools", {})
        if tools_config:
            request_kwargs.update(self._build_tools(tools_config, setup))

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

    async def _process_messages(
        self, messages: list[dict[str, str]], provider: OpenAIProvider
    ) -> list[dict[str, Any]]:
        """将扁平消息列表转换为 OpenAI messages 格式。"""
        processed_messages: list[dict[str, str]] = []
        user_content: list[dict[str, str]] = []
        system_content: list[dict[str, str]] = []

        for message in messages:
            for key, value in message.items():
                if key == "system":
                    system_content.append({"type": "text", "text": value})
                elif key == "prompt":
                    user_content.append({"type": "text", "text": value})
                elif key == "location":
                    file_object = await provider.client.files.create(
                        file=open(value, "rb"), purpose="user_data"
                    )
                    user_content.append(
                        {"type": "file", "file_id": file_object.id}
                    )
                elif key == "content" and "role" in message:
                    role = message["role"]
                    if role == "system":
                        system_content.append({"type": "text", "text": value})
                    else:
                        user_content.append({"type": "text", "text": value})

        if system_content:
            processed_messages.append({"role": "system", "content": system_content})
        if user_content:
            processed_messages.append({"role": "user", "content": user_content})

        return processed_messages

    def _build_structured_output(self, gold_answer_path: str) -> dict[str, Any] | None:
        """从 gold_answer_path 构建 response_format schema。"""
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

        schema = _dict_to_response_format(payload, strict=True)

        return {
            "format": {
                "type": "json_schema",
                "strict": True,
                "schema": schema,
            }
        }

    def _build_tools(
        self,
        tools_config: dict[str, Any],
        method_setup: dict[str, Any],
    ) -> dict[str, Any]:
        """根据 tools 配置构建 tools 请求参数。"""
        tools = []
        if tools_config.get("mcp_server", {}):
            tools.append(
                {
                    "type": "mcp",
                    "server_label": tools_config["mcp_server"].get("server_label", None),
                    "server_description": tools_config["mcp_server"].get(
                        "server_description", None
                    ),
                    "server_url": tools_config["mcp_server"].get("server_url", None),
                    "require_approval": tools_config["mcp_server"].get(
                        "require_approval", None
                    ),
                }
            )
        if tools_config.get("web_search", False):
            tools.append({"type": "web_search"})
        if tools_config.get("file_search", []):
            tools.append({"type": "file_search", "vector_store_ids": []})
        if tools_config.get("tool_search", {}):
            custom_namespace = {
                "type": "namespace",
                "name": tools_config["tool_search"].get("name", "unknown"),
                "description": tools_config["tool_search"].get("description", "unknown"),
                "tools": tools,
            }
            request: dict[str, Any] = {
                "tools": [custom_namespace, {"type": "tool_search"}],
            }
        else:
            request = {"tools": tools}
        request["tool_choice"] = method_setup.get("tool_choice", "auto")
        return request

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_1k = 0.00007
        output_cost_per_1k = 0.00007

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        """关闭 Provider 连接。"""
        if isinstance(provider, OpenAIProvider):
            await provider.close()
