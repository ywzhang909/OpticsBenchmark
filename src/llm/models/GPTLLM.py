"""
GPTLLM - OpenAI GPT 模型调用类

支持通过 OpenAIProvider 进行 OpenAI 官方 API 调用。
提供两种调用方式，通过 setup.api_method 选择：
  - chat_completions: Chat Completions API (client.chat.completions.create)
  - responses:        Responses API        (client.responses.create)

setup 采用统一扁平参数，两种调用方式的差异由本类按 api_method 自动映射：
  - max_tokens            → max_completion_tokens / max_output_tokens
  - response_format: true → response_format / text.format (json_object)
  - frequency/presence_penalty 仅 Chat Completions 有效，Responses 忽略
响应结构差异：Chat Completions 返回 choices[].message，usage 为
prompt/completion_tokens；Responses 返回 output_text，usage 为
input/output_tokens。
当选择 Responses API 但模型不兼容时，程序会警告并终止执行。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.OpenAIProvider import OpenAIProvider
from src.utils import logger

# 不支持 Responses API 的 legacy 模型
_RESPONSES_UNSUPPORTED_MODELS: set[str] = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
}

_VALID_API_METHODS: set[str] = {"chat_completions", "responses"}


def _supports_responses(model_name: str) -> bool:
    """判断模型是否支持 Responses API（legacy 模型不支持）。"""
    name = model_name.strip().lower()
    if name in _RESPONSES_UNSUPPORTED_MODELS:
        return False
    if name.startswith("gpt-3.5") or name.startswith("gpt-4-turbo"):
        return False
    return True


class GPTLLM(BaseLLM):
    """OpenAI GPT 模型，支持 OpenAIProvider。"""

    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(provider, OpenAIProvider):
            raise ValueError(
                f"GPTLLM 不支持 provider: {type(provider).__name__}，仅支持 OpenAIProvider"
            )

        setup = kwargs.get("setup", {})
        api_method = setup.get("api_method", "chat_completions")

        if api_method not in _VALID_API_METHODS:
            logger.warning(
                f"不支持的 api_method: '{api_method}'，可选: {sorted(_VALID_API_METHODS)}"
            )
            raise SystemExit(f"不支持的 api_method: '{api_method}'")

        if api_method == "responses" and not _supports_responses(self.model_name):
            logger.warning(
                f"模型 '{self.model_name}' 不支持 Responses API，"
                f"请改用支持的新模型（如 gpt-4o、gpt-5）或切换 "
                f"api_method='chat_completions'"
            )
            raise SystemExit(f"模型 '{self.model_name}' 与 api_method='responses' 不兼容")

        if api_method == "responses":
            return await self._chat_responses(messages, provider, setup)
        return await self._chat_completions(messages, provider, setup)

    async def _chat_completions(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
        setup: dict[str, Any],
    ) -> dict[str, Any]:
        start_time = time.time()

        processed_messages: list[dict[str, str]] = []
        user_content: list[dict[str, str]] = []
        for message in messages:
            for key, value in message.items():
                if key == "prompt":
                    content = {"type": "text", "text": value}
                    if self.prompt_cache_key:
                        content["prompt_cache_breakpoint"] = {"mode":"explicit"}
                    user_content.insert(0, content)
                # elif key == "location":
                #     file_object = await provider.client.files.create(
                #         file=open(value, "rb"),
                #         purpose="user_data"
                #     )
                #     user_content.append(
                #         {"type": "file", "file_id": file_object.id}
                #     )
        processed_messages.append({"role": "user", "content": user_content})

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": setup.get("temperature", 0.0),
            "logit_bias": setup.get("logit_bias", None),
            "max_completion_tokens": setup.get("max_completion_tokens", 4096),
            "metadata": setup.get("metadata", None),
            "modalities": setup.get("modalities", "text"),
            "moderation": setup.get("moderation", None),
            "n": setup.get("n", 1),
            "parallel_tools_calls": setup.get("parallel_tools_calls", True),
            "prompt_cache_key": setup.get("prompt_cache_key", None),
            "prompt_cache_options": setup.get("prompt_cache_options", None),
            "reasoning_effort": setup.get("reasoning_effort", "none"),
            "service_tier": setup.get("service_tier", "auto"),
            "stop": setup.get("stop", None),
            "store": setup.get("store", False),
            "verbosity": setup.get("verbosity", "medium"),
            "web_search_options": setup.get("web_search_options", None),
            "top_p": setup.get("top_p", 1.0),
            "presence_penalty": setup.get("presence_penalty", 0.0),
            "frequency_penalty": setup.get("frequency_penalty", 0.0),
            "logprobs": setup.get("logprobs", False),
            "top_logprobs": setup.get("top_logprobs", 0),
        }

        if setup.get("response_format", False):
            request_kwargs["response_format"] = {"type": "json_object"}

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

    async def _chat_responses(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
        setup: dict[str, Any],
    ) -> dict[str, Any]:
        start_time = time.time()

        processed_messages: list[dict[str, str]] = []
        user_content: list[dict[str, str]] = []
        for message in messages:
            for key, value in message.items():
                if key == "prompt":
                    content = {"type": "text", "text": value}
                    if self.prompt_cache_key:
                        content["prompt_cache_breakpoint"] = {"mode":"explicit"}
                    user_content.insert(0, content)
                # elif key == "location":
                #     file_object = await provider.client.files.create(
                #         file=open(value, "rb"),
                #         purpose="user_data"
                #     )
                #     user_content.append(
                #         {"type": "file", "file_id": file_object.id}
                #     )
        processed_messages.append({"role": "user", "content": user_content})

        # system 消息提取为 instructions
        instructions = ""
        input_messages: list[dict[str, str]] = []
        for message in processed_messages:
            if message["role"] == "system" and not instructions:
                instructions = message["content"]
            else:
                input_messages.append(message)

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": input_messages,
            "temperature": setup.get("temperature", 0.0),
            "max_output_tokens": self._max_tokens(setup, "max_output_tokens"),
            "top_p": setup.get("top_p", 1.0),
            "store": setup.get("store", False),
        }

        if instructions:
            request_kwargs["instructions"] = instructions

        # Responses 不支持 frequency/presence_penalty，此处不传递

        if setup.get("response_format", False):
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        tools_config = setup.get("tools", {})
        # if tools_config:
            # request_kwargs.update(self._build_tools(tools_config, setup))

        try:
            response = await provider.client.responses.create(**request_kwargs)
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
                    "require_approval": tools_config["mcp_server"].get("require_approval", None),
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
        input_cost_per_1k = 0.01
        output_cost_per_1k = 0.03

        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        if isinstance(provider, OpenAIProvider):
            await provider.close()
