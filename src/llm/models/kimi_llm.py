"""
KimiLLM - Kimi (Moonshot AI) 模型调用类

支持通过 OpenAIProvider 进行 OpenAI 兼容 API 调用。
Kimi API 完全兼容 OpenAI Chat Completions API。
base_url: https://api.moonshot.ai/v1

参考文档:
- API Overview: https://platform.kimi.ai/docs/api/overview
- Chat Completions: https://platform.kimi.ai/docs/api/chat
- K3 Quickstart: https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
- Thinking Mode: https://platform.kimi.ai/docs/guide/use-thinking-models
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.openai_provider import OpenAIProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format

# Pricing per million tokens (input, output) for Kimi models
_KIMI_PRICES: dict[str, tuple[float, float]] = {
    "kimi-k3": (0.60, 2.40),
    "kimi-k2.7-code": (0.60, 2.40),
    "kimi-k2.7-code-highspeed": (0.60, 2.40),
    "kimi-k2.6": (0.60, 2.40),
    "kimi-k2.5": (0.60, 2.40),
    "moonshot-v1-8k": (0.12, 0.12),
    "moonshot-v1-32k": (0.12, 0.12),
    "moonshot-v1-128k": (0.12, 0.12),
    "moonshot-v1-auto": (0.12, 0.12),
}

_DEFAULT_PRICE: tuple[float, float] = (0.60, 2.40)


class KimiLLM(BaseLLM):
    """Kimi (Moonshot AI) 模型，支持 OpenAIProvider。"""

    _USE_MAX_TOKENS_HOSTS: set[str] = set()

    def __init__(self, model_name: str = "kimi-k3"):
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
        if isinstance(provider, OpenAIProvider):
            return await self._chat_openai(messages, provider, **kwargs)
        raise ValueError(
            f"KimiLLM does not support provider: {type(provider).__name__}, "
            f"only OpenAIProvider is supported"
        )

    async def _chat_openai(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})

        # Process messages into OpenAI format
        processed_messages = await self._process_messages(messages, provider)

        # Build request parameters
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_completion_tokens": setup.get("max_completion_tokens", 4096),
            "logprobs": setup.get("logprobs", False),
            "top_logprobs": setup.get("top_logprobs", 0),
            "stop": setup.get("stop", None),
            "stream": setup.get("stream", False),
            "stream_options": setup.get("stream_options", False),
            "prompt_cache_key": setup.get("prompt_cache_key", None),
        }

        # Reasoning effort or thinking mode based on model type
        if self.model_name == "kimi-k3":
            # K3: use reasoning_effort ("low", "high", "max"), default "low"
            request_kwargs["reasoning_effort"] = setup.get("reasoning_effort", "low")
        elif self.model_name.startswith("kimi-k2"):
            # K2.x: use thinking config (thinking.type, thinking.keep)
            request_kwargs["thinking"] = setup.get("thinking", None)
        else:
            logger.warning(
                f"Unknown Kimi model: {self.model_name}, "
                f"skipping reasoning_effort/thinking parameters"
            )

        # Response format handling
        # Supports: text, json_object, json_schema
        response_format_config = setup.get("response_format")
        if response_format_config:
            format_type = (
                response_format_config.get("type", "text")
                if isinstance(response_format_config, dict)
                else "json_object"
            )
            if format_type in ("text", "json_object"):
                # Pass response_format directly
                request_kwargs["response_format"] = (
                    response_format_config
                    if isinstance(response_format_config, dict)
                    else {"type": format_type}
                )
            elif format_type == "json_schema":
                # Build json_schema from config or gold_answer_path
                gold_answer_path = kwargs.get("gold_answer_path", None)
                json_schema = response_format_config.get("json_schema", None)
                if not json_schema:
                    schema = self._build_structured_output(gold_answer_path)
                else:
                    schema = json_schema.get("schema", None)
                    if not schema:
                        schema = self._build_structured_output(gold_answer_path)
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema.get("name", "response_schema"),
                        "strict": json_schema.get("strict", True),
                        "schema": schema,
                    },
                }

        # Tools configuration (web_search, file_search, tool_search, mcp_server, custom)
        tools_config = setup.get("tools", {})
        if tools_config:
            tools = self._build_tools(tools_config, provider)
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = setup.get("tool_choice", "auto")

        try:
            response = await provider.client.chat.completions.create(**request_kwargs)
            latency = time.time() - start_time

            choice = response.choices[0]
            content = choice.message.content or ""

            # Extract reasoning_content if present (thinking mode)
            reasoning_content = getattr(choice.message, "reasoning_content", None)

            usage = response.usage.model_dump() if response.usage else {}
            cost = self._calculate_cost(usage)
            self._log_usage(usage, cost, latency)

            result: dict[str, Any] = {
                "content": content,
                "usage": usage,
                "cost": cost,
                "latency": latency,
            }
            if reasoning_content:
                result["thinking"] = reasoning_content
            return result
        except Exception as e:
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def _process_messages(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
    ) -> list[dict[str, Any]]:
        """Convert flat message list into OpenAI-format messages.

        Processes each message key and converts to the appropriate role:
        - prompt/user → user message
        - system/developer → system message
        - assistant → assistant message
        - location → file upload via Files API, content as system message

        Args:
            messages: Flat list of message dicts with role keys.
            provider: OpenAIProvider instance for file operations.

        Returns:
            List of OpenAI-format message dicts.
        """
        processed: list[dict[str, Any]] = []

        for message in messages:
            for key, value in message.items():
                if key == "prompt":
                    processed.append({"role": "user", "content": value})
                elif key in ("system", "developer"):
                    processed.append({"role": "system", "content": value})
                elif key == "assistant":
                    processed.append({"role": "assistant", "content": value})
                elif key == "location":
                    # Upload file via OpenAI-compatible Files API
                    file_object = await provider.client.files.create(
                        file=Path(value),
                        purpose="file-extract",
                    )
                    file_content = await provider.client.files.content(
                        file_id=file_object.id
                    ).text
                    processed.append(
                        {"role": "system", "content": file_content}
                    )

        return processed

    @staticmethod
    def _build_structured_output(gold_answer_path: str | None) -> dict[str, Any] | None:
        """Build JSON Schema from gold_answer_path for structured output.

        Reads the gold answer JSON file and infers the JSON Schema from
        the first data entry, similar to ClaudeLLM._build_structured_output.

        Args:
            gold_answer_path: Path to the gold answer JSON file.

        Returns:
            JSON Schema dict, or None if the file is missing or invalid.
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
        schema = _dict_to_response_format(payload, strict=True)
        return dict(schema)

    @staticmethod
    def _build_tools(
        tools_config: dict[str, Any], provider: OpenAIProvider
    ) -> list[dict[str, Any]]:
        """Build tools list from tools configuration.

        Supports web_search, file_search, tool_search, mcp_server, and custom tools.

        Args:
            tools_config: Tools configuration dictionary.
            provider: OpenAIProvider instance for file operations.

        Returns:
            List of tool definitions.
        """
        tools: list[dict[str, Any]] = []

        # MCP Server
        if tools_config.get("mcp_server", {}):
            mcp = tools_config["mcp_server"]
            tools.append(
                {
                    "type": "mcp",
                    "server_label": mcp.get("server_label"),
                    "server_description": mcp.get("server_description"),
                    "server_url": mcp.get("server_url"),
                    "require_approval": mcp.get("require_approval"),
                }
            )

        # Web search
        if tools_config.get("web_search", False):
            tools.append({"type": "web_search"})

        # File search (requires vector store)
        if tools_config.get("file_search", []):
            vector_store = None
            file_ids: list[str] = []

            # Create vector store
            try:
                import asyncio

                vector_store = asyncio.get_event_loop().run_until_complete(
                    provider.client.vector_stores.create(name="knowledge_base")
                )
            except RuntimeError:
                # If no event loop, skip file_search
                pass

            if vector_store:
                for file_url in tools_config["file_search"]:
                    file_path_obj = Path(file_url)
                    if file_path_obj.exists():
                        try:
                            file_obj = asyncio.get_event_loop().run_until_complete(
                                provider.client.files.create(
                                    file=file_path_obj.open("rb"),
                                    purpose="file-extract",
                                )
                            )
                            file_ids.append(file_obj.id)
                        except Exception:
                            continue

                if file_ids:
                    try:
                        asyncio.get_event_loop().run_until_complete(
                            provider.client.vector_stores.file_batches.create_and_poll(
                                vector_store_id=vector_store.id,
                                file_ids=file_ids,
                            )
                        )
                    except Exception:
                        pass

                tools.append(
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store.id],
                    }
                )

        # Tool search (namespace + tool_search)
        if tools_config.get("tool_search", {}):
            ts = tools_config["tool_search"]
            custom_namespace = {
                "type": "namespace",
                "name": ts.get("name", "unknown"),
                "description": ts.get("description", "unknown"),
                "tools": tools,
            }
            return [custom_namespace, {"type": "tool_search"}]

        # Custom tools
        if tools_config.get("custom", []):
            tools.extend(tools_config["custom"])

        return tools

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        """Calculate cost based on token usage.

        Args:
            usage: Token usage dictionary from API response.

        Returns:
            Cost in USD.
        """
        input_cost_per_m, output_cost_per_m = _KIMI_PRICES.get(
            self.model_name, _DEFAULT_PRICE
        )

        input_tokens = float(usage.get("prompt_tokens", 0) or 0)
        output_tokens = float(usage.get("completion_tokens", 0) or 0)

        return (
            input_tokens / 1_000_000 * input_cost_per_m
            + output_tokens / 1_000_000 * output_cost_per_m
        )

    async def close(self, provider: Any) -> None:
        """Close the provider connection."""
        if isinstance(provider, OpenAIProvider):
            await provider.close()
