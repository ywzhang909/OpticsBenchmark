"""
ClaudeLLM - Anthropic Claude 模型调用类

支持通过 AnthropicProvider 调用 Anthropic Messages API（beta 端点）。
使用 provider.client.beta.messages.create，以支持 mcp_servers、
output_config、speed 等 beta/新特性（参考官方 beta/messages/create 文档）。

setup 采用统一扁平参数：
  - max_tokens / max_completion_tokens → max_tokens
  - response_format: true              → output_config.format (json_schema)
  - thinking                            → 扩展思考（type/budget_tokens/display）
  - tools.mcp_server                    → mcp_servers
  - tools.web_search / web_fetch / tool_search / custom → tools
  - effort / task_budget                → output_config
结构化输出：与 GPTLLM 一致，从 gold_answer_path 推断 JSON Schema。
响应解析：拼接 text 块为 content，thinking 块单独放入返回的 thinking 字段。
"""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format

# 每百万 token 价格（input, output, cache_write, cache_read）
_CLAUDE_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4": (15.0, 75.0, 18.75, 1.50),
    "claude-sonnet-4": (3.0, 15.0, 3.75, 0.30),
    "claude-3-7-sonnet-20250219": (3.0, 15.0, 3.75, 0.30),
    "claude-3-5-sonnet-20241022": (3.0, 15.0, 3.75, 0.30),
    "claude-3-5-haiku-20241022": (0.8, 4.0, 1.0, 0.08),
    "claude-3-opus-20240229": (15.0, 75.0, 18.75, 1.50),
    "claude-3-haiku-20240307": (0.25, 1.25, 0.30, 0.03),
}

_DEFAULT_PRICE: tuple[float, float, float, float] = (3.0, 15.0, 3.75, 0.30)


# =============================================================================
# Classes
# =============================================================================

class ClaudeLLM(BaseLLM):
    """Anthropic Claude 模型，支持 AnthropicProvider（beta Messages API）。"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送聊天请求。

        根据 provider 类型分发到对应实现，仅支持 AnthropicProvider。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            provider: Provider 实例（须为 AnthropicProvider）
            **kwargs: 额外参数:
                - setup: API 调用参数字典（temperature、max_tokens 等）
                - gold_answer_path: gold answer JSON 路径，用于结构化输出
                - 其他透传给底层 API 的参数

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: provider 类型不受支持时
        """
        if isinstance(provider, AnthropicProvider):
            return await self._chat_anthropic(messages, provider, **kwargs)
        raise ValueError(
            f"ClaudeLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 AnthropicProvider"
        )

    async def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        provider: AnthropicProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})
        gold_answer_path = kwargs.get("gold_answer_path", None)

        processed_messages, system_content = await self._process_messages(
            messages, provider, setup
        )

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_tokens": setup.get("max_tokens", 4096),
            "cache_control": setup.get("cache_control", None),
            "context_management": setup.get("context_management", None),
            "inference_geo": setup.get("inference_geo", None),
            "mcp_servers": setup.get("mcp_servers", None),
            "metadata": setup.get("metadata", None),
            "service_tier": setup.get("service_tier", "auto"),
            "speed": setup.get("speed", "fast"),
            "stop_sequences": setup.get("stop_sequences", None),
            "stream": setup.get("stream", False),
            "betas": setup.get("betas", None),
            "temperature": setup.get("temperature", 1.0),
            "top_k": setup.get("top_k", None),
            "top_p": setup.get("top_p", 1.0),
        }

        if system_content:
            if setup.get("cache_control"):
                request_kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                request_kwargs["system"] = system_content

        if setup.get("output_config"):
            request_kwargs["output_config"] = self._build_output_config(setup, gold_answer_path)

        # Anthropic extended thinking
        thinking_config = setup.get("thinking")
        if thinking_config:
            request_kwargs["thinking"] = thinking_config
        else:
            request_kwargs["thinking"] = {"type": "disabled"}

        tools_config = setup.get("tools", {})
        if tools_config:
            built = self._build_tools(tools_config)
            if built.get("tools"):
                request_kwargs["tools"] = built["tools"]
            request_kwargs["tool_choice"] = setup.get("tool_choice", "auto")

        try:
            response = await provider.client.beta.messages.create(**request_kwargs)
            latency = time.time() - start_time

            content = ""
            thinking_text = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "thinking":
                    thinking_text += block.thinking

            usage = response.usage.model_dump() if response.usage else {}
            cost = self._calculate_cost(usage)
            self._log_usage(usage, cost, latency)

            result: dict[str, Any] = {
                "content": content,
                "usage": usage,
                "cost": cost,
                "latency": latency,
            }
            if thinking_text:
                result["thinking"] = thinking_text
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
        provider: AnthropicProvider,
        setup: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        """将扁平消息列表转换为 Claude messages，并提取 system 内容。"""
        system_parts: list[str] = []
        buckets: dict[str, list[dict[str, Any]]] = {"user": [], "assistant": []}

        for message in messages:
            for key, value in message.items():
                if key in ("system", "developer"):
                    system_parts.append(value)
                elif key in ("prompt", "user"):
                    buckets["user"].append({"type": "text", "text": value})
                elif key == "assistant":
                    buckets["assistant"].append({"type": "text", "text": value})
                elif key == "location":
                    buckets["user"].append(await self._build_document(value, provider))
                elif key == "content" and "role" in message:
                    role = "assistant" if message["role"] == "assistant" else "user"
                    buckets[role].append({"type": "text", "text": value})

        processed: list[dict[str, Any]] = []
        for role in ("user", "assistant"):
            blocks = buckets[role]
            if blocks:
                if len(blocks) == 1 and blocks[0]["type"] == "text":
                    content: Any = blocks[0]["text"]
                else:
                    content = blocks
                processed.append({"role": role, "content": content})

        return processed, "\n\n".join(system_parts)

    async def _build_document(self, path: str, provider: AnthropicProvider) -> dict[str, Any]:
        """Upload a local file via the Anthropic Files API and return a document block.

        Reads the file, uploads it using provider.client.beta.files.upload(),
        and returns a Claude document content block referencing the uploaded file_id.
        The document title is derived from the full filename (including extension).

        Args:
            path: Path to the local file to upload.
            provider: AnthropicProvider instance with an initialized client.

        Returns:
            A dict with type="document", title, and source (file_id) fields.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        # Determine MIME type; default to application/pdf for Claude document blocks
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/pdf"

        # Upload the file via the Anthropic Files API
        with open(file_path, "rb") as f:
            file_upload = await provider.client.beta.files.upload(
                file=(file_path.name, f, media_type)
            )

        # Return a document block referencing the uploaded file by ID
        return {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": file_upload.id,
            },
        }

    def _build_output_config(
        self,
        setup: dict[str, Any],
        gold_answer_path: str | None,
    ) -> dict[str, Any] | None:
        """根据 setup 构建 output_config（format/effort/task_budget）。"""
        output_config: dict[str, Any] = {}

        if setup.get("output_config"):
            output_config = setup["output_config"]

        schema = output_config.get("format", None)
        if not schema:
            schema = self._build_structured_output(gold_answer_path)
        else:
            logger.warning(
                "response_format 已启用但无法从 gold_answer_path 生成 JSON Schema，"
                "将忽略结构化输出"
            )
        output_config["format"] = {
            "type": "json_schema",
            "schema": schema,
        }
        return output_config or None

    def _build_structured_output(self, gold_answer_path: str | None) -> dict[str, Any] | None:
        """从 gold_answer_path 构建 JSON Schema（参照 GPTLLM）。"""
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
    def _build_tools(tools_config: dict[str, Any]) -> dict[str, Any]:
        """根据 tools 配置构建 mcp_servers 与 tools 请求参数。"""
        built: dict[str, Any] = {}

        tools: list[dict[str, Any]] = []
        if tools_config.get("web_search", {}):
            ws = tools_config["web_search"]
            tools.append(
                ClaudeLLM._clean(
                    {
                        "type": ws.get("type", "web_search_20250305"),
                        "name": ws.get("name", "web_search"),
                        "max_uses": ws.get("max_uses"),
                        "allowed_domains": ws.get("allowed_domains"),
                        "blocked_domains": ws.get("blocked_domains"),
                        "user_location": ws.get("user_location"),
                    }
                )
            )
        if tools_config.get("web_fetch", {}):
            wf = tools_config["web_fetch"]
            tools.append(
                ClaudeLLM._clean(
                    {
                        "type": wf.get("type", "web_fetch_20250910"),
                        "name": wf.get("name", "web_fetch"),
                        "max_uses": wf.get("max_uses"),
                        "allowed_domains": wf.get("allowed_domains"),
                        "blocked_domains": wf.get("blocked_domains"),
                        "citations": wf.get("citations"),
                        "max_content_tokens": wf.get("max_content_tokens"),
                    }
                )
            )
        if tools_config.get("tool_search", {}):
            ts = tools_config["tool_search"]
            tools.append(
                ClaudeLLM._clean(
                    {
                        "type": ts.get("type", "tool_search_tool_regex_20251119"),
                        "name": ts.get("name", "tool_search"),
                    }
                )
            )
        if tools_config.get("custom", []):
            tools.extend(tools_config["custom"])

        if tools:
            built["tools"] = tools
        return built

    @staticmethod
    def _clean(data: dict[str, Any]) -> dict[str, Any]:
        """剔除值为 None 的字段，避免向 API 发送 null。"""
        return {k: v for k, v in data.items() if v is not None}

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        (
            input_cost_per_m,
            output_cost_per_m,
            cache_write_cost_per_m,
            cache_read_cost_per_m,
        ) = _CLAUDE_PRICES.get(self.model_name, _DEFAULT_PRICE)

        input_tokens = float(usage.get("input_tokens", 0) or 0)
        output_tokens = float(usage.get("output_tokens", 0) or 0)
        cache_read_tokens = float(usage.get("cache_read_input_tokens", 0) or 0)
        cache_write_tokens = float(usage.get("cache_creation_input_tokens", 0) or 0)

        return (
            input_tokens / 1_000_000 * input_cost_per_m
            + output_tokens / 1_000_000 * output_cost_per_m
            + cache_write_tokens / 1_000_000 * cache_write_cost_per_m
            + cache_read_tokens / 1_000_000 * cache_read_cost_per_m
        )

    async def close(self, provider: Any) -> None:
        """关闭 Provider 连接。"""
        if isinstance(provider, AnthropicProvider):
            await provider.close()
