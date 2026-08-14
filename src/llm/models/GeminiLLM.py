"""
GeminiLLM - Google Gemini 模型调用类

支持通过 GoogleProvider 调用 Google GenAI API（generate_content）。

与 GPTLLM 一致，setup 采用统一扁平参数：
  - max_tokens / max_completion_tokens → max_output_tokens
  - response_format: true / gold_answer_path → response_mime_type +
    response_schema（从 gold_answer_path 推断 JSON Schema，参照 GPTLLM）
  - thinking                            → ThinkingConfig
  - cached_content / prompt_cache_key   → cached_content（缓存内容）
  - tools.web_search / code_execution / file_search / function_declarations
                                        → types.Tool
  - tools.tool_choice                   → types.ToolConfig
  - api_params                          → 覆盖或扩展 GenerateContentConfig
消息处理：与 GPTLLM 相同，支持扁平消息字典，其中 prompt/record 为文本，
location 为本地文件路径（通过 files.upload 上传），system 为系统指令。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from google.genai import types

from src.llm.base import BaseLLM
from src.llm.providers.GoogleProvider import GoogleProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format

# 每百万 token 价格（input, output）
_GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}

_DEFAULT_PRICE: tuple[float, float] = (1.25, 5.00)


class GeminiLLM(BaseLLM):
    """Google Gemini 模型，支持 GoogleProvider。"""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        super().__init__(model_name)
        self._cache_names: dict[str, str] = {}

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
        gold_answer_path = kwargs.get("gold_answer_path", None)

        contents, system_instruction = await self._process_messages(
            messages, provider, setup
        )

        schema = None
        if setup.get("response_format", False) or gold_answer_path:
            schema = self._build_structured_output(gold_answer_path)
            if schema is None and setup.get("response_format", False):
                logger.warning(
                    "response_format 已启用但无法从 gold_answer_path 生成 JSON Schema，"
                    "将忽略结构化输出"
                )

        cached_content = await self._get_cached_content(
            provider, setup, contents, system_instruction
        )

        tools, tool_config = [], None
        if setup.get("tools", {}):
            tools, tool_config = self._build_tools(setup["tools"])

        config = types.GenerateContentConfig(
            **self._build_config(
                setup,
                system_instruction,
                schema,
                cached_content,
                tools,
                tool_config,
            )
        )

        try:
            response = await provider.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            latency = time.time() - start_time

            content = response.text or ""

            usage_meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "completion_tokens": (
                    getattr(usage_meta, "candidates_token_count", 0) or 0
                ),
                "cached_content_token_count": (
                    getattr(usage_meta, "cached_content_token_count", 0) or 0
                ),
            }
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
        self,
        messages: list[dict[str, str]],
        provider: GoogleProvider,
        setup: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        """将扁平消息列表转换为 Gemini contents，并提取 system 指令。"""
        system_parts: list[str] = []
        user_parts: list[Any] = []
        model_parts: list[Any] = []

        for message in messages:
            role = message.get("role", "")
            for key, value in message.items():
                if key in ("role", "input"):
                    continue
                if key in ("system", "developer") or (
                    key == "content" and role in ("system", "developer")
                ):
                    system_parts.append(value)
                elif key in ("prompt", "user", "record"):
                    user_parts.append({"text": value})
                elif key == "assistant":
                    model_parts.append({"text": value})
                elif key == "location":
                    file_obj = await provider.aio.files.upload(file=value)
                    user_parts.append(file_obj)
                elif key == "content":
                    target = model_parts if role == "assistant" else user_parts
                    target.append({"text": value})
                elif isinstance(value, str):
                    user_parts.append({"text": value})

        contents: list[dict[str, Any]] = []
        if user_parts:
            contents.append({"role": "user", "parts": user_parts})
        if model_parts:
            contents.append({"role": "model", "parts": model_parts})

        return contents, "\n\n".join(system_parts)

    async def _get_cached_content(
        self,
        provider: GoogleProvider,
        setup: dict[str, Any],
        contents: list[dict[str, Any]],
        system_instruction: str,
    ) -> str | None:
        """解析 cached_content 名称：直接指定或通过 prompt_cache_key 创建。"""
        cached_name = setup.get("cached_content")
        if cached_name:
            return cached_name

        cache_key = setup.get("prompt_cache_key")
        if not cache_key:
            return None
        if cache_key in self._cache_names:
            return self._cache_names[cache_key]

        if not contents:
            logger.warning("prompt_cache_key 已配置但无可缓存内容，跳过缓存创建")
            return None

        options = setup.get("prompt_cache_options", {}) or {}
        ttl = options.get("ttl", "3600s")
        if not isinstance(ttl, str):
            ttl = f"{int(ttl)}s"

        try:
            cache = await provider.aio.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    contents=contents,
                    system_instruction=system_instruction or None,
                    display_name=options.get(
                        "display_name", f"prompt_cache_{cache_key}"
                    ),
                    ttl=ttl,
                ),
            )
            self._cache_names[cache_key] = cache.name
            return cache.name
        except Exception as e:
            logger.warning(f"创建 Gemini 缓存失败，跳过缓存: {e}")
            return None

    def _build_config(
        self,
        setup: dict[str, Any],
        system_instruction: str,
        schema: dict[str, Any] | None,
        cached_content: str | None,
        tools: list[Any],
        tool_config: Any,
    ) -> dict[str, Any]:
        """构建 GenerateContentConfig 参数（扁平 setup + api_params 覆盖）。"""
        params: dict[str, Any] = {
            "system_instruction": system_instruction or None,
            "temperature": setup.get("temperature"),
            "top_p": setup.get("top_p"),
            "top_k": setup.get("top_k"),
            "candidate_count": setup.get("candidate_count"),
            "max_output_tokens": setup.get("max_completion_tokens"),
            "stop_sequences": setup.get("stop_sequences"),
            "presence_penalty": setup.get("presence_penalty"),
            "frequency_penalty": setup.get("frequency_penalty"),
            "seed": setup.get("seed"),
            "safety_settings": (
                [types.SafetySetting(**s) for s in setup["safety_settings"]]
                if setup.get("safety_settings")
                else None
            ),
            "thinking_config": (
                types.ThinkingConfig(
                    include_thoughts=setup["thinking"].get("include_thoughts"),
                    thinking_budget=(
                        setup["thinking"].get("thinking_budget")
                        or setup["thinking"].get("thinking_budgets")
                    ),
                    thinking_level=setup["thinking"].get("thinking_level"),
                )
                if setup.get("thinking")
                else None
            ),
            "cached_content": cached_content,
            "tools": tools or None,
            "tool_config": tool_config or None,
        }

        if schema:
            params["response_mime_type"] = "application/json"
            params["response_schema"] = schema

        params.update(setup.get("api_params", {}))
        return params

    def _build_structured_output(
        self, gold_answer_path: str | None
    ) -> dict[str, Any] | None:
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
        return _dict_to_response_format(payload, strict=True)

    @staticmethod
    def _build_tools(tools_config: dict[str, Any]) -> tuple[list[Any], Any]:
        """根据 tools 配置构建 Gemini Tool / ToolConfig。"""
        tools: list[Any] = []

        if tools_config.get("web_search", False):
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if tools_config.get("code_execution", False):
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
        if tools_config.get("file_search"):
            tools.append(
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=tools_config["file_search"]
                    )
                )
            )
        if tools_config.get("mcp_server", {}):
            mcp = tools_config["mcp_server"]
            tools.append(
                types.Tool(
                    mcp_servers=[
                        types.McpServer(
                            name=mcp.get("name"),
                            streamable_http_transport=mcp.get(
                                "streamable_http_transport", mcp.get("url")
                            ),
                        )
                    ]
                )
            )
        if tools_config.get("function_declarations", []):
            tools.append(
                types.Tool(function_declarations=tools_config["function_declarations"])
            )
        if tools_config.get("custom", []):
            tools.extend(
                types.Tool(function_declarations=[fn])
                for fn in tools_config["custom"]
            )

        tool_config = None
        choice = tools_config.get("tool_choice", {}) or {}
        if choice:
            mode = choice.get("mode", "AUTO")
            if isinstance(mode, str):
                mode = mode.upper()
            function_calling_config = types.FunctionCallingConfig(mode=mode)
            if choice.get("allowed_function_names"):
                function_calling_config.allowed_function_names = choice[
                    "allowed_function_names"
                ]
            tool_config = types.ToolConfig(
                function_calling_config=function_calling_config
            )

        return tools, tool_config

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_m, output_cost_per_m = _GEMINI_PRICES.get(
            self.model_name, _DEFAULT_PRICE
        )

        input_tokens = float(usage.get("prompt_tokens", 0) or 0)
        output_tokens = float(usage.get("completion_tokens", 0) or 0)
        cached_tokens = float(usage.get("cached_content_token_count", 0) or 0)

        # 缓存命中的 token 已按更低费率计费，不再计入常规输入
        input_tokens = max(input_tokens - cached_tokens, 0)

        return (
            input_tokens / 1_000_000 * input_cost_per_m
            + output_tokens / 1_000_000 * output_cost_per_m
            + cached_tokens / 1_000_000 * input_cost_per_m * 0.25
        )

    async def close(self, provider: Any) -> None:
        if isinstance(provider, GoogleProvider):
            await provider.close()
