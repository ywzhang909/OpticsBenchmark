"""
LlamaLLM - Llama 模型调用类（基于 Together AI）

支持两种调用方式：
  - _chat_together: 通过 TogetherAIProvider（httpx raw HTTP）
  - _chat_openai:    通过 OpenAIProvider（OpenAI SDK）

Together AI 兼容 OpenAI SDK 的 chat.completions.create。
Endpoint: https://api.together.ai/v1

setup 采用统一扁平参数（与 GPTLLM 一致）：
  - max_tokens / max_completion_tokens → max_tokens
  - response_format: true / gold_answer_path → response_format
    （json_schema，structured outputs；无 schema 时退回 json_object）
  - reasoning / reasoning_effort          → 推理开关与强度
  - tools.function_declarations / custom  → function tools
  - tools.tool_choice                     → tool_choice
  - api_params                            → 覆盖或扩展请求体

Together AI 参数兼容性（来自官方文档）：
  - temperature, top_p, max_tokens: ✅ 完全支持
  - frequency_penalty, presence_penalty: ✅ 支持
  - stop, seed, n: ✅ 支持（n 部分模型）
  - response_format (json_object/json_schema): ✅ 支持
  - tools, tool_choice: ✅ 支持
  - logprobs, top_logprobs: ✅ 支持（格式略有不同）
  - reasoning_effort: ⚠️ 仅 GPT-OSS 模型
  - logit_bias: ❌ 大部分模型不支持
  - service_tier, store, metadata: ⚠️ 接受但忽略
若输入不支持的参数，API 将返回错误并记录日志。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.OpenAIProvider import OpenAIProvider
from src.llm.providers.TogetherAIProvider import TogetherAIProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format

# 每百万 token 价格（input, output），Together AI 计价
_LLAMA_PRICES: dict[str, tuple[float, float]] = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
}

_DEFAULT_PRICE: tuple[float, float] = (0.18, 0.59)


class LlamaLLM(BaseLLM):
    """Llama 模型，支持 TogetherAIProvider 和 OpenAIProvider。"""

    def __init__(self, model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(provider, OpenAIProvider):
            return await self._chat_openai(messages, provider, **kwargs)
        if isinstance(provider, TogetherAIProvider):
            return await self._chat_together(messages, provider, **kwargs)
        raise ValueError(
            f"LlamaLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 OpenAIProvider 或 TogetherAIProvider"
        )

    async def _chat_openai(
        self,
        messages: list[dict[str, str]],
        provider: OpenAIProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})
        gold_answer_path = kwargs.get("gold_answer_path", None)

        processed_messages = await self._process_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": setup.get("temperature", 0.0),
            "top_p": setup.get("top_p", 1.0),
            "max_tokens": setup.get("max_tokens")
            or setup.get("max_completion_tokens", 4096),
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
            "seed": setup.get("seed", None),
        }

        # response_format 处理
        if gold_answer_path:
            rf = self._build_structured_output(gold_answer_path)
            if rf:
                request_kwargs["response_format"] = rf
        if setup.get("response_format", False) and "response_format" not in request_kwargs:
            request_kwargs["response_format"] = {"type": "json_object"}

        # tools 处理
        tools_config = setup.get("tools", {})
        if tools_config:
            request_kwargs.update(self._build_tools(tools_config, setup))

        # api_params 覆盖
        request_kwargs.update(setup.get("api_params", {}))

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
            logger.error(f"OpenAI API error for model {self.model_name}: {e}")
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def _chat_together(
        self,
        messages: list[dict[str, str]],
        provider: TogetherAIProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})
        gold_answer_path = kwargs.get("gold_answer_path", None)

        processed_messages = await self._process_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_tokens": setup.get("max_tokens")
            or setup.get("max_completion_tokens", 4096),
            "temperature": setup.get("temperature", 0.0),
            "top_p": setup.get("top_p", 1.0),
            "top_k": setup.get("top_k"),
            "stop": setup.get("stop", None),
            "seed": setup.get("seed"),
            "n": setup.get("n", 1),
            "min_p": setup.get("min_p"),
            "repetition_penalty": setup.get("repetition_penalty"),
            "presence_penalty": setup.get("presence_penalty", 0.0),
            "frequency_penalty": setup.get("frequency_penalty", 0.0),
            "logit_bias": setup.get("logit_bias"),
            "context_length_exceeded_behavior": setup.get(
                "context_length_exceeded_behavior"
            ),
            "stream": setup.get("stream", False),
            "logprobs": setup.get("logprobs"),
            "echo": setup.get("echo", False),
            "safety_model": setup.get("safety_model"),
            "chat_template_kwargs": setup.get("chat_template_kwargs"),
            "reasoning": setup.get("reasoning"),
            "reasoning_effort": setup.get("reasoning_effort"),
        }

        tools_config = setup.get("tools", {})
        if tools_config:
            built_tools = self._build_tools(tools_config)
            if built_tools.get("tools"):
                request_kwargs["tools"] = built_tools["tools"]
            if built_tools.get("tool_choice") is not None:
                request_kwargs["tool_choice"] = built_tools["tool_choice"]

        response_format = self._build_structured_output(gold_answer_path)
        if response_format:
            request_kwargs["response_format"] = response_format
        elif setup.get("response_format", False):
            logger.warning(
                "response_format enabled but cannot generate JSON Schema "
                "from gold_answer_path, falling back to json_object"
            )
            request_kwargs["response_format"] = {"type": "json_object"}

        request_kwargs.update(setup.get("api_params", {}))

        try:
            response = await provider.client.post(
                "/v1/chat/completions", json=request_kwargs
            )
            response.raise_for_status()
            data = response.json()

            latency = time.time() - start_time

            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""

            usage = data.get("usage", {})
            cost = self._calculate_cost(usage)
            self._log_usage(usage, cost, latency)

            return {
                "content": content,
                "usage": usage,
                "cost": cost,
                "latency": latency,
            }
        except Exception as e:
            logger.error(f"Together API error for model {self.model_name}: {e}")
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

    @staticmethod
    def _read_file(path: str) -> str:
        """读取本地文件内容（文本内联，Together AI 无文件上传 API）。"""
        file_path = Path(path)
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _build_structured_output(
        self, gold_answer_path: str | None
    ) -> dict[str, Any] | None:
        """从 gold_answer_path 构建 response_format（json_schema）。"""
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
            "type": "json_schema",
            "json_schema": {
                "name": "response_schema",
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _build_tools(tools_config: dict[str, Any]) -> dict[str, Any]:
        """根据 tools 配置构建 function tools。"""
        tools: list[dict[str, Any]] = []
        declarations = tools_config.get("function_declarations", []) or []
        declarations += tools_config.get("custom", [])

        for fn in declarations:
            function: dict[str, Any] = {
                "name": fn.get("name", "unknown"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
            tools.append(
                {
                    "type": "function",
                    "function": {k: v for k, v in function.items() if v is not None},
                }
            )

        built: dict[str, Any] = {}
        if tools:
            built["tools"] = tools
        if tools_config.get("tool_choice") is not None:
            built["tool_choice"] = tools_config["tool_choice"]
        return built

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_m, output_cost_per_m = _LLAMA_PRICES.get(
            self.model_name, _DEFAULT_PRICE
        )

        input_tokens = float(usage.get("prompt_tokens", 0) or 0)
        output_tokens = float(usage.get("completion_tokens", 0) or 0)

        return (
            input_tokens / 1_000_000 * input_cost_per_m
            + output_tokens / 1_000_000 * output_cost_per_m
        )

    async def close(self, provider: Any) -> None:
        if isinstance(provider, OpenAIProvider):
            await provider.close()
        elif isinstance(provider, TogetherAIProvider):
            await provider.close()
