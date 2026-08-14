"""
LlamaLLM - Llama 模型调用类（基于 Together AI）

支持通过 TogetherAIProvider 调用 Together AI 的 OpenAI 兼容 API
（POST /v1/chat/completions，参考官方 Chat Completions 文档）。

setup 采用统一扁平参数（与 GPTLLM 一致）：
  - max_tokens / max_completion_tokens → max_tokens
  - response_format: true / gold_answer_path → response_format
    （json_schema，structured outputs；无 schema 时退回 json_object）
  - reasoning / reasoning_effort          → 推理开关与强度
  - tools.function_declarations / custom  → function tools
  - tools.tool_choice                     → tool_choice
  - api_params                            → 覆盖或扩展请求体
消息处理：与 GPTLLM 相同，支持扁平消息字典，其中 prompt/record 为文本，
location 为本地文件路径（读取文本内联，Together AI 无文件上传 API），
system 为系统消息。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.TogetherAIProvider import TogetherAIProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format

# 每百万 token 价格（input, output），Together AI 计价
_LLAMA_PRICES: dict[str, tuple[float, float]] = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
}

_DEFAULT_PRICE: tuple[float, float] = (0.18, 0.59)


class LlamaLLM(BaseLLM):
    """Llama 模型，支持 TogetherAIProvider。"""

    def __init__(self, model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(provider, TogetherAIProvider):
            return await self._chat_together(messages, provider, **kwargs)
        raise ValueError(
            f"LlamaLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 TogetherAIProvider"
        )

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
                "response_format 已启用但无法从 gold_answer_path 生成 JSON Schema，"
                "退回 json_object"
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
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def _process_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """将扁平消息列表转换为 OpenAI 兼容的 messages 列表。"""
        processed: list[dict[str, str]] = []

        for message in messages:
            role = message.get("role", "")
            for key, value in message.items():
                if key in ("role", "input"):
                    continue
                if key in ("system", "developer") or (
                    key == "content" and role in ("system", "developer")
                ):
                    processed.append({"role": "system", "content": value})
                elif key in ("prompt", "user", "record"):
                    processed.append({"role": "user", "content": value})
                elif key == "assistant":
                    processed.append({"role": "assistant", "content": value})
                elif key == "location":
                    processed.append({"role": "user", "content": self._read_file(value)})
                elif key == "content":
                    target_role = "assistant" if role == "assistant" else "user"
                    processed.append({"role": target_role, "content": value})
                elif isinstance(value, str):
                    processed.append({"role": "user", "content": value})

        return processed

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
        """从 gold_answer_path 构建 Together response_format（json_schema）。"""
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
        """根据 tools 配置构建 function tools（Together 仅支持函数工具）。"""
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
        if isinstance(provider, TogetherAIProvider):
            await provider.close()
