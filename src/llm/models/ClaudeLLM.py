"""
ClaudeLLM - Anthropic Claude 模型调用类

支持通过 AnthropicProvider 进行 Anthropic API 调用。
"""

from __future__ import annotations

import time
from typing import Any

from src.llm.base import BaseLLM, build_response_format
from src.llm.providers.AnthropicProvider import AnthropicProvider


class ClaudeLLM(BaseLLM):
    """Anthropic Claude 模型，支持 AnthropicProvider。"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

        # 提取 system 消息，转换消息格式
        system_content = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "assistant":
                if msg.get("content"):
                    api_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "max_tokens": setup.get("max_completion_tokens", 4096),
        }

        if system_content:
            request_kwargs["system"] = system_content

        # Extended thinking
        if setup.get("thinking"):
            request_kwargs["thinking"] = {
                "type": setup["thinking"].get("type", "enabled"),
                "budget_tokens": setup["thinking"].get("budget_tokens", 512),
            }

        # MCP servers
        tools_config = kwargs.get("tools_config", {})
        if tools_config.get("mcp_server"):
            request_kwargs["mcp_servers"] = [tools_config["mcp_server"]]

        # api_params 覆盖
        request_kwargs.update(setup.get("api_params", {}))

        if setup.get("response_format", False):
            rf = build_response_format(kwargs.get("gold_answer_path"))
            if rf:
                js = rf["json_schema"]
                request_kwargs["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": js["schema"],
                    }
                }

        try:
            response = await provider.client.messages.create(**request_kwargs)
            latency = time.time() - start_time

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
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

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self, provider: Any) -> None:
        if isinstance(provider, AnthropicProvider):
            await provider.close()
