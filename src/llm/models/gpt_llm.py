"""
GPTLLM - OpenAI GPT model call class

Calls the official OpenAI API through OpenAIProvider.
Two call modes are available, selected via setup.api_method:
  - chat_completions: Chat Completions API (client.chat.completions.create)
  - responses:        Responses API        (client.responses.create)

setup uses unified flat parameters; this class maps the differences between
the two modes automatically based on api_method:
  - max_tokens            → max_completion_tokens / max_output_tokens
  - response_format: true → response_format / text.format (json_object)
  - frequency/presence_penalty apply to Chat Completions only; Responses ignores them
Response structure differences: Chat Completions returns choices[].message with
usage as prompt/completion_tokens; Responses returns output_text with usage as
input/output_tokens.
When the Responses API is selected but the model is incompatible, the program
warns and terminates.
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

# Legacy models that do not support the Responses API
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
    """Check whether the model supports the Responses API (legacy ones do not)."""
    name = model_name.strip().lower()
    if name in _RESPONSES_UNSUPPORTED_MODELS:
        return False
    if name.startswith("gpt-3.5") or name.startswith("gpt-4-turbo"):
        return False
    return True


# =============================================================================
# Classes
# =============================================================================

class GPTLLM(BaseLLM):
    """OpenAI GPT model, supports OpenAIProvider."""

    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request.

        Dispatch to the matching implementation based on the provider type;
        only OpenAIProvider is supported.

        Args:
            messages: Message list [{"role": "user", "content": "..."}]
            provider: Provider instance (must be an OpenAIProvider)
            **kwargs: Extra parameters:
                - setup: API call parameter dict (temperature, max_tokens, etc.)
                - gold_answer_path: Path to the gold answer JSON, for structured output
                - Other parameters passed through to the underlying API

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: If the provider type is not supported
        """
        if not isinstance(provider, OpenAIProvider):
            raise ValueError(
                f"GPTLLM does not support provider: {type(provider).__name__}, "
                f"only OpenAIProvider is supported"
            )

        setup = kwargs.get("setup", {})
        gold_answer_path = kwargs.get("gold_answer_path", None)
        if gold_answer_path:
            response_format = self._build_structured_output(gold_answer_path)
            if response_format:
                setup["text_format"] = response_format

        api_method = setup.get("api_method", "chat_completions")

        if api_method not in _VALID_API_METHODS:
            logger.warning(
                f"Unsupported api_method: '{api_method}', "
                f"supported options: {sorted(_VALID_API_METHODS)}"
            )
            raise SystemExit(f"Unsupported api_method: '{api_method}'")

        if api_method == "responses" and not _supports_responses(self.model_name):
            logger.warning(
                f"Model '{self.model_name}' does not support the Responses API; "
                f"use a supported newer model (e.g. gpt-4o, gpt-5) or switch to "
                f"api_method='chat_completions'"
            )
            raise SystemExit(
                f"Model '{self.model_name}' is incompatible with api_method='responses'"
            )

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
                        content["prompt_cache_breakpoint"] = {"mode": "explicit"}
                    user_content.insert(0, content)
                elif key == "location":
                    file_object = await provider.client.files.create(
                        file=open(value, "rb"),
                        purpose="user_data"
                    )
                    user_content.append(
                        {"type": "file", "file_id": file_object.id}
                    )
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
        instructions = ""
        for message in messages:
            for key, value in message.items():
                if key == "prompt":
                    content = {"type": "input_text", "text": value}
                    if self.prompt_cache_key:
                        content["prompt_cache_breakpoint"] = {"mode": "explicit"}
                    user_content.insert(0, content)
                elif key == "location":
                    file_object = await provider.client.files.create(
                        file=open(value, "rb"),
                        purpose="user_data"
                    )
                    user_content.append(
                        {
                            "type": "input_file",
                            "file_id": file_object.id,
                            "detail": "auto",
                        }
                    )
                elif key == "system":
                    instructions = value
        processed_messages.append({"role": "user", "content": user_content})

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": processed_messages,
            "instructions": instructions if instructions != "" else None,
            "background": setup.get("background", False),
            "context_management": setup.get("context_management", None),
            "include": setup.get("include", None),
            "max_tool_calls": setup.get("max_tool_calls", None),
            "metadata": setup.get("metadata", None),
            "moderation": setup.get("moderation", None),
            "parallel_tool_calls": setup.get("parallel_tool_calls", True),
            "prompt_cache_key": setup.get("prompt_cache_key", None),
            "prompt_cache_options": setup.get("prompt_cache_options", None),
            "reasoning": setup.get("reasoning_effort", "none"),
            "service_tier": setup.get("service_tier", "auto"),
            "store": setup.get("store", False),
            "stream": setup.get("stream", False),
            "stream_options": setup.get("stream_options", None),
            "temperature": setup.get("temperature", 0.0),
            "max_output_tokens": setup.get("max_output_tokens", 4096),
            "top_logprobs": setup.get("top_logprobs", 0),
            "top_p": setup.get("top_p", 1.0),
        }

        if setup.get("text_format", False):
            request_kwargs["text"] = {
                setup.get("text_format", None)
            }

        if instructions:
            request_kwargs["instructions"] = instructions

        # Responses does not support frequency/presence_penalty, so they are not passed

        if setup.get("response_format", False):
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        tools_config = setup.get("tools", {})
        if tools_config:
            request_kwargs.update(self._build_tools(tools_config, setup))

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

    def _build_structured_output(self, gold_answer_path: str) -> dict[str, Any] | None:
        """Build response_format from gold_answer_path."""
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

        schema = _dict_to_response_format(
            payload, strict=True
        )

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
        """Build the tools request parameters from the tools config."""
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
        """Close the provider connection."""
        if isinstance(provider, OpenAIProvider):
            await provider.close()
