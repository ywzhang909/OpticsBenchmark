"""
LlamaLLM - Llama model class (based on Together AI)

Two invocation modes are supported:
  - _chat_together: via TogetherAIProvider (httpx raw HTTP)
  - _chat_openai:    via OpenAIProvider (OpenAI SDK)

Together AI is compatible with the OpenAI SDK chat.completions.create.
Endpoint: https://api.together.ai/v1

setup uses unified flat parameters (consistent with GPTLLM):
  - max_tokens / max_completion_tokens → max_tokens
  - response_format: true / gold_answer_path → response_format
    (json_schema, structured outputs; falls back to json_object without a schema)
  - reasoning / reasoning_effort          → reasoning switch and strength
  - tools.function_declarations / custom  → function tools
  - tools.tool_choice                     → tool_choice
  - api_params                            → override or extend the request body

Together AI parameter compatibility (from the official docs):
  - temperature, top_p, max_tokens: ✅ fully supported
  - frequency_penalty, presence_penalty: ✅ supported
  - stop, seed, n: ✅ supported (n on some models)
  - response_format (json_object/json_schema): ✅ supported
  - tools, tool_choice: ✅ supported
  - logprobs, top_logprobs: ✅ supported (format differs slightly)
  - reasoning_effort: ⚠️ GPT-OSS models only
  - logit_bias: ❌ unsupported by most models
  - service_tier, store, metadata: ⚠️ accepted but ignored
If an unsupported parameter is provided, the API returns an error and logs it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.providers.together_ai_provider import TogetherAIProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format

# Price per million tokens (input, output), Together AI pricing
_LLAMA_PRICES: dict[str, tuple[float, float]] = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
}

_DEFAULT_PRICE: tuple[float, float] = (0.18, 0.59)


# =============================================================================
# Classes
# =============================================================================

class LlamaLLM(BaseLLM):
    """Llama model, supports TogetherAIProvider and OpenAIProvider."""

    def __init__(self, model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request.

        Dispatch to the implementation matching the provider type; only
        TogetherAIProvider or OpenAIProvider is supported.

        Args:
            messages: List of messages [{"role": "user", "content": "..."}]
            provider: Provider instance (must be a TogetherAIProvider or OpenAIProvider)
            **kwargs: Extra parameters:
                - setup: API call parameter dict (temperature, max_tokens, etc.)
                - gold_answer_path: gold answer JSON path, used for structured output
                - Other parameters passed through to the underlying API

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: When the provider type is not supported
        """
        if isinstance(provider, OpenAIProvider):
            return await self._chat_openai(messages, provider, **kwargs)
        if isinstance(provider, TogetherAIProvider):
            return await self._chat_together(messages, provider, **kwargs)
        raise ValueError(
            f"LlamaLLM does not support provider: {type(provider).__name__}, "
            f"only OpenAIProvider or TogetherAIProvider is supported"
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

        processed_messages = await self._process_messages(messages, provider)

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

        # response_format handling
        if gold_answer_path:
            rf = self._build_structured_output(gold_answer_path)
            if rf:
                request_kwargs["response_format"] = rf
        if setup.get("response_format", False) and "response_format" not in request_kwargs:
            request_kwargs["response_format"] = {"type": "json_object"}

        # tools handling
        tools_config = setup.get("tools", {})
        if tools_config:
            request_kwargs.update(self._build_tools(tools_config, setup))

        # api_params override
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

        processed_messages = await self._process_messages(messages, provider)

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

        # The official SDK uses Omit semantics; drop None values for v2 validation
        body = {k: v for k, v in request_kwargs.items() if v is not None}

        try:
            response = await provider.client.chat.completions.create(**body)

            latency = time.time() - start_time

            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None) or ""

            usage_obj = response.usage
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
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
            logger.error(f"Together API error for model {self.model_name}: {e}")
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def _process_messages(
        self, messages: list[dict[str, str]], provider: Any
    ) -> list[dict[str, Any]]:
        """Convert a flat message list into chat completion message format.

        Handle location files by provider type:
        - OpenAIProvider: upload the file and reference its file_id
        - TogetherAIProvider: read the file text and inline it into the message
        """
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
                    if isinstance(provider, TogetherAIProvider):
                        text = self._read_location_text(value)
                        if text:
                            user_content.append({"type": "text", "text": text})
                    elif isinstance(provider, OpenAIProvider):
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
    def _read_location_text(path: str) -> str:
        """Read a local file's text (used to inline location files for Together)."""
        file_path = Path(path)
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logger.warning(f"Failed to read location file: {path}")
            return ""

    def _build_structured_output(
        self, gold_answer_path: str | None
    ) -> dict[str, Any] | None:
        """Build response_format (json_schema) from gold_answer_path."""
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
        """Build function tools from the tools configuration."""
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
        """Close the provider connection."""
        if isinstance(provider, OpenAIProvider):
            await provider.close()
        elif isinstance(provider, TogetherAIProvider):
            await provider.close()
