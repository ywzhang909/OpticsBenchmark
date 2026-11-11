"""
MistralLLM - Mistral model call class

Calls the API through the official Mistral AI SDK (mistralai).
SDK: pip install mistralai
Reference: https://docs.mistral.ai/studio/conversations/chat-completion

setup uses unified flat parameters:
  - max_tokens / max_completion_tokens → max_tokens
  - response_format: true / gold_answer_path → response_format
    (json_schema, structured outputs; falls back to json_object without a schema)
  - random_seed / safe_prompt            → Mistral specific parameters
  - reasoning / reasoning_effort          → reasoning switch and intensity
  - tools.function_declarations / custom  → function tools
  - tools.tool_choice                     → tool_choice
  - api_params                            → override or extend the request body
Message handling: supports flat message dicts, where prompt/record is text and location
is a local markdown file path (read and inlined; falls back to file upload on read
failure); system is the system message.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.mistral_provider import MistralProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format, read_text_file

# Price per million tokens (input, output)
_MISTRAL_PRICES: dict[str, tuple[float, float]] = {
    "mistral-small-latest": (0.10, 0.30),
    "mistral-medium-latest": (1.50, 7.50),
    "mistral-medium-3.5": (1.50, 7.50),
    "mistral-large-latest": (2.00, 6.00),
    "open-mistral-7b": (0.25, 0.25),
    "open-mixtral-8x7b": (0.70, 0.70),
    "open-mixtral-8x22b": (2.00, 6.00),
}

_DEFAULT_PRICE: tuple[float, float] = (1.50, 7.50)


# =============================================================================
# Classes
# =============================================================================

class MistralLLM(BaseLLM):
    """Mistral model, backed by MistralProvider (official SDK)."""

    def __init__(self, model_name: str = "mistral-large-latest"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request.

        Dispatches to the matching implementation based on the provider type;
        only MistralProvider is supported.

        Args:
            messages: message list [{"role": "user", "content": "..."}]
            provider: Provider instance (must be a MistralProvider)
            **kwargs: extra parameters:
                - setup: API call parameter dict (temperature, max_tokens, etc.)
                - gold_answer_path: gold answer JSON path, used for structured output
                - other parameters passed through to the underlying API

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: when the provider type is not supported
        """
        if isinstance(provider, MistralProvider):
            return await self._chat_mistral(messages, provider, **kwargs)
        raise ValueError(
            f"MistralLLM does not support provider: {type(provider).__name__}, "
            f"only MistralProvider (official SDK) is supported"
        )

    async def _chat_mistral(
        self,
        messages: list[dict[str, str]],
        provider: MistralProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})
        gold_answer_path = kwargs.get("gold_answer_path", None)

        processed_messages = await self._process_messages(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "frequency_penalty": setup.get("frequency_penalty", 0.0),
            "max_tokens": setup.get("max_tokens", 4096),
            "metadata": setup.get("metadata", None),
            "n": setup.get("n", 1),
            "parallel_tool_calls": setup.get("parallel_tool_calls", True),
            "prediction": setup.get("prediction", None),
            "presence_penalty": setup.get("presence_penalty", 0.0),
            "prompt_cache_key": setup.get("prompt_cache_key", None),
            "prompt_mode": setup.get("prompt_mode", None),
            "random_seed": setup.get("random_seed", None),
            "reasoning_effort": setup.get("reasoning_effort", None),
            "safe_prompt": setup.get("safe_prompt", False),
            "service_tier": setup.get("service_tier", None),
            "stop": setup.get("stop", None),
            "temperature": setup.get("temperature", 0.0),
            "top_p": setup.get("top_p", 1.0),
        }

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
                        "schema_definition": schema,
                        "description": json_schema.get("description", None),
                    },
                }

        # tools handling
        if setup.get("tools", None):
            request_kwargs["tools"] = setup["tools"]

        try:
            response = await provider.client.chat.complete_async(**request_kwargs)
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
            logger.error(f"Mistral API error for model {self.model_name}: {e}")
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def _process_messages(
        self, messages: list[dict[str, str]], provider: MistralProvider
    ) -> list[dict[str, Any]]:
        """Convert a flat message list into the OpenAI messages format."""
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
                    text = read_text_file(value)
                    if text is not None:
                        user_content.append({"type": "text", "text": text})
                    else:
                        # Read failed; fall back to uploading the file
                        logger.warning(
                            f"Failed to read location file, falling back to upload: {value}"
                        )
                        file_path = Path(value)
                        file_object = await provider.client.files.upload(
                            file={
                                "file_name": file_path.name,
                                "content": open(file_path, "rb")
                            }
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
        """Read local file content (inlined as text, the Mistral API has no file upload)."""
        file_path = Path(path)
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
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

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_m, output_cost_per_m = _MISTRAL_PRICES.get(
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
        if isinstance(provider, MistralProvider):
            await provider.close()
