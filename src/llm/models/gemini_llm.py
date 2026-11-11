"""
GeminiLLM - Google Gemini model call class

Calls the Google GenAI Interactions API through GoogleProvider.

setup parameters:
  - max_completion_tokens             → max_output_tokens
  - response_format / gold_answer_path → response_format (JSON Schema)
  - thinking.thinking_level / thinking.thinking_budget
  - tools.web_search / code_execution / function_declarations
  - tools.tool_choice                 → generation_config.tool_choice
  - api_params                        → override or extend generation_config

Message handling: same as GPTLLM, supports flat message dicts where prompt/record is
text, location is a local markdown file path (read and inlined; falls back to file upload
on read failure), and system is the system instruction.
"""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.google_provider import GoogleProvider
from src.utils import logger
from src.utils.general import _dict_to_response_format, read_text_file

# Price per million tokens (input, output)
_GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}

_DEFAULT_PRICE: tuple[float, float] = (1.25, 5.00)


# =============================================================================
# Classes
# =============================================================================

class GeminiLLM(BaseLLM):
    """Google Gemini model, backed by GoogleProvider (Interactions API)."""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request.

        Dispatches to the matching implementation based on the provider type;
        only GoogleProvider is supported.

        Args:
            messages: message list [{"role": "user", "content": "..."}]
            provider: Provider instance (must be a GoogleProvider)
            **kwargs: extra parameters:
                - setup: API call parameter dict (temperature, max_tokens, etc.)
                - gold_answer_path: gold answer JSON path, used for structured output
                - other parameters passed through to the underlying API

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: when the provider type is not supported
        """
        if isinstance(provider, GoogleProvider):
            return await self._chat_google(messages, provider, **kwargs)
        raise ValueError(
            f"GeminiLLM does not support provider: {type(provider).__name__}, "
            f"only GoogleProvider is supported"
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

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": contents,
            "system_instruction": system_instruction if system_instruction else None,
            "tools": setup.get("tools", None),
            "stream": setup.get("stream", False),
            "store": setup.get("store", False),
            "background": setup.get("background", False),
            "generation_config": setup.get("generation_config", None),
            "safety_settings": setup.get("safety_settings", None),
            "service_tier": setup.get("service_tier", None),
            "webhook_config": setup.get("webhook_config", None),
        }

        # Build the structured output response_format
        response_format = setup.get("response_format", None)
        if response_format:
            schema = response_format.get("schema", None)
            if not schema:
                schema = self._build_structured_output(gold_answer_path)

            if schema is None:
                logger.warning(
                    "response_format enabled but cannot generate JSON Schema "
                    "from gold_answer_path, ignoring structured output"
                )
                request_kwargs["response_format"] = None
            else:
                request_kwargs["response_format"] = {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                }
                request_kwargs["response_mime_type"] = "application/json"

        try:
            interaction = await provider.aio.interactions.create(
                **request_kwargs
            )
            latency = time.time() - start_time

            content = interaction.output_text or ""

            usage_meta = interaction.usage
            usage = {
                "prompt_tokens": getattr(usage_meta, "total_input_tokens", 0) or 0,
                "completion_tokens": (
                    getattr(usage_meta, "total_output_tokens", 0) or 0
                ),
                "total_thought_tokens": (
                    getattr(usage_meta, "total_thought_tokens", 0) or 0
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
        """Convert a flat message list into Gemini contents and extract the system instruction."""
        system_parts: list[str] = []
        user_parts: list[Any] = []
        model_parts: list[Any] = []

        for message in messages:
            for key, value in message.items():
                if key in ("system", "input"):
                    system_parts.append(value)
                elif key in ("prompt", "user", "record"):
                    user_parts.append({"type": "text", "text": value})
                elif key == "assistant":
                    model_parts.append({"type": "text", "text": value})
                elif key == "location":
                    text = read_text_file(value)
                    if text is not None:
                        user_parts.append({"type": "text", "text": text})
                    else:
                        # Read failed; fall back to uploading the document
                        logger.warning(
                            f"Failed to read location file, falling back to upload: {value}"
                        )
                        user_parts.append(await self._build_document(value, provider))

        contents: list[dict[str, Any]] = []
        if user_parts:
            contents.append({"role": "user", "parts": user_parts})
        if model_parts:
            contents.append({"role": "model", "parts": model_parts})

        return contents, "\n\n".join(system_parts)

    async def _build_document(self, path: str, provider: GoogleProvider) -> dict[str, Any]:
        """Upload a local file via the Google Files API and return a document block.

        Reads the file, uploads it using provider.client.beta.files.upload(),
        and returns a Claude document content block referencing the uploaded file_id.
        The document title is derived from the full filename (including extension).

        Args:
            path: Path to the local file to upload.
            provider: GoogleProvider instance with an initialized client.

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

        # Upload the file via the Google Files API
        file_upload = await provider.client.beta.files.upload(file=(file_path))

        # Return a document block referencing the uploaded file by ID
        return {
            "type": "document",
            "uri": file_upload.uri,
            "mime_type": media_type,
        }

    def _build_structured_output(
        self, gold_answer_path: str | None
    ) -> dict[str, Any] | None:
        """Build a JSON Schema from gold_answer_path."""
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

    def _calculate_cost(self, usage: dict[str, Any]) -> float:
        input_cost_per_m, output_cost_per_m = _GEMINI_PRICES.get(
            self.model_name, _DEFAULT_PRICE
        )

        input_tokens = float(usage.get("prompt_tokens", 0) or 0)
        output_tokens = float(usage.get("completion_tokens", 0) or 0)
        thought_tokens = float(usage.get("total_thought_tokens", 0) or 0)

        # Thinking tokens are billed at the output price
        return (
            input_tokens / 1_000_000 * input_cost_per_m
            + (output_tokens + thought_tokens) / 1_000_000 * output_cost_per_m
        )

    async def close(self, provider: Any) -> None:
        """Close the provider connection."""
        if isinstance(provider, GoogleProvider):
            await provider.close()
