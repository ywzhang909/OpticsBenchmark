"""
OpenAI Provider - Wrapper around the OpenAI AsyncOpenAI SDK

Supports all OpenAI-compatible APIs (OpenAI, DeepSeek, Qwen, Llama, Mistral, etc.).
"""

from __future__ import annotations

from openai import AsyncOpenAI

# =============================================================================
# Classes
# =============================================================================


class OpenAIProvider:
    """Wraps the OpenAI AsyncOpenAI SDK and provides async call support."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """
        Initialize the OpenAI Provider.

        Args:
            api_key: API key
            base_url: API endpoint URL
        """
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def client(self):
        """Return the AsyncOpenAI client instance."""
        return self._client

    async def close(self) -> None:
        """Close the API client connection."""
        await self._client.close()
