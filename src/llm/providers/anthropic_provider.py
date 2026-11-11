"""
Anthropic Provider - Wrapper around the Anthropic AsyncAnthropic SDK

Supports async calls for the Anthropic Claude model family.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

# =============================================================================
# Classes
# =============================================================================


class AnthropicProvider:
    """Wraps the Anthropic AsyncAnthropic SDK and provides async call support."""

    def __init__(self, api_key: str):
        """
        Initialize the Anthropic Provider.

        Args:
            api_key: Anthropic API key
        """
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def client(self):
        """Return the AsyncAnthropic client instance."""
        return self._client

    async def close(self) -> None:
        """Close the API client connection."""
        await self._client.close()
