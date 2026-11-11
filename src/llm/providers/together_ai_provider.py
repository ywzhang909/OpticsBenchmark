"""
TogetherAI Provider - Wrapper around the official Together AI Python SDK

Implements async calls via together.AsyncTogether.
"""

from __future__ import annotations

from together import AsyncTogether

# =============================================================================
# Classes
# =============================================================================


class TogetherAIProvider:
    """Wraps the official Together AI SDK (AsyncTogether) and provides async call support."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.together.ai/v1",
        timeout: float = 120.0,
    ):
        """
        Initialize the Together AI Provider.

        Args:
            api_key: Together AI API key
            base_url: API endpoint URL
            timeout: Request timeout in seconds
        """
        self._client = AsyncTogether(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def client(self) -> AsyncTogether:
        """Return the async client instance of the official Together SDK."""
        return self._client

    async def close(self) -> None:
        """Close the official SDK client connection."""
        await self._client.close()
