"""
Mistral Provider - Wrapper around the official Mistral AI SDK

Implements async calls via mistralai.client.Mistral.
"""

from __future__ import annotations

from mistralai.client import Mistral

# =============================================================================
# Classes
# =============================================================================


class MistralProvider:
    """Wraps the official Mistral AI SDK and provides async call support."""

    def __init__(self, api_key: str):
        """
        Initialize the Mistral Provider.

        Args:
            api_key: Mistral AI API key
        """
        self._client = Mistral(api_key=api_key)

    @property
    def client(self) -> Mistral:
        """Return the Mistral client instance."""
        return self._client

    async def close(self) -> None:
        """Close the client (the mistralai SDK requires no explicit close)."""
        pass
