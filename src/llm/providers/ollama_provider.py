"""
Ollama Provider - Wrapper around the Ollama HTTP API

Supports async calls for local Ollama models (via httpx.AsyncClient).
"""

from __future__ import annotations

import httpx

# =============================================================================
# Classes
# =============================================================================


class OllamaProvider:
    """Wraps the Ollama HTTP API, implementing async calls via httpx.AsyncClient."""

    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama Provider.

        Args:
            host: Ollama service address
        """
        self._client = httpx.AsyncClient(base_url=host, timeout=120.0)

    @property
    def client(self):
        """Return the httpx.AsyncClient instance."""
        return self._client

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self._client.aclose()
