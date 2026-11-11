"""
Google Provider - Wrapper around the Google GenAI Client

Supports async calls for the Google Gemini model family (via client.aio).
"""

from __future__ import annotations

from typing import Any

from google import genai

# =============================================================================
# Classes
# =============================================================================


class GoogleProvider:
    """Wraps the Google GenAI Client, implementing async calls via client.aio."""

    def __init__(
        self,
        api_key: str,
        vertexai: bool = False,
        project_id: str | None = None,
        location: str | None = None,
    ):
        """
        Initialize the Google Provider.

        Args:
            api_key: Google API key
            vertexai: Whether to use Vertex AI
            project_id: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud location (required for Vertex AI)
        """
        self._client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project_id=project_id,
            location=location,
        )

    @property
    def client(self):
        """Return the genai.Client instance."""
        return self._client

    @property
    def aio(self) -> Any:
        """Return the async interface client.aio, used by calls such as models.generate_content()."""
        return self._client.aio

    async def close(self) -> None:
        """Close the client (genai.Client requires no explicit close)."""
        pass
