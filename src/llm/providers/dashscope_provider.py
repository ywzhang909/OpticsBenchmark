"""
DashScope Provider - Wrapper around the Alibaba Cloud Bailian DashScope SDK

Provides an async client for the DashScope API, supporting the Qwen model family.
"""

from __future__ import annotations

import dashscope


class DashScopeProvider:
    """DashScope API client wrapper."""

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize the DashScope client.

        Args:
            api_key: DashScope API Key
            base_url: API endpoint URL (optional)
        """
        dashscope.api_key = api_key
        if base_url:
            dashscope.base_http_api_url = base_url

    @property
    def client(self) -> dashscope:
        """Return the dashscope module reference."""
        return dashscope

    async def close(self) -> None:
        """Close the client (no explicit close required)."""
        pass
