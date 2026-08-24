"""
TogetherAI Provider - 封装 Together AI HTTP API

通过 httpx.AsyncClient 实现异步调用。
"""

from __future__ import annotations

import httpx


class TogetherAIProvider:
    """封装 Together AI httpx.AsyncClient，提供异步调用支持。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.together.xyz",
    ):
        """
        初始化 Together AI Provider。

        Args:
            api_key: Together AI API 密钥
            base_url: API 端点地址
        """
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def client(self):
        """返回 httpx.AsyncClient 实例。"""
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端连接。"""
        await self._client.aclose()
