"""
Groq Provider - 封装 Groq AsyncGroq SDK

支持 Groq 托管模型的异步调用。
"""

from __future__ import annotations

from groq import AsyncGroq


class GroqProvider:
    """封装 Groq AsyncGroq SDK，提供异步调用支持。"""

    def __init__(self, api_key: str):
        """
        初始化 Groq Provider。

        Args:
            api_key: Groq API 密钥
        """
        self._client = AsyncGroq(api_key=api_key)

    @property
    def client(self):
        """返回 AsyncGroq 客户端实例。"""
        return self._client

    async def close(self) -> None:
        """关闭 API 客户端连接。"""
        await self._client.close()
