"""
Ollama Provider - 封装 Ollama HTTP API

支持本地 Ollama 模型的异步调用（通过 httpx.AsyncClient）。
"""

from __future__ import annotations

import httpx


class OllamaProvider:
    """封装 Ollama HTTP API，通过 httpx.AsyncClient 实现异步调用。"""

    def __init__(self, host: str = "http://localhost:11434"):
        """
        初始化 Ollama Provider。

        Args:
            host: Ollama 服务地址
        """
        self._client = httpx.AsyncClient(base_url=host, timeout=120.0)

    @property
    def client(self):
        """返回 httpx.AsyncClient 实例。"""
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端连接。"""
        await self._client.aclose()
