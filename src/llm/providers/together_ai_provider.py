"""
TogetherAI Provider - 封装 Together AI 官方 Python SDK

通过 together.AsyncTogether 实现异步调用。
"""

from __future__ import annotations

from together import AsyncTogether

# =============================================================================
# Classes
# =============================================================================


class TogetherAIProvider:
    """封装 Together AI 官方 SDK（AsyncTogether），提供异步调用支持。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.together.ai/v1",
        timeout: float = 120.0,
    ):
        """
        初始化 Together AI Provider。

        Args:
            api_key: Together AI API 密钥
            base_url: API 端点地址
            timeout: 请求超时秒数
        """
        self._client = AsyncTogether(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @property
    def client(self) -> AsyncTogether:
        """返回 Together 官方 SDK 异步客户端实例。"""
        return self._client

    async def close(self) -> None:
        """关闭官方 SDK 客户端连接。"""
        await self._client.close()
