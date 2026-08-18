"""
Mistral Provider - 封装 Mistral AI 官方 SDK

通过 mistralai.client.Mistral 实现异步调用。
"""

from __future__ import annotations

from mistralai.client import Mistral


class MistralProvider:
    """封装 Mistral AI 官方 SDK，提供异步调用支持。"""

    def __init__(self, api_key: str):
        """
        初始化 Mistral Provider。

        Args:
            api_key: Mistral AI API 密钥
        """
        self._client = Mistral(api_key=api_key)

    @property
    def client(self) -> Mistral:
        """返回 Mistral 客户端实例。"""
        return self._client

    async def close(self) -> None:
        """关闭客户端（mistralai SDK 无需显式关闭）。"""
        pass
