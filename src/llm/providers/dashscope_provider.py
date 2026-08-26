"""
DashScope Provider - 封装阿里云百炼 DashScope SDK

提供 DashScope API 的异步客户端，支持 Qwen 系列模型。
"""

from __future__ import annotations

import dashscope


class DashScopeProvider:
    """DashScope API 客户端包装。"""

    def __init__(self, api_key: str, base_url: str | None = None):
        """初始化 DashScope 客户端。

        Args:
            api_key: DashScope API Key
            base_url: API 端点地址（可选）
        """
        dashscope.api_key = api_key
        if base_url:
            dashscope.base_http_api_url = base_url

    @property
    def client(self) -> dashscope:
        """返回 dashscope 模块引用。"""
        return dashscope

    async def close(self) -> None:
        """关闭客户端（无需显式关闭）。"""
        pass
