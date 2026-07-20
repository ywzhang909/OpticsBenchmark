"""
LLM Base - 抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """LLM 模型抽象基类，定义统一接口。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        发送聊天请求。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            provider: Provider 实例
            **kwargs: 额外参数 (temperature, max_tokens 等)

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}
        """
        pass

    @abstractmethod
    async def close(self, provider: Any) -> None:
        """关闭 Provider 连接。"""
        pass
