"""
LLM Providers - 各厂商 SDK 异步封装

每个 Provider 封装对应厂商的 SDK 客户端，提供异步调用支持。
使用延迟导入，缺失可选 SDK 时不会在模块导入阶段报错。
"""

from __future__ import annotations

from typing import Any

_PROVIDER_CLASSES: dict[str, str] = {
    "OpenAIProvider": "src.llm.providers.openai_provider",
    "AnthropicProvider": "src.llm.providers.anthropic_provider",
    "GoogleProvider": "src.llm.providers.google_provider",
    "OllamaProvider": "src.llm.providers.ollama_provider",
    "BedrockProvider": "src.llm.providers.bedrock_provider",
    "TogetherAIProvider": "src.llm.providers.together_ai_provider",
}

__all__ = list(_PROVIDER_CLASSES.keys())


def __getattr__(name: str) -> Any:
    """延迟导入：访问 src.llm.providers.XXX 时才真正导入。"""
    if name in _PROVIDER_CLASSES:
        import importlib

        mod = importlib.import_module(_PROVIDER_CLASSES[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
