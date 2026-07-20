"""
OptiS Benchmark - LLM Module

Provider 和 LLM 模型的分层架构。
- providers/: 各厂商 SDK 异步封装
- models/: 各模型的调用逻辑，支持多 Provider 切换
"""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLM
from src.llm.models import (
    ClaudeLLM,
    DeepSeekLLM,
    GeminiLLM,
    GroqLLM,
    LlamaLLM,
    MistralLLM,
    OllamaLLM,
    QwenLLM,
)
from src.llm.providers import (
    AnthropicProvider,
    BedrockProvider,
    GoogleProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
    TogetherAIProvider,
)

# ---------------------------------------------------------------------------
# Provider / LLM 类型映射
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "bedrock": BedrockProvider,
    "together": TogetherAIProvider,
}

_LLM_MAP: dict[str, type] = {
    "qwen": QwenLLM,
    "deepseek": DeepSeekLLM,
    "llama": LlamaLLM,
    "mistral": MistralLLM,
    "gemini": GeminiLLM,
    "claude": ClaudeLLM,
    "groq": GroqLLM,
    "ollama": OllamaLLM,
}


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_provider(provider_config: dict[str, Any]) -> Any:
    """根据配置创建 Provider 实例。

    Args:
        provider_config: Provider 配置字典，必须包含 "type" 字段。
            示例: {"type": "openai", "api_key": "...", "base_url": "..."}

    Returns:
        Provider 实例

    Raises:
        ValueError: 不支持的 Provider 类型
    """
    provider_type = provider_config.get("type", "")
    if not provider_type:
        raise ValueError("provider_config 必须包含 'type' 字段")

    cls = _PROVIDER_MAP.get(provider_type)
    if cls is None:
        raise ValueError(f"不支持的 Provider 类型: {provider_type}")

    # 过滤掉 type 字段，其余作为构造参数
    kwargs = {k: v for k, v in provider_config.items() if k != "type"}
    return cls(**kwargs)


def create_llm(model_config: dict[str, Any]) -> Any:
    """根据配置创建 LLM 实例。

    Args:
        model_config: 模型配置字典，必须包含 "type" 字段。
            示例: {"type": "qwen", "name": "qwen3.5-plus"}

    Returns:
        LLM 实例

    Raises:
        ValueError: 不支持的 LLM 模型类型
    """
    model_type = model_config.get("type", "")
    if not model_type:
        raise ValueError("model_config 必须包含 'type' 字段")

    cls = _LLM_MAP.get(model_type)
    if cls is None:
        raise ValueError(f"不支持的 LLM 模型类型: {model_type}")

    model_name = model_config.get("name", "")
    return cls(model_name=model_name)


__all__ = [
    # 基类
    "BaseLLM",
    # Provider 类
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "GroqProvider",
    "OllamaProvider",
    "BedrockProvider",
    "TogetherAIProvider",
    # LLM 类
    "QwenLLM",
    "DeepSeekLLM",
    "LlamaLLM",
    "MistralLLM",
    "GeminiLLM",
    "ClaudeLLM",
    "GroqLLM",
    "OllamaLLM",
    # 工厂函数
    "create_provider",
    "create_llm",
]
