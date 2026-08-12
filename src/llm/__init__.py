"""
OptiS Benchmark - LLM Module

Provider 和 LLM 模型的分层架构。
- providers/: 各厂商 SDK 异步封装
- models/: 各模型的调用逻辑，支持多 Provider 切换
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Provider / LLM 类型映射（延迟导入）
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, str] = {
    "openai": "src.llm.providers.OpenAIProvider",
    "anthropic": "src.llm.providers.AnthropicProvider",
    "google": "src.llm.providers.GoogleProvider",
    "groq": "src.llm.providers.GroqProvider",
    "ollama": "src.llm.providers.OllamaProvider",
    "bedrock": "src.llm.providers.BedrockProvider",
    "together": "src.llm.providers.TogetherAIProvider",
}

_LLM_MAP: dict[str, str] = {
    "qwen": "src.llm.models.QwenLLM",
    "deepseek": "src.llm.models.DeepSeekLLM",
    "llama": "src.llm.models.LlamaLLM",
    "mistral": "src.llm.models.MistralLLM",
    "gemini": "src.llm.models.GeminiLLM",
    "claude": "src.llm.models.ClaudeLLM",
    "groq": "src.llm.models.GroqLLM",
    "ollama": "src.llm.models.OllamaLLM",
    "glm": "src.llm.models.GlmLLM",
    "gpt": "src.llm.models.GPTLLM",
}


def _lazy_import(dotted_path: str) -> Any:
    """按需导入：'src.llm.providers.OpenAIProvider' -> OpenAIProvider 类。"""
    import importlib

    module_path, _, class_name = dotted_path.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


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

    dotted = _PROVIDER_MAP.get(provider_type)
    if dotted is None:
        raise ValueError(f"不支持的 Provider 类型: {provider_type}")

    cls = _lazy_import(dotted)
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

    dotted = _LLM_MAP.get(model_type)
    if dotted is None:
        raise ValueError(f"不支持的 LLM 模型类型: {model_type}")

    cls = _lazy_import(dotted)
    model_name = model_config.get("name", "")
    return cls(model_name=model_name)


__all__ = [
    # 基类
    "BaseLLM",
    # 工具函数
    "build_response_format",
    # 工厂函数
    "create_provider",
    "create_llm",
]
