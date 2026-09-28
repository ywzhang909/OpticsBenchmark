"""
Optis Benchmark - LLM Module

Layered architecture for providers and LLM models.
- providers/: asynchronous wrappers around each vendor's SDK
- models/: per-model call logic, with support for switching providers
"""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLM, build_response_format

# ---------------------------------------------------------------------------
# Provider / LLM type maps (lazily imported)
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, str] = {
    "openai": "src.llm.providers.openai_provider:OpenAIProvider",
    "anthropic": "src.llm.providers.anthropic_provider:AnthropicProvider",
    "google": "src.llm.providers.google_provider:GoogleProvider",
    "mistral": "src.llm.providers.mistral_provider:MistralProvider",
    "ollama": "src.llm.providers.ollama_provider:OllamaProvider",
    "bedrock": "src.llm.providers.bedrock_provider:BedrockProvider",
    "together": "src.llm.providers.together_ai_provider:TogetherAIProvider",
    "dashscope": "src.llm.providers.dashscope_provider:DashScopeProvider",
}

_LLM_MAP: dict[str, str] = {
    "qwen": "src.llm.models.qwen_llm:QwenLLM",
    "deepseek": "src.llm.models.deepseek_llm:DeepSeekLLM",
    "llama": "src.llm.models.llama_llm:LlamaLLM",
    "mistral": "src.llm.models.mistral_llm:MistralLLM",
    "gemini": "src.llm.models.gemini_llm:GeminiLLM",
    "claude": "src.llm.models.claude_llm:ClaudeLLM",
    "ollama": "src.llm.models.ollama_llm:OllamaLLM",
    "glm": "src.llm.models.glm_llm:GlmLLM",
    "gpt": "src.llm.models.gpt_llm:GPTLLM",
    "kimi": "src.llm.models.kimi_llm:KimiLLM",
}


def _lazy_import(target: str) -> Any:
    """Import on demand: 'src.llm.providers.openai_provider:OpenAIProvider' -> the class."""
    import importlib

    module_path, _, class_name = target.partition(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_provider(provider_config: dict[str, Any]) -> Any:
    """Create a provider instance from its config.

    Args:
        provider_config: Provider config dict; must contain a "type" field.
            Example: {"type": "openai", "api_key": "...", "base_url": "..."}

    Returns:
        The provider instance

    Raises:
        ValueError: Unsupported provider type
    """
    provider_type = provider_config.get("type", "")
    if not provider_type:
        raise ValueError("provider_config must contain a 'type' field")

    dotted = _PROVIDER_MAP.get(provider_type)
    if dotted is None:
        raise ValueError(f"Unsupported provider type: {provider_type}")

    cls = _lazy_import(dotted)
    kwargs = {k: v for k, v in provider_config.items() if k != "type"}
    return cls(**kwargs)


def create_llm(model_config: dict[str, Any]) -> Any:
    """Create an LLM instance from its config.

    Args:
        model_config: Model config dict; must contain a "type" field.
            Example: {"type": "qwen", "name": "qwen3.5-plus"}

    Returns:
        The LLM instance

    Raises:
        ValueError: Unsupported LLM model type
    """
    model_type = model_config.get("type", "")
    if not model_type:
        raise ValueError("model_config must contain a 'type' field")

    dotted = _LLM_MAP.get(model_type)
    if dotted is None:
        raise ValueError(f"Unsupported LLM model type: {model_type}")

    cls = _lazy_import(dotted)
    model_name = model_config.get("name", "")
    return cls(model_name=model_name)


__all__ = [
    # Base class
    "BaseLLM",
    # Utility function
    "build_response_format",
    # Factory functions
    "create_provider",
    "create_llm",
]
