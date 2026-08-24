# src/llm/providers — API Client Providers

## Purpose

HTTP client wrappers, one per API surface. Each provider exposes the client object consumed by its matching model class.

## Files

| File | Class | Notes |
|------|-------|-------|
| `openai_provider.py` | `OpenAIProvider` | Chat Completions + Responses API |
| `anthropic_provider.py` | `AnthropicProvider` | Anthropic Messages API |
| `google_provider.py` | `GoogleProvider` | Google GenAI SDK |
| `mistral_provider.py` | `MistralProvider` | Official Mistral AI SDK |
| `ollama_provider.py` | `OllamaProvider` | Local endpoint |
| `bedrock_provider.py` | `BedrockProvider` | AWS Bedrock |
| `together_ai_provider.py` | `TogetherAIProvider` | Together AI (OpenAI-compatible) |

Registries: `_PROVIDER_CLASSES` in `__init__.py`, top-level map in `src/llm/__init__.py`.
