# LLM ABSTRACTION LAYER

**Path:** `src/llm/` — 8 model classes + 7 provider classes.

## OVERVIEW
Separate LLM abstraction layer (known duplication with `core/agent.py`). Used by `llm_pred.py` for prediction pipeline. Splits concerns: model classes handle formatting/tokenization, provider classes handle API transport.

## STRUCTURE
```
src/llm/
├── base.py           # BaseLLM ABC: chat(messages, provider, setup) → dict
├── models/           # Model-specific implementations
│   ├── ClaudeLLM.py, DeepSeekLLM.py, GeminiLLM.py, GlmLLM.py
│   ├── GroqLLM.py, LlamaLLM.py, MistralLLM.py, OllamaLLM.py, QwenLLM.py
└── providers/        # Provider-specific API clients
    ├── AnthropicProvider.py, BedrockProvider.py, GoogleProvider.py
    ├── GroqProvider.py, OllamaProvider.py, OpenAIProvider.py, TogetherAIProvider.py
```

## KEY PATTERNS
- **Model/Provider split**: Models implement `chat()` with provider-agnostic logic; providers handle raw API calls.
- **BaseLLM ABC**: `chat(messages, provider, setup)` → dict with `content`, `cost`, `usage`. `model_name` property.
- **Known duplication**: `core/agent.py` has its own `BaseAgent` with 7 provider implementations. This layer grew independently — eventual unification planned.

## KNOWN ISSUES
- Duplication with `core/agent.py` — both layers implement LLM chat.
- Not all model+provider combinations are validated.
