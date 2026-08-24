# src/llm/models — LLM Model Implementations

## Purpose

One module per supported model family. Each class extends `BaseLLM` and implements `chat()` (single-turn generation) and `close()`.

## Files

| File | Class | Provider backend |
|------|-------|-----------------|
| `gpt_llm.py` | `GPTLLM` | OpenAI (Chat Completions / Responses API) |
| `qwen_llm.py` | `QwenLLM` | OpenAI-compatible (Alibaba Cloud) |
| `deepseek_llm.py` | `DeepSeekLLM` | OpenAI-compatible |
| `llama_llm.py` | `LlamaLLM` | Together AI / OpenAI-compatible |
| `mistral_llm.py` | `MistralLLM` | Official `mistralai` SDK |
| `gemini_llm.py` | `GeminiLLM` | Google GenAI SDK |
| `claude_llm.py` | `ClaudeLLM` | Anthropic SDK |
| `glm_llm.py` | `GLMLLM` | OpenAI-compatible (Zhipu) |
| `kimi_llm.py` | `KimiLLM` | OpenAI-compatible (Moonshot) |
| `ollama_llm.py` | `OllamaLLM` | Local Ollama endpoint |

Naming convention: `snake_case` file name, PascalCase class name (see STANDARDS.md §2).
