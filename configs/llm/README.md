# configs/llm — LLM Provider Configs

## Purpose

One full YAML config per supported model family (provider, model, generation params, task section). See parent [README](../README.md) for field reference.

## Files

`GPT_OpenAI.yaml`, `claude_anthropic.yaml`, `gemini_google.yaml`, `kimi_openai.yaml`, `glm_openai.yaml`, `deepseek_openai.yaml`, `qwen_openai.yaml`, `llama_openai.yaml`, `mistral_official.yaml`

## Usage

```bash
uv run python src/llm_pred.py -a configs/llm/qwen_openai.yaml -t paper_info_extract
```
