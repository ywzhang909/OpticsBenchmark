# utils — Standalone Scripts

One-off maintenance scripts (not part of the `src/` package).

| File | Description |
|------|-------------|
| `list_provider_support_models.py` | List the LLM models a provider exposes and save them as a Markdown table: built-in presets for openai/qwen/deepseek/glm/kimi/together/ollama/anthropic/google/mistral/bedrock, `--provider` or interactive menu, `--config` to reuse `configs/llm/*.yaml`, default output `models/<provider>_models.md` |
| `build_finetune_dataset.py` | Build OpenAI fine-tuning JSONL dataset from benchmark gold answers |
| `paper_data_to_dateset.py` | Convert raw paper data into benchmark dataset format |
| `paper_review_filter.py` | Filter markdown paper reviews via LLM prompt into a consolidated JSONL |
| `review_convert_jsonl.py` | Convert filtered markdown reviews into `data_v1/reviews_data.jsonl` with completeness checks |
| `reviews_to_meta-reviews.py` | Generate `data_v1/meta-reviews_data.jsonl` from four inputs per record: prompt, reviewer comments (`filted_reviews_md`), paper text (`filted_papers`) and that paper's figures (`filted_papers/<title>_images`); strengths/weaknesses are stored as lists of bullet points |
| `reviews_to_strengths_weakness.py` | Generate `data_v1/strengths-weaknesses_data.jsonl` via `prompts/paper_review/strengths_weakness.txt`: reviews built only from `reviews_data.jsonl` (`## Reviewer #N` sections) plus paper text and figures; strict-JSON parsing with schema validation (quote `Reviewer #N:` prefix with reviewer membership, stance enum, dispute reviewer membership) |
