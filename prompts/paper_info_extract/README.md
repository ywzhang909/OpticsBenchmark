# prompts/paper_info_extract — Paper Info Extraction Prompts

## Purpose

Zero-shot prompt for structured information extraction from optical science papers.

## Files

| File | Description |
|------|-------------|
| `zero-shot_v1.0.txt` | Zero-shot extraction prompt (13 fields, JSON output) |

## Task

Extract 13 fields from an optical science paper: Title, Publication Year, DOI, Journal, Ten
Keywords, Authors, Corresponding Authors, Affiliations, Abstract, Objectives, Novelty,
Methods, and Performance Metrics.

Direct fields (Title, Year, DOI, Journal, Authors, Corresponding Authors, Affiliations,
Abstract) must be extracted verbatim; summarized fields (Keywords, Objectives, Novelty,
Methods, Performance Metrics) are capped at 300 words each.

## Prompt structure

After a two-line header (comment + blank, skipped at load time), the file uses four sections:

```
***TASK***          Extraction objective
***INPUT***         Paper full text (from an uploaded PDF)
***OUTPUT***        The 13 fields as a JSON string
***DOCUMENTATION*** Extraction rules: verbatim quoting, no fabrication,
                    no external knowledge, empty-string/empty-list fallbacks
```

## Usage

Reference the prompt via `task.prompt_file` in an LLM config:

```yaml
task:
  dataset_path: "dataset/paper_info_extract/dataset_json/dataset_v1.json"
  prompt_file: "prompts/paper_info_extract/zero-shot_v1.0.txt"
  gold_answer_path: "dataset/paper_info_extract/dataset_json/gold_answer_v1.json"
```

Example in the repo: `configs/llm/GPT_OpenAI.yaml`.

Combined with the system prompt template in `prompts/system/templates.txt`, the system
message supplies global model instructions while this file supplies the task instructions.