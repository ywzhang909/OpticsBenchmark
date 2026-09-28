# prompts/optics_question_answers — Optics Q&A Prompts

## Purpose

Zero-shot prompt for literature-based question answering in optics.

## Files

| File | Description |
|------|-------------|
| `zero-shot_v1.0.txt` | Zero-shot question answering prompt (multi-paragraph, cited) |

## Task

Answer an input research question by consulting the specified paper(s) or literature and
provide a comprehensive, multi-paragraph response grounded in the provided references.

## Prompt structure

After a two-line header (comment + blank, skipped at load time), the file uses four sections:

```
***TASK***          Question answering objective
***INPUT***         An optics research question plus one or more reference documents
***OUTPUT***        Multi-paragraph academic response between [Response Start]
                    and [Response End]
***DOCUMENTATION*** Citation rules: reference numbers at the end of sentences
                    (e.g., "[1]"), cite only directly supporting documents,
                    synthesize across references instead of copying
```

The output requires inline citation markers (e.g., `[1]`, `[1][2]`) and does not require a
separate reference list.

## Usage

Reference the prompt via `task.prompt_file` in an LLM config:

```yaml
task:
  dataset_path: "dataset/optics_question_answer/..."  # Q&A dataset
  prompt_file: "prompts/optics_question_answers/zero-shot_v1.0.txt"
```

Combined with the system prompt template in `prompts/system/templates.txt`, the system
message supplies global model instructions while this file supplies the task instructions.