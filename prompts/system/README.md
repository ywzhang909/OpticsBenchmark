# prompts/system — Generic System Prompt Template

## Purpose

This directory holds the **generic system prompt template** for LLM tasks in Optis
Benchmark. It defines the system-level instructions (`system` message) that accompany the
task-specific user prompts stored in the task directories.

The project targets **plain LLMs**, not agents. The system prompt template does not define
an agent role or tool-using agent behavior; it tells the LLM how to behave globally so that
it can execute the task instructions found in the user message.

## Files

| File | Description |
|------|-------------|
| `templates.txt` | Generic system prompt template with `{{placeholders}}` to be filled per task |

## Why this template exists

The task-specific zero-shot prompts (`prompts/paper_info_extract/zero-shot_v1.0.txt`,
`prompts/paper_review/zero-shot_v1.0.txt`, `prompts/optics_question_answers/zero-shot_v1.0.txt`)
all share a four-section after a two-line header:

```
***TASK***          The objective to accomplish
***INPUT***         The material to work from
***OUTPUT***        The exact output format
***DOCUMENTATION*** Rules, constraints, and evaluation criteria
```

Analyzing what those task prompts require at the model level yields a common set of
system-level instructions:

- Follow the user message's structured instructions exactly;
- Ground every claim in the provided material, never fabricate or use external knowledge;
- Honor quoting and citation rules (verbatim fields, citation markers);
- Require structured output, skip unavailable information explicitly;
- Keep the response in the requested language;
- Remain rigorous and professional throughout.

`templates.txt` captures these shared instructions as a single template filled via
placeholders.

## Placeholders

| Placeholder | Meaning |
|-------------|---------|
| `{{task_name}}` | Name of the task being executed (e.g., `paper_info_extract`) |
| `{{task_specification}}` | Key requirements, output contract, or constraints the system should be aware of |
| `{{output_language}}` | Language the model must respond in (e.g., `English`, `Chinese`) |

## How the system prompt is consumed

Each LLM backend accepted a flat message whose `system` key is mapped to the model's system
prompt / instructions field:

- `src/llm/models/gpt_llm.py` (`system` key → `instructions`)
- `src/llm/models/claude_llm.py` (`system`/`developer` → system messages)
- `src/llm/models/gemini_llm.py` (`system`/`input` → system instruction)
- `src/llm/models/glm_llm.py`, `src/llm/models/llama_llm.py`, `src/llm/models/kimi_llm.py`
  (`system` → system message)

Task prompts are loaded from `task.prompt_file` and sent as the `prompt` key (user
content). Providing a system prompt is complementary and orthogonal to the task prompt.

## Usage

1. Copy `templates.txt` and fill in the `{{placeholders}}`.
2. Reference the filled system prompt in the composed request, e.g. building the message as
   `{"system": <system_prompt>, "prompt": <task_prompt>}`.

Example:

```bash
# Inspect the template
cat prompts/system/templates.txt
```

Note: `templates.txt` keeps the same two-line header (comment line + blank line) as the task
prompt files so that, whenever it is loaded through the prompt loader
(`src/core/llm_runner.py:_load_prompt`), the header is skipped automatically.

## Creating variant system prompts

Place new or task-specific system prompt files in this directory and document them here.
Keep the structure of `templates.txt` so that every variant stays consistent with the
two-line header and placeholder conventions.