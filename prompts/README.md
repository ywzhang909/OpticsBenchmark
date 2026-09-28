# Prompts

**Path:** `prompts/` — LLM prompt files for Optis Benchmark.

Two-tier prompt architecture for plain LLM inference (no agents): a **system prompt
template** (generic model-level instructions) plus **task-specific prompts** (per-task
zero-shot instructions and shared Handlebars templates).

---

## Directory Structure

```
prompts/
├── system/                        # Generic system prompt template
│   └── templates.txt             # System prompt template with {{placeholders}}
├── templates/                     # Shared task templates (Handlebars)
│   └── paper_review.txt          # Paper review task template
├── paper_info_extract/            # Paper info extraction task
│   └── zero-shot_v1.0.txt       # Zero-shot extraction prompt
├── paper_review/                  # Paper review task
│   └── zero-shot_v1.0.txt       # Zero-shot review prompt
├── optics_question_answers/       # Optics Q&A task
│   └── zero-shot_v1.0.txt       # Zero-shot Q&A prompt
└── README.md                      # This file
```

---

## System Prompt Template (`system/`)

`prompts/system/templates.txt` is a generic system prompt to be filled in and sent as the
`system` message. It is derived from the shared structure of the task zero-shot prompts
(`***TASK***`, `***INPUT***`, `***OUTPUT***`, `***DOCUMENTATION***`) and tells the model to:

- Follow the user message's structured instructions exactly;
- Ground every claim in the provided material (no fabrication, no external knowledge);
- Honor quoting and citation rules;
- Keep to the required output format and language.

It uses `{{placeholders}}` (e.g., `{{task_name}}`, `{{task_specification}}`,
`{{output_language}}`) that are filled in per task. See `prompts/system/README.md`.

---

## Task Prompt Files (`paper_info_extract/`, `paper_review/`, `optics_question_answers/`)

Each task directory holds a zero-shot prompt file referenced by `task.prompt_file` in the
LLM configs (`configs/llm/*.yaml`).

| Task | File | Output |
|------|------|--------|
| Paper info extraction | `paper_info_extract/zero-shot_v1.0.txt` | JSON with 13 fields |
| Paper review | `paper_review/zero-shot_v1.0.txt` | Structured review report |
| Optics Q&A | `optics_question_answers/zero-shot_v1.0.txt` | Multi-paragraph academic response |

All zero-shot prompts share the same structure: a two-line header (comment + blank, skipped
at load time) followed by `***TASK***`, `***INPUT***`, `***OUTPUT***`, and
`***DOCUMENTATION***` sections.

---

## Shared Task Templates (`templates/`)

`prompts/templates/` holds Handlebars-style task templates with variable injection and
conditional/iteration blocks:

- `paper_review.txt` — paper review template using `{{variables}}`, `{{#if}}`, `{{#each}}`,
  and `{{add @index 1}}` helpers.

Additional templates are planned: `lens_design.txt`, `system_analysis.txt`,
`paper_retrieval.txt`, `multi_doc_summary.txt`, `research_overview.txt`.

---

## Prompt Flow

```
LLM config (configs/llm/*.yaml)
        │  task.prompt_file ──────────► task zero-shot prompt (user content, "prompt" key)
        │  system prompt (optional) ───► system template (filled in, "system" key)
        ▼
┌──────────────────────────────┐
│ LLM request                 │
│  system = filled template   │
│  user   = rendered prompt + │
│           task data         │
└──────────────────────────────┘
```

The two-line header of every prompt file is skipped by `_load_prompt()` in
`src/core/llm_runner.py`.

---

## Usage

```bash
# View the system prompt template
cat prompts/system/templates.txt

# View a zero-shot prompt
cat prompts/paper_info_extract/zero-shot_v1.0.txt

# View a shared task template
cat prompts/templates/paper_review.txt
```

---

## Contributing

1. Add or edit generic system-level instructions in `prompts/system/templates.txt`.
2. Add task-specific zero-shot prompts in the matching task subdirectory, using the
   `***TASK***` / `***INPUT***` / `***OUTPUT***` / `***DOCUMENTATION***` structure.
3. Add shared variable-based templates in `prompts/templates/` using Handlebars syntax.
4. Reference task prompt paths in `configs/llm/*.yaml` via `task.prompt_file`.
5. Keep the two-line header (comment + blank) so the loader skips it correctly.