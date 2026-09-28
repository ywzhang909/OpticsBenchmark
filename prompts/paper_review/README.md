# prompts/paper_review — Paper Review Prompts

## Purpose

Zero-shot prompt for rigorous, critical peer review of optics manuscripts.

## Files

| File | Description |
|------|-------------|
| `zero-shot_v1.0.txt` | Zero-shot review prompt (structured review report) |

## Task

Systematically identify and articulate the issues, weaknesses, logical gaps, methodological
flaws, or errors in an input manuscript, covering the full chain from research motivation to
final conclusions.

## Prompt structure

After a two-line header (comment + blank, skipped at load time), the file uses four sections:

```
***TASK***          Critical review objective
***INPUT***         Manuscript full text, optionally with its references
***OUTPUT***        Structured report between [Review Start] and [Review End]:
                    overall assessment, itemized issues, review verdict
***DOCUMENTATION*** Review standards: ground each criticism in specific content,
                    avoid vague remarks, acknowledge strengths, no fabricated flaws
```

The review must reference concrete passages, equations, figures, or data from the manuscript
and must not fabricate non-existent problems.

## Usage

Reference the prompt via `task.prompt_file` in an LLM config:

```yaml
task:
  dataset_path: "dataset/paper_review/..."         # review dataset
  prompt_file: "prompts/paper_review/zero-shot_v1.0.txt"
```

Note: a separate Handlebars template for the same task exists at
`prompts/templates/paper_review.txt`; it is a shared task template (variable-based), distinct
from this zero-shot prompt.

Combined with the system prompt template in `prompts/system/templates.txt`, the system
message supplies global model instructions while this file supplies the task instructions.