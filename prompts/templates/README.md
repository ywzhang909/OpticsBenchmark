# prompts/templates — Shared Task Templates

## Purpose

Reusable Handlebars-style task templates shared across tasks. These are variable-based
templates (with conditional and iteration blocks), distinct from the fixed zero-shot prompts
in the task directories (`paper_info_extract/`, `paper_review/`,
`optics_question_answers/`).

## Files

| File | Description |
|------|-------------|
| `paper_review.txt` | Base paper review template |

## Template syntax

The templates support Handlebars-style constructs:

- **Variable injection**: `{{focal_length}}`, `{{field_of_view}}`
- **Conditional blocks**: `{{#if variable}}...{{/if}}`
- **Iteration**: `{{#each items}}...{{/each}}`
- **Helper functions**: `{{add @index 1}}` (1-based indexing)

Example (`paper_review.txt`):

```
{{#if paper_id}}
- **Paper ID**: {{paper_id}}
{{/if}}

{{#each items}}
- {{this}}
{{/each}}
```

## Usage

```bash
# View a shared template
cat prompts/templates/paper_review.txt
```

## Planned templates

`lens_design.txt`, `system_analysis.txt`, `paper_retrieval.txt`, `multi_doc_summary.txt`,
`research_overview.txt`.

Add new shared templates in this directory and document them in the table above.