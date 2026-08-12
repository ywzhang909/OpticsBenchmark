# Prompts

**Path:** `prompts/` — LLM prompt templates for Optis Benchmark.

Two-tier prompt architecture: **system prompts** (agent role definition) and **task prompts** (per-task instruction templates + zero-shot prompts).

---

## Directory Structure

```
prompts/
├── system/                        # Agent system prompts (role definition)
│   ├── optical_agent.txt         # Optical design & engineering agent
│   └── research_agent.txt        # Academic research & paper analysis agent
├── templates/                     # Task-specific prompt templates
│   └── paper_review.txt          # Paper review task template
├── paper_info_extract/            # Paper info extraction task
│   └── zero-shot_v1.0.txt       # Zero-shot extraction prompt
├── paper_review/                  # Paper review task
│   └── zero-shot_v1.0.txt       # Zero-shot review prompt (placeholder)
└── README.md                      # This file
```

---

## System Prompts (`system/`)

Define the agent's role, expertise, work principles, and output format. They are referenced by `system_prompt_file` in agent configs (`configs/agents/*.yaml`) and task configs (`configs/tasks/*.yaml`).

### `optical_agent.txt` — Optical Design Agent

| Aspect | Content |
|--------|---------|
| **Role** | Professional optical design engineer |
| **Expertise** | Lens design & optimization, system performance analysis, tolerance analysis, aberration correction |
| **Software** | Zemax OpticStudio (ZOS-API), CODE V, ASAP |
| **Programming** | Python (NumPy, SciPy, matplotlib), data analysis & visualization |
| **Principles** | Accuracy, completeness, reproducibility, transparency |
| **Output format** | Structured markdown with design parameters, system configuration table, performance metrics |
| **Language** | Chinese |
| **Used by** | `lens_design`, `system_analysis` tasks |

### `research_agent.txt` — Research Agent

| Aspect | Content |
|--------|---------|
| **Role** | Academic research & paper analysis specialist |
| **Expertise** | Paper review, literature retrieval & citation, multi-document summarization, research field overview |
| **Principles** | Accuracy, comprehensiveness, objectivity, structured output, traceability |
| **Output format** | Task-specific markdown templates (review scores, citation lists, structured summaries) |
| **Language** | Chinese |
| **Used by** | `paper_review`, `paper_retrieval`, `multi_doc_summary`, `research_overview` tasks |

---

## Task Prompt Templates (`templates/`)

Currently contains one task-specific prompt template. Additional templates are planned.

| Template | Task | Language |
|----------|------|----------|
| `paper_review.txt` | Paper review & critique | Chinese |

Additional templates are planned: `lens_design.txt`, `system_analysis.txt`, `paper_retrieval.txt`, `multi_doc_summary.txt`, `research_overview.txt`.

### Template Features

- **Variable injection**: `{{focal_length}}`, `{{field_of_view}}`
- **Conditional blocks**: `{{#if variable}}...{{/if}}`
- **Iteration**: `{{#each items}}...{{/each}}`
- **Helper functions**: `{{add @index 1}}` (1-based indexing)
- **Structured output**: All templates specify exact markdown output format and evaluation criteria

---

## Zero-Shot Prompts (`paper_info_extract/`, `paper_review/`)

### `paper_info_extract/zero-shot_v1.0.txt` — Paper Info Extraction

English-language zero-shot prompt for structured information extraction from optical science papers.

| Aspect | Content |
|--------|---------|
| **Task** | Extract 13 fields from optical science papers: Title, Publication Year, DOI, Journal, Ten Keywords, Authors, Corresponding Authors, Affiliations, Abstract, Objectives, Novelty, Methods, Performance Metrics |
| **Output format** | JSON |
| **Field rules** | Title/Year/DOI/Journal/Authors/Corresponding Authors/Affiliations/Abstract → direct extraction (must match original text); Keywords/Objectives/Novelty/Methods/Performance Metrics → summarize from full text |
| **Constraints** | 300 words max per summarized field; no external knowledge; exact quotes required for direct fields |

### `paper_review/zero-shot_v1.0.txt` — Paper Review (Placeholder)

Currently contains only a header; content to be developed.

---

## Prompt Flow

```
Agent config                         Task config
(system_prompt_file)                 (system_file + template_file)
         │                                  │
         ▼                                  ▼
┌─────────────────┐            ┌──────────────────────────┐
│ optical_agent   │◄───────────│ paper_info_extract.yaml  │
│ research_agent  │            │ paper_review.yaml        │
└─────────────────┘            │ optics_question_answer.yaml│
                               │   ...                    │
       ┌───────────────────────┤                          │
       │                       │ template_file            │
       ▼                       └──────────────────────────┘
┌─────────────────┐
│ paper_review.txt│  ← Handlebars template with {{variables}}
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Final prompt    │  = system_prompt + rendered_template + task_data
│ sent to LLM     │
└─────────────────┘
```

---

## Usage

```bash
# View a system prompt
cat prompts/system/optical_agent.txt

# View a task template
cat prompts/templates/lens_design.txt

# View a zero-shot prompt
cat prompts/paper_info_extract/zero-shot_v1.0.txt
```

---

## Contributing

1. Place new system prompts in `prompts/system/`
2. Place new task templates in `prompts/templates/` using Handlebars syntax
3. Place zero-shot/variant prompts in task-specific subdirectories
4. Reference the prompt paths in `configs/tasks/*.yaml` (`system_file`, `template_file`, `task_file`)
