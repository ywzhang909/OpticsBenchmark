# Prompts

**Path:** `prompts/` — LLM prompt templates for OptiS Benchmark.

Two-tier prompt architecture: **system prompts** (agent role definition) and **task prompts** (per-task instruction templates + zero-shot prompts).

---

## Directory Structure

```
prompts/
├── system/                        # Agent system prompts (role definition)
│   ├── optical_agent.txt         # Optical design & engineering agent
│   └── research_agent.txt        # Academic research & paper analysis agent
├── templates/                     # Task-specific prompt templates (Handlebars)
│   ├── lens_design.txt           # Lens design optimization
│   ├── system_analysis.txt       # Optical system performance analysis
│   ├── paper_review.txt          # Academic paper review
│   ├── paper_retrieval.txt       # Paper retrieval & citation
│   ├── multi_doc_summary.txt     # Multi-document summarization
│   └── research_overview.txt     # Research field overview
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

Handlebars-style templates with `{{variable}}` and `{{#each}}` constructs. Variables are injected from task YAML configs (`configs/tasks/*.yaml`) at runtime.

| Template | Task | Variables | Language |
|----------|------|-----------|----------|
| `lens_design.txt` | Lens design optimization | `focal_length`, `f_number`, `field_of_view`, `working_distance`, `wavelength_range`, `mtf_frequency`, `mtf_target`, `wavefront_error`, `distortion_target`, `transmission_target`, `constraints` | Chinese |
| `system_analysis.txt` | Optical system analysis | `system_type`, `application`, `entrance_pupil`, `focal_length`, `lens_file`, `system_data`, `analysis_items` | Chinese |
| `paper_review.txt` | Paper review & critique | `paper_id`, `title`, `authors`, `venue`, `year`, `paper_content` | Chinese |
| `paper_retrieval.txt` | Paper retrieval & citation | `topic`, `time_range`, `specific_keywords`, `query` | Chinese |
| `multi_doc_summary.txt` | Multi-document summarization | `topic`, `num_documents`, `total_length`, `document_sources`, `summary_type`, `documents` | Chinese |
| `research_overview.txt` | Research field overview | `field`, `subfields`, `time_range`, `language`, `research_topic` | Chinese |

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
│ optical_agent   │◄───────────│ lens_design.yaml         │
│ research_agent  │            │ paper_review.yaml        │
└─────────────────┘            │ system_analysis.yaml    │
                               │   ...                    │
       ┌───────────────────────┤                          │
       │                       │ template_file            │
       ▼                       └──────────────────────────┘
┌─────────────────┐
│ lens_design.txt │  ← Handlebars template with {{variables}}
│ paper_review.txt│
│   ...           │
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
