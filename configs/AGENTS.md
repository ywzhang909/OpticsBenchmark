# Configuration

**Path:** `configs/` — YAML-driven configuration for agents, tasks, evaluation, and system.

## Overview

Four-part config system: pluggable agent configs (7 LLM providers, 11 model configs), task configs (8 task types), evaluation configs, and a global system config. All snake_case, `${ENV_VAR}` for secrets.

## Files

```
configs/
├── agents/              # 7 provider subdirectories, 18 YAML files
│   ├── anthropic/      # claude-3.yaml + template.yaml
│   ├── bedrock/        # bedrock.yaml + template.yaml
│   ├── google/         # gemini.yaml + template.yaml
│   ├── groq/           # groq.yaml + template.yaml
│   ├── ollama/         # ollama.yaml + template.yaml
│   ├── openai/         # 5 model configs (gpt-4, deepseek-v4-pro, llama4-scout,
│   │                   #   mistral-medium-3.5, qwen3.7-max) + template.yaml
│   └── together/       # together.yaml + template.yaml
├── tasks/               # 9 YAML files: lens_design, system_analysis, paper_review,
│                        #   paper_retrieval_eval, paper_info_extract, multi_doc_summary,
│                        #   optics_question_answer, research_overview + template
├── evaluation/          # 1 YAML file: template.yaml
├── eval_scoring.yaml    # Composite scoring config (dimensions, weights, anti-patterns)
└── system.yaml          # Global: logging, parallel, sandbox, rate-limit,
                         #   evaluation, export, security, paths
```

## Conventions

- **Field naming**: snake_case everywhere (agent name, model name, api_key, max_tokens).
- **Secrets**: `${OPENAI_API_KEY}` syntax — expanded at load time by `ConfigParser.expand_env_vars()`.
- **Schema per domain**:

### Agent YAML (`configs/agents/*/`)
```yaml
agent:
  name: "gpt-4"
  version: "1.0.0"
  description: "OpenAI GPT-4 agent for optical design tasks"
model:
  provider: "openai"
  name: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.0
  max_tokens: 4096
execution:
  timeout: 300
  max_retries: 3
```

### Task YAML (`configs/tasks/*.yaml`)
```yaml
task:
  id: "lens_design"
  name: "Lens Design Optimization"
  description: "Design and optimize optical lens systems"
  category: "lens_design"
  difficulty: 3
dataset:
  path: "dataset/processed/lens_design.jsonl"
  num_samples: 50
  shuffle: false
evaluation:
  scoring_method: "metric_based"
  metrics:
    - name: "mtf_performance"
      type: "numeric"
  success_criteria:
    - metric: "mtf_performance"
      operator: ">="
      value: 0.7
prompt:
  system_file: "prompts/system/optical_agent.txt"
  template_file: "prompts/templates/lens_design.txt"
environment:
  type: "optical_sandbox"
  sandbox:
    timeout: 600
cost:
  max_cost_per_task: 10.0
```

### Evaluation YAML (`configs/evaluation/*.yaml`)
```yaml
scoring_method: "composite"  # metric_based | exact_match | partial_match |
                             # summarization/rouge | citation/retrieval | composite
metrics:
  - name: "mtf_performance"
    type: "numeric"
success_criteria:
  - metric: "mtf_performance"
    operator: ">="
    value: 0.7
dimensions:            # composite only
  - name: "optical_accuracy"
    weight: 0.25
llm_judge_weight: 0.3  # composite only
static_weight: 0.7
anti_patterns:          # composite only
  - name: "empty_output"
    penalty: 0.6
llm_judge:              # composite only
  provider: "openai"
  model: "gpt-4"
```

### System YAML (`configs/system.yaml`)
```yaml
logging:
  level: "INFO"
  file: "logs/optis.log"
  rotation: "100 MB"
parallel:
  max_workers: 4
sandbox:
  timeout: 300
rate_limit:
  requests_per_minute: 60
```

## Known Issues

- `system.yaml` references Docker sandbox (`image: optis_benchmark/sandbox:latest`) but no Dockerfile exists.
- `system.yaml` security config `save_api_keys: false` has `# NEVER set to true in production` comment but no runtime enforcement.
- Agent configs list `provider` but factory is hardcoded to known providers — no plugin/discovery mechanism.
- `eval_scoring.yaml` overlaps with the composite scoring config that could be placed in `configs/evaluation/`.
