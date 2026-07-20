# Configuration

**Path:** `configs/` — YAML-driven configuration for agents, tasks, evaluation, and system.

## Overview

Four-part config system: pluggable agent configs (4 LLM providers, 9 model configs), task configs (3 task types), evaluation configs, and a global system config template. All snake_case, `${ENV_VAR}` for secrets.

## Files

```
configs/
├── agents/              # 4 provider subdirectories, 9 YAML files
│   ├── anthropic/      # claude-3.yaml + template.yaml
│   ├── google/         # gemini.yaml + template.yaml
│   ├── ollama/         # ollama.yaml + template.yaml
│   └── openai/         # 6 model configs (gpt-4, deepseek-v4-pro, llama4-scout,
│                       #   mistral-medium-3.5, qwen3.5-plus, qwen3.7-max) + template.yaml
├── tasks/               # 3 YAML files + template: paper_info_extract,
│                        #   paper_review, optics_question_answer
├── evaluations/         # paper_info_extract.yaml + template.yaml
├── llm/                 # LLM provider configs (4 YAML files)
├── system/
│   └── template.yaml    # Global settings template
└── README.md
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

### System YAML (`configs/system/template.yaml`)
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

- `system/template.yaml` is a minimal template with most settings commented out — not a full config.
- Agent configs list `provider` but factory is hardcoded to known providers — no plugin/discovery mechanism.
- Provider directories are incomplete: Groq, Bedrock, Together AI configs are in `configs/llm/` but agent configs only exist for Anthropic, Google, Ollama, and OpenAI.
