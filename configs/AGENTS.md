# CONFIGURATION

**Path:** `configs/` — YAML-driven configuration for agents, tasks, and system.

## OVERVIEW
Three-part config system: pluggable agent configs (7 LLM providers), task configs (6 task types), and a global system config. All snake_case, `${ENV_VAR}` for secrets.

## FILES
```
configs/
├── agents/             # 8 YAML files: gpt-4, claude-3, gemini, groq, ollama, bedrock, together + template
├── tasks/              # 7 YAML files: lens_design, system_analysis, paper_review, paper_retrieval_eval, multi_doc_summary, research_overview + template
└── system.yaml         # Global: logging, parallel, sandbox, rate-limit, evaluation, export, security, paths
```

## CONVENTIONS
- **Field naming**: snake_case everywhere (agent name, model name, api_key, max_tokens).
- **Secrets**: `${OPENAI_API_KEY}` syntax — expanded at load time by `ConfigParser.expand_env_vars()`.
- **Schema per domain**:
  - Agent YAML: `agent.{name,version,description}` + `model.{provider,name,api_key,temperature,max_tokens}` + `execution.{timeout,max_retries}`
  - Task YAML: `task.{id,name,description,category,difficulty}` + `dataset.{path,num_samples,shuffle}` + `evaluation.{scoring_method,...}`
  - System YAML: grouped under `logging`, `parallel`, `sandbox`, `rate_limit`, `evaluation`, `export`, `security`, `paths`

## KNOWN ISSUES
- `system.yaml` references Docker sandbox (`image: optis_benchmark/sandbox:latest`) but no Dockerfile exists.
- `system.yaml` security config `save_api_keys: false` has `# NEVER set to true in production` comment but no runtime enforcement.
- Agent configs list `provider` but factory is hardcoded to known providers — no plugin/discovery mechanism.
