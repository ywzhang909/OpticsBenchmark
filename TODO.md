# Optis Benchmark - TODO & Roadmap

> Project to-do list and development roadmap

---

## Project Status

| Status | Description |
|--------|-------------|
| `v0.1.0-alpha` | Basic framework scaffolding complete |
| `v0.2.0-alpha` | Feature refinement in progress (current) |
| `v1.0.0` | Official release (goal) |

---

## Version Roadmap

### v0.1.0-alpha (done)

- [x] Project structure design
- [x] Core module implementation (Agent, Evaluator, Runner)
- [x] Configuration file system
- [x] Basic CLI interface
- [x] Local environment
- [x] ZOS-API environment (basics)
- [x] Basic documentation

### v0.2.0-alpha (current version)

- [x] Support OpenAI GPT (Chat Completions + Responses API)
- [x] Support Anthropic Claude (official SDK)
- [x] Support Google Gemini (Interactions API)
- [x] Support Moonshot Kimi (OpenAI compatible)
- [x] Support Zhipu GLM (OpenAI SDK)
- [x] Support DeepSeek (OpenAI compatible)
- [x] Support Alibaba Qwen (OpenAI compatible)
- [x] Support Meta Llama (Together AI)
- [x] Support Mistral (official SDK)
- [x] Support Ollama (local inference)
- [x] LLM Judge evaluation implementation
- [x] Structured output support (OpenAI Responses API)
- [x] Report generator (HTML/Markdown)
- [x] Fine-tuning support (OpenAI API) — data conversion / job management / status monitoring
- [ ] Complete the ZOS-API environment
  - [ ] Lens loading/saving
  - [ ] MTF analysis
  - [ ] Ray tracing
  - [ ] Optimization loop
- [ ] Improve unit test coverage

### v0.3.0-alpha

- [ ] Support more optical software
  - [ ] CODE V integration
  - [ ] ASAP integration
- [ ] Multi-agent collaboration evaluation
- [ ] Dynamic difficulty adjustment

### v0.5.0-beta

- [ ] Release the test dataset v1
- [ ] Performance optimization
- [ ] Improved error handling
- [ ] Complete documentation

### v1.0.0 (goal)

- [ ] Stable API
- [ ] Complete tests
- [ ] Community datasets
- [ ] Official website
- [ ] Public leaderboard

---

## Known Issues

| Issue | Description | Priority | Status |
|-------|-------------|----------|--------|
| pyproject.toml entry point | `pyproject.toml` references `src.main:main` but the actual entry point is `src/llm_pred.py` | Medium | To fix |
| ZOS-API stub | High-level methods of `ZOSAPIEnvironment` return placeholder data | Low | To implement |
| Missing CI/CD | No GitHub Actions workflow | Medium | To create |

---

## Feature TODO

### Core Features

- [ ] **Evaluator extensions**
  - [ ] Multi-metric weight configuration
  - [ ] Confidence interval computation
  - [ ] Statistical significance testing

- [ ] **Runner optimization**
  - [ ] Resumable execution
  - [ ] Incremental evaluation
  - [ ] Distributed execution
  - [ ] Resource limits

### Environment Integrations

- [ ] **Complete ZOS-API**
  - [ ] Tolerance analysis
  - [ ] Multi-wavelength MTF
  - [ ] Non-sequential tracing
  - [ ] CAD export

- [ ] **New environments**
  - [ ] CODE V environment
  - [ ] ASAP environment
  - [ ] Python optics library environment (POPPY, raytracing)
  - [ ] Docker isolated environment

### Datasets

- [ ] **Core datasets**
  - [ ] Lens design dataset v1 (100+ samples)
  - [ ] System analysis dataset v1 (150+ samples)
  - [ ] Tolerance analysis dataset v1 (50+ samples)

- [ ] **Extended datasets**
  - [ ] Multi-band dataset (SWIR, MWIR, LWIR)
  - [ ] Microscope objective dataset
  - [ ] Photographic lens dataset
  - [ ] Laser optics dataset

---

## Documentation TODO

- [x] README (bilingual)
- [x] Contribution guide
- [x] Evaluation theory
- [ ] Optical basics (docs/foundation/optical-basics.md)
- [ ] Agent theory (docs/foundation/agent-theory.md)
- [ ] Evaluation methodology (docs/foundation/evaluation.md)
- [ ] API reference
- [ ] Deployment guide
- [ ] FAQ

---

## Technical TODO

### Code Quality

- [ ] Unit test coverage > 80%
- [ ] Complete type hints
- [ ] Complete docstrings
- [ ] Ruff checks pass

### CI/CD

- [ ] GitHub Actions workflow
  - [ ] Unit tests
  - [ ] Code quality checks
  - [ ] Automatic documentation deployment

### Performance

- [ ] Evaluation throughput > 10 tasks/min
- [ ] Memory footprint optimization
- [ ] Concurrent model optimization

---

## Long-term Vision

### Ecosystem

- [ ] Optis Hub - plugin marketplace
- [ ] Optis Studio - web interface
- [ ] Optis Leaderboard - public leaderboard

### Research Directions

- [ ] Multimodal optics understanding
- [ ] Knowledge graph for optical design
- [ ] Active-learning data annotation
- [ ] Cross-domain knowledge transfer

---

## Development Plan (by priority)

### P0 - Must complete

1. [ ] Complete the ZOS-API integration
2. [ ] Release the core datasets
3. [ ] Complete test coverage

### P1 - Important

4. [ ] CODE V environment
5. [ ] Report generation optimization
6. [ ] Improved error handling
7. [ ] Complete documentation

### P2 - Enhancements

8. [ ] Docker support
9. [ ] Distributed execution
10. [ ] Result visualization

### P3 - Future

11. [ ] Optis Studio
12. [ ] Active learning
13. [ ] Multimodal support

---

## Where to Help

If you would like to contribute, look for issues with the following labels:

| Label | Description | Difficulty |
|-------|-------------|------------|
| `good first issue` | Beginner-friendly starter task | Low |
| `help wanted` | Help needed | Medium |
| `documentation` | Documentation related | Low |
| `testing` | Testing related | Medium |

---

*Last updated: 2026-08-19*
