# Scripts

Standalone scripts for dataset management, evaluation, report generation, and environment setup.

## Directory Structure

```
scripts/
├── download_data.sh                # Dataset downloader
├── generate_report.py              # HTML/Markdown report generator
├── install_torch.py                # PyTorch CUDA auto-installer
├── optics_paper_extract_eval.py    # Paper extraction evaluation pipeline
├── run_eval.sh                     # Evaluation runner wrapper
└── utils/                          # 12 standalone evaluation metric modules
    ├── bertScore_eval_utils.py
    ├── bleu_eval_utils.py
    ├── cider_eval_utils.py
    ├── citation_eval_utils.py
    ├── edit_distance_utils.py
    ├── em_eval_utils.py
    ├── hungarian_algorithm_utils.py
    ├── jaccard_similarity_utils.py
    ├── meteor_eval_utils.py
    ├── perplexity_eval_utils.py
    ├── rouge_eval_utils.py
    └── sentence_similarity_utils.py
```

---

## install_torch.py

Auto-detect CUDA driver and install the matching PyTorch variant via `uv pip`.

```bash
uv run python scripts/install_torch.py
```

It detects the CUDA version from `nvidia-smi`, maps it to the correct `--torch-backend` (e.g. `cu124`, `cu130`), installs `torch`/`torchvision`/`torchaudio`, and updates `pyproject.toml` so future `uv sync` calls use the same index. Falls back to CPU-only when no GPU is found.

## download_data.sh

Download and extract evaluation datasets from a GitHub release.

```bash
bash scripts/download_data.sh          # Download all datasets
bash scripts/download_data.sh --sample # Download sample dataset only
```

Uses `curl` or `wget`, extracts `.tar.gz` archives into `dataset/processed/`.

## run_eval.sh

Wrapper script that runs Phase 1 (agent evaluation) via `src/main.py`.

```bash
bash scripts/run_eval.sh -a configs/agents/openai/gpt-4.yaml -t lens_design
bash scripts/run_eval.sh -a configs/agents/anthropic/claude-3.yaml -t system_analysis -c 4 -v
```

| Option | Description |
|--------|-------------|
| `-a, --agent <config>` | Agent config YAML (required) |
| `-t, --task <name>` | Task set name (required) |
| `-o, --output <path>` | Output path (default: `results/output.jsonl`) |
| `-c, --concurrency <n>` | Max parallel tasks (default: 1) |
| `--timeout <seconds>` | Per-task timeout (default: 300) |
| `-v, --verbose` | Verbose output |

## generate_report.py

Generate HTML and/or Markdown reports from evaluation result JSON files.

```bash
# Generate HTML report
python scripts/generate_report.py results/eval_results.json

# Generate Markdown report
python scripts/generate_report.py results/eval_results.json --format markdown

# Generate both
python scripts/generate_report.py results/eval_results.json --format both -o results/report.html
```

| Option | Description |
|--------|-------------|
| `results` | Path to results file (JSON or JSONL) |
| `-o, --output <path>` | Output path |
| `--format <format>` | Report format: `html`, `markdown`, or `both` (default: `html`) |

Supports both aggregated JSON (from `eval.py`) and per-task JSONL formats.

## optics_paper_extract_eval.py

Evaluate paper information extraction predictions against gold-standard data using 12 metric modules.

```bash
python scripts/optics_paper_extract_eval.py \
  --pred-file results/predictions.json \
  --gold-file dataset/gold.json \
  --match --rouge --bertScore --bleu --meteor --cider --edit-distance --jaccard --perplexity
```

| Option | Description |
|--------|-------------|
| `--pred-file <path>` | Prediction JSON file (required) |
| `--gold-file <path>` | Gold reference JSON file (required) |
| `--match` | Exact match scoring |
| `--rouge` | ROUGE-L F1 |
| `--bleu` | BLEU n-gram precision |
| `--edit-distance` | Normalized edit similarity |
| `--jaccard` | Jaccard similarity |
| `--bertScore` | BERTScore (semantic similarity) |
| `--perplexity` | Perplexity (GPT-2 based) |
| `--meteor` | METEOR (synonym-aware) |
| `--cider` | CIDEr (TF-IDF n-gram consensus) |
| `--bertScore-model <name>` | BERTScore model (default: `microsoft/deberta-xlarge-mnli`) |

Processes 8 standard paper entry fields (title, authors, abstract, DOI, etc.) with string-level metrics, and multi-sentence fields using Hungarian matching with sentence embeddings.

## utils/ — Evaluation Metric Modules

The `utils/` directory contains 12 self-contained evaluation modules used by `optics_paper_extract_eval.py`. Each module has its own README with detailed documentation:

- **`em_eval_utils.py`** — Exact Match
- **`rouge_eval_utils.py`** — ROUGE-L F1
- **`bleu_eval_utils.py`** — BLEU
- **`meteor_eval_utils.py`** — METEOR (WordNet synonym matching)
- **`cider_eval_utils.py`** — CIDEr (TF-IDF weighted n-gram)
- **`edit_distance_utils.py`** — Levenshtein distance / WER
- **`jaccard_similarity_utils.py`** — Jaccard / Dice / keyword coverage
- **`perplexity_eval_utils.py`** — GPT-2 based perplexity
- **`bertScore_eval_utils.py`** — BERTScore (semantic similarity)
- **`sentence_similarity_utils.py`** — Sentence embeddings + Hungarian matching
- **`hungarian_algorithm_utils.py`** — Optimal assignment matching
- **`citation_eval_utils.py`** — Citation F1 via NLI

See `scripts/utils/README.md` for full documentation.
