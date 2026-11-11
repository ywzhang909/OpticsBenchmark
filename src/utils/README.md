# src/utils — Infrastructure Utilities

## Purpose

Cross-cutting infrastructure shared by all entry points.

## Files

| File | Description |
|------|-------------|
| `logger.py` | Logging setup (console + optional rotating file) |
| `parser.py` | Argument/env parsing helpers (`replace_env`) |
| `general.py` | General-purpose helpers |
| `generate_report.py` | Markdown/HTML evaluation report generation |
| `eval_consistency.py` | Command-line entry point for consistency evaluation (calls `src/algorithm/consistency.py` and cross-checks the `judge_model` / `output_model` / `context` metadata) |
| `eval_bias.py` | Command-line entry point for bias evaluation (calls `src/algorithm/bias.py`; measures a single judge's preference for its own output; reads a single-file `n × m` grid, taking the judge name from each record's `judge_model`, overridable with `--judge-model`) |
| `score_io.py` | Score record reading and parsing (`load_scores` / `extract_scores` / `records_by_id` / `records_by_key` / `summarize_records` / `cross_check_metadata`) |
| `pdf_convert_md_marker.py` | Batch PDF → Markdown conversion via `marker-pdf` (CLI: `python -m src.utils.pdf_convert_md_marker --input-dir DIR`); loads models once per run, saves marker's extracted images into a per-document `<md name>_images/` folder (rewriting the Markdown links) unless `--no-images` is passed, supports `--recursive` / `--skip-existing` / `--overwrite` / `--log-file` |
| `md_image_filter.py` | Prunes non-content images (journal logos, license badges, icons) out of a converted Markdown tree (CLI: `python -m src.utils.md_image_filter --dir DIR [--dry-run]`); keeps an image when its pixel area reaches `--min-area` or a figure number sits next to it, and only lets publisher boilerplate veto an image that neither signal claims on its own; drops the Markdown link line too and archives removed files in `<md name>_images_removed/`, with a JSONL `--report` |
