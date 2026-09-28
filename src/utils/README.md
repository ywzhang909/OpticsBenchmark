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
| `eval_consistency.py` | 一致性评估命令行入口（调用 `src/algorithm/consistency.py`，并交叉校验 `judge_model` / `output_model` / `context` 元数据） |
| `eval_bias.py` | 偏见评估命令行入口（调用 `src/algorithm/bias.py`，评估单个裁判对自身输出的偏好；读取单文件 `n × m` 网格，裁判名取自记录的 `judge_model`，可用 `--judge-model` 覆盖） |
| `score_io.py` | 评分记录读取与解析（`load_scores` / `extract_scores` / `records_by_id` / `records_by_key` / `summarize_records` / `cross_check_metadata`） |
