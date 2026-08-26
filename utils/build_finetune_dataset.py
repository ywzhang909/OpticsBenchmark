#!/usr/bin/env python3
"""
Optis Benchmark - Fine-tune Dataset Builder

将基准数据集的 gold answer + prompt 转换为 OpenAI 微调所需的 JSONL 格式。

每行输出一个训练样本:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "<gold answer JSON>"}]}

工作流位置（阶段 A，显式步骤）：
    本脚本产出 train.jsonl / val.jsonl
        ↓ 文件即契约
    configs/fine_tuning/*.yaml 的 training_file / validation_file 指向上述文件
        ↓
    python src/finetune.py -c <config> [--wait]

Examples:
    # paper_info_extract 任务：以标题作为 user 消息
    python utils/build_finetune_dataset.py \\
        -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \\
        -p prompts/paper_info_extract/zero-shot_v1.0.txt \\
        -d dataset/paper_info_extract/dataset_json/dataset_v1.json \\
        -o results/finetune/train.jsonl --val-ratio 0.2

    # 使用 PDF 提取全文作为 user 消息（--user-field location）
    python utils/build_finetune_dataset.py \\
        -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \\
        -p prompts/paper_info_extract/zero-shot_v1.0.txt \\
        -d dataset/paper_info_extract/dataset_json/dataset_v1.json \\
        --user-field location -o results/finetune/train_fulltext.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 单样本 token 上限告警阈值（gpt-4o-mini 训练样本上限约 65k input tokens）
MAX_SAMPLE_TOKENS_WARNING = 60000

# 官方建议的最少训练样本数
MIN_RECOMMENDED_SAMPLES = 50

# PyPDF2 为软依赖，缺失时降级为仅使用标题文本
try:
    from PyPDF2 import PdfReader

    _PDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    _PDF_AVAILABLE = False


# =============================================================================
# Data Loading
# =============================================================================


def load_prompt(path: str | Path) -> str:
    """加载 prompt 文件内容作为 system 消息。

    与 LLMPredRunner._load_prompt 保持一致：跳过前两行（注释行 + 空行）。

    Args:
        path: prompt 文件路径

    Returns:
        system 消息内容；文件不存在时返回空字符串并记录警告
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Prompt file not found: {path}")
        return ""

    content = path_obj.read_text(encoding="utf-8")
    lines = content.split("\n")
    if len(lines) > 2:
        return "\n".join(lines[2:])
    logger.warning(f"Prompt file '{path}' has only {len(lines)} lines, using raw content.")
    return content


def extract_pdf_text(pdf_path: str | Path, max_chars: int) -> str | None:
    """从 PDF 提取文本（截断到 max_chars）。

    Args:
        pdf_path: PDF 文件路径
        max_chars: 最大字符数

    Returns:
        提取的文本；不可用时返回 None 并记录警告
    """
    if not _PDF_AVAILABLE:
        logger.warning("PyPDF2 not installed, cannot extract PDF text. Fallback to title only.")
        return None

    path_obj = Path(pdf_path)
    if not path_obj.exists():
        logger.warning(f"PDF file not found: {pdf_path}")
        return None

    try:
        reader = PdfReader(str(path_obj))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text[:max_chars] if text.strip() else None
    except Exception as e:
        logger.warning(f"PDF extraction failed for '{path_obj.name}': {e}")
        return None


def build_user_content(
    record: dict[str, Any],
    user_field: str,
    max_chars: int,
) -> str:
    """构建 user 消息内容。

    若指定字段的值为已存在的文件路径且为 PDF，则提取文本；
    否则直接使用字段值作为文本。

    Args:
        record: 数据集记录（如 {"title": ..., "location": ...}）
        user_field: 取值字段名
        max_chars: 文本最大字符数

    Returns:
        user 消息文本；字段缺失时返回空字符串
    """
    value = record.get(user_field, "")
    if not value:
        return ""

    value_str = str(value)
    path_obj = Path(value_str)
    if path_obj.suffix.lower() == ".pdf":
        text = extract_pdf_text(path_obj, max_chars)
        return text if text else record.get("title", "")

    return value_str[:max_chars]


def match_dataset_records(
    gold_records: list[dict[str, Any]],
    dataset_records: list[dict[str, Any]] | None,
    gold_key: str,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """按标题（大小写不敏感）匹配 gold answer 与数据集记录。

    Args:
        gold_records: gold answer 记录列表 [{id, data: {...}}]
        dataset_records: 数据集记录列表 [{title, location, ...}]，可为 None
        gold_key: gold answer 中载荷所在的键名（默认 "data"）

    Returns:
        [(gold_record, matched_dataset_record_or_None), ...]
    """
    title_to_record: dict[str, dict[str, Any]] = {}
    if dataset_records:
        for rec in dataset_records:
            title = str(rec.get("title", "")).strip().lower()
            if title:
                title_to_record[title] = rec

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for gold in gold_records:
        payload = gold.get(gold_key, gold)
        if not isinstance(payload, dict):
            logger.warning(
                f"Skip invalid gold record id={gold.get('id', '?')}: '{gold_key}' is not a dict"
            )
            continue

        matched = None
        if title_to_record:
            title = str(payload.get("title", "")).strip().lower()
            matched = title_to_record.get(title)
            if matched is None:
                logger.warning(
                    f"No dataset record matched title '{payload.get('title', '')}', "
                    f"fallback to title-only user content"
                )

        pairs.append((gold, matched))
    return pairs


# =============================================================================
# Sample Building & Validation
# =============================================================================


def build_samples(
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]],
    system_prompt: str,
    gold_key: str,
    user_field: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    """构建 OpenAI 微调格式样本列表。

    Args:
        pairs: match_dataset_records() 的输出
        system_prompt: system 消息内容（可为空）
        gold_key: gold answer 载荷键名
        user_field: 数据集记录中 user 内容的字段名
        max_chars: user 内容最大字符数

    Returns:
        样本列表 [{"messages": [...]}]，已过滤无效样本
    """
    samples: list[dict[str, Any]] = []
    skipped = 0

    for gold, matched in pairs:
        payload = gold.get(gold_key, gold)

        if matched is not None:
            user_content = build_user_content(matched, user_field, max_chars)
        else:
            user_content = str(payload.get("title", ""))
        if isinstance(user_content, str):
            user_content = user_content.strip()

        assistant_content = json.dumps(payload, ensure_ascii=False)

        if not user_content or not assistant_content:
            skipped += 1
            continue

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

        samples.append({"messages": messages})

    if skipped:
        logger.warning(f"Skipped {skipped} invalid samples (empty user/assistant content)")
    return samples


def validate_sample(sample: dict[str, Any]) -> list[str]:
    """校验单个样本的消息结构。

    Args:
        sample: 待校验样本

    Returns:
        错误列表；空列表表示合法
    """
    errors: list[str] = []
    messages = sample.get("messages", [])

    if len(messages) < 2:
        errors.append("messages must contain at least user + assistant")
        return errors

    expected_roles = ["user", "assistant"]
    actual_roles = [m.get("role", "") for m in messages]
    if messages[0].get("role") == "system":
        actual_roles = actual_roles[1:]

    if actual_roles != expected_roles:
        errors.append(f"invalid role sequence: {actual_roles}, expected {expected_roles}")

    for i, m in enumerate(messages):
        if not m.get("content"):
            errors.append(f"message[{i}] ({m.get('role')}) has empty content")

    return errors


def get_encoding(encoding_model: str) -> Any:
    """获取 tiktoken 编码器，未知模型时回退到 o200k_base。"""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(encoding_model)
    except KeyError:
        logger.warning(f"Unknown model '{encoding_model}' for tiktoken, fallback to o200k_base")
        return tiktoken.get_encoding("o200k_base")


def count_tokens(sample: dict[str, Any], encoding: Any) -> int:
    """统计单样本总 token 数（含消息结构开销）。"""
    total = 0
    for m in sample["messages"]:
        total += len(encoding.encode(m["content"])) + 4
    total += 3
    return total


# =============================================================================
# Split & Save
# =============================================================================


def split_train_val(
    samples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按固定种子随机切分训练集与验证集。

    Args:
        samples: 全部样本
        val_ratio: 验证集比例 [0, 1)
        seed: 随机种子（保证可复现）

    Returns:
        (train_samples, val_samples)
    """
    if val_ratio <= 0 or len(samples) < 2:
        return list(samples), []

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) // 2)

    return shuffled[val_count:], shuffled[:val_count]


def save_jsonl(samples: list[dict[str, Any]], path: str | Path) -> int:
    """保存样本到 JSONL 文件。

    Args:
        samples: 样本列表
        path: 输出路径

    Returns:
        写入的样本数
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(samples)} samples: {path}")
    return len(samples)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Build OpenAI fine-tuning JSONL dataset from benchmark gold answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Title-only user content (fast, no PDF parsing)
  python utils/build_finetune_dataset.py \\
      -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \\
      -p prompts/paper_info_extract/zero-shot_v1.0.txt \\
      -o results/finetune/train.jsonl

  # Full-text user content extracted from PDFs (--user-field location)
  python utils/build_finetune_dataset.py \\
      -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \\
      -p prompts/paper_info_extract/zero-shot_v1.0.txt \\
      -d dataset/paper_info_extract/dataset_json/dataset_v1.json \\
      --user-field location -o results/finetune/train_fulltext.jsonl
        """,
    )

    parser.add_argument(
        "-g",
        "--gold",
        type=str,
        required=True,
        help="Gold answer JSON file path (array of {id, data})",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Prompt template file path (used as system message)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Dataset JSON file path ({title, location, ...}), matched by title",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/finetune/train.jsonl",
        help="Output train JSONL path (default: results/finetune/train.jsonl)",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default=None,
        help="Output validation JSONL path (default: <output>_val.jsonl)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio in [0, 1) (default: 0.2, 0 disables split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--gold-key",
        type=str,
        default="data",
        help="Key holding answer payload in gold records (default: data)",
    )
    parser.add_argument(
        "--user-field",
        type=str,
        default="title",
        help="Dataset record field used as user message (default: title)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100000,
        help="Max characters of user content (default: 100000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to build (default: all)",
    )
    parser.add_argument(
        "--encoding-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for tiktoken token counting (default: gpt-4o-mini)",
    )

    return parser.parse_args()


def main() -> int:
    """主入口：加载 → 匹配 → 构建 → 校验 → 切分 → 保存。"""
    args = parse_args()

    try:
        # 加载 gold answer
        gold_path = Path(args.gold)
        if not gold_path.exists():
            logger.error(f"Gold answer file not found: {args.gold}")
            return 1
        with open(gold_path, encoding="utf-8") as f:
            gold_records = json.load(f)
        if not isinstance(gold_records, list) or not gold_records:
            logger.error(f"Expected non-empty JSON array in gold file: {args.gold}")
            return 1

        # 加载数据集记录（可选）
        dataset_records = None
        if args.dataset:
            dataset_path = Path(args.dataset)
            if not dataset_path.exists():
                logger.error(f"Dataset file not found: {args.dataset}")
                return 1
            with open(dataset_path, encoding="utf-8") as f:
                dataset_records = json.load(f)

        # 构建 prompt
        system_prompt = load_prompt(args.prompt) if args.prompt else ""

        # 匹配并构建样本
        pairs = match_dataset_records(gold_records, dataset_records, args.gold_key)
        samples = build_samples(
            pairs=pairs,
            system_prompt=system_prompt,
            gold_key=args.gold_key,
            user_field=args.user_field,
            max_chars=args.max_chars,
        )

        if args.max_samples is not None:
            samples = samples[: args.max_samples]

        if not samples:
            logger.error("No valid samples built")
            return 1

        # 结构校验
        invalid = [(i, errs) for i, s in enumerate(samples) if (errs := validate_sample(s))]
        if invalid:
            for idx, errs in invalid:
                logger.error(f"Sample[{idx}] validation failed: {'; '.join(errs)}")
            return 1

        # token 统计与告警
        encoding = get_encoding(args.encoding_model)
        token_counts = [count_tokens(s, encoding) for s in samples]
        oversized = sum(1 for t in token_counts if t > MAX_SAMPLE_TOKENS_WARNING)
        if oversized:
            logger.warning(
                f"{oversized}/{len(samples)} samples exceed {MAX_SAMPLE_TOKENS_WARNING} tokens"
            )
        if len(samples) < MIN_RECOMMENDED_SAMPLES:
            logger.warning(
                f"Only {len(samples)} samples; OpenAI recommends >= "
                f"{MIN_RECOMMENDED_SAMPLES} for meaningful fine-tuning"
            )

        # 切分并保存
        train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)

        val_output = args.val_output or str(Path(args.output).with_suffix("")) + "_val.jsonl"
        save_jsonl(train_samples, args.output)
        if val_samples:
            save_jsonl(val_samples, val_output)

        # 统计报告
        avg_tokens = sum(token_counts) / len(token_counts)
        logger.info("=" * 60)
        logger.info("Fine-tune dataset built successfully")
        logger.info(f"Total samples : {len(samples)}")
        logger.info(f"Train         : {len(train_samples)} -> {args.output}")
        if val_samples:
            logger.info(f"Validation    : {len(val_samples)} -> {val_output}")
        logger.info(f"Tokens/sample : avg {avg_tokens:.0f}, max {max(token_counts)}")
        logger.info(f"Total tokens  : ~{sum(token_counts):,} (training cost scales with this)")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Failed to build fine-tune dataset: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
