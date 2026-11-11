#!/usr/bin/env python3
"""
Optis Benchmark - Fine-tune Dataset Builder

Converts the benchmark dataset gold answer + prompt into the JSONL format required by OpenAI fine-tuning.

Emits one training sample per line:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "<gold answer JSON>"}]}

Position in the workflow (stage A, an explicit step):
    This script produces train.jsonl / val.jsonl
        ↓ the files are the contract
    the training_file / validation_file in configs/fine_tuning/*.yaml point at the above
        ↓
    python src/finetune.py -c <config> [--wait]

Examples:
    # paper_info_extract task: use the title as the user message
    python utils/build_finetune_dataset.py \\
        -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \\
        -p prompts/paper_info_extract/zero-shot_v1.0.txt \\
        -d dataset/paper_info_extract/dataset_json/dataset_v1.json \\
        -o results/finetune/train.jsonl --val-ratio 0.2

    # Use the full text extracted from the PDF as the user message (--user-field location)
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

# Per-sample token limit warning threshold (gpt-4o-mini caps a training sample at ~65k input tokens)
MAX_SAMPLE_TOKENS_WARNING = 60000

# Minimum number of training samples officially recommended
MIN_RECOMMENDED_SAMPLES = 50

# PyPDF2 is a soft dependency; when missing, degrade to using the title text only
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
    """Load the prompt file content as the system message.

    Consistent with LLMPredRunner._load_prompt: skip the first two lines (comment line + blank line).

    Args:
        path: Path to the prompt file

    Returns:
        The system message content; an empty string plus a warning if the file does not exist
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
    """Extract text from a PDF (truncated to max_chars).

    Args:
        pdf_path: Path to the PDF file
        max_chars: Maximum number of characters

    Returns:
        The extracted text; None plus a warning when unavailable
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
    """Build the user message content.

    If the value of the given field is an existing path to a PDF file, extract its text;
    otherwise use the field value directly as the text.

    Args:
        record: A dataset record (e.g. {"title": ..., "location": ...})
        user_field: Name of the field to read the value from
        max_chars: Maximum number of characters for the text

    Returns:
        The user message text; an empty string if the field is missing
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
    """Match gold answers to dataset records by title (case-insensitive).

    Args:
        gold_records: List of gold answer records [{id, data: {...}}]
        dataset_records: List of dataset records [{title, location, ...}], or None
        gold_key: Key holding the payload in the gold answer (default "data")

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
    """Build the list of samples in the OpenAI fine-tuning format.

    Args:
        pairs: The output of match_dataset_records()
        system_prompt: The system message content (may be empty)
        gold_key: Key holding the gold answer payload
        user_field: Name of the field holding the user content in a dataset record
        max_chars: Maximum number of characters for the user content

    Returns:
        The list of samples [{"messages": [...]}], with invalid samples filtered out
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
    """Validate the message structure of a single sample.

    Args:
        sample: The sample to validate

    Returns:
        A list of errors; an empty list means the sample is valid
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
    """Get the tiktoken encoder, falling back to o200k_base for unknown models."""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(encoding_model)
    except KeyError:
        logger.warning(f"Unknown model '{encoding_model}' for tiktoken, fallback to o200k_base")
        return tiktoken.get_encoding("o200k_base")


def count_tokens(sample: dict[str, Any], encoding: Any) -> int:
    """Count the total tokens of a single sample (including message structure overhead)."""
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
    """Randomly split the dataset into train and validation sets with a fixed seed.

    Args:
        samples: All samples
        val_ratio: Validation ratio [0, 1)
        seed: Random seed (for reproducibility)

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
    """Save the samples to a JSONL file.

    Args:
        samples: The list of samples
        path: Output path

    Returns:
        The number of samples written
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
    """Parse the command-line arguments."""
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
    """Main entry point: load -> match -> build -> validate -> split -> save."""
    args = parse_args()

    try:
        # Load the gold answer
        gold_path = Path(args.gold)
        if not gold_path.exists():
            logger.error(f"Gold answer file not found: {args.gold}")
            return 1
        with open(gold_path, encoding="utf-8") as f:
            gold_records = json.load(f)
        if not isinstance(gold_records, list) or not gold_records:
            logger.error(f"Expected non-empty JSON array in gold file: {args.gold}")
            return 1

        # Load the dataset records (optional)
        dataset_records = None
        if args.dataset:
            dataset_path = Path(args.dataset)
            if not dataset_path.exists():
                logger.error(f"Dataset file not found: {args.dataset}")
                return 1
            with open(dataset_path, encoding="utf-8") as f:
                dataset_records = json.load(f)

        # Build the prompt
        system_prompt = load_prompt(args.prompt) if args.prompt else ""

        # Match and build the samples
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

        # Structural validation
        invalid = [(i, errs) for i, s in enumerate(samples) if (errs := validate_sample(s))]
        if invalid:
            for idx, errs in invalid:
                logger.error(f"Sample[{idx}] validation failed: {'; '.join(errs)}")
            return 1

        # Token statistics and warnings
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

        # Split and save
        train_samples, val_samples = split_train_val(samples, args.val_ratio, args.seed)

        val_output = args.val_output or str(Path(args.output).with_suffix("")) + "_val.jsonl"
        save_jsonl(train_samples, args.output)
        if val_samples:
            save_jsonl(val_samples, val_output)

        # Summary report
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
