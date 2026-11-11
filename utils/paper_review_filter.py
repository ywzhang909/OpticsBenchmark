#!/usr/bin/env python3
"""
Optis Benchmark - Paper Review Data Filter

Read markdown paper-review files from a folder, send each file together with
the filter prompt (prompts/paper_review/data_filter.txt) to an LLM, parse the
JSON response, and consolidate all results into a single JSONL file.

Each record carries an auto-increment integer id starting at 1 (assigned in
sorted-file order); on resume, already-successful files keep their ids and new
files continue from the highest id used so far.

LLM invocation follows the same Provider + LLM architecture as src/eval.py
(see src/core/llm_runner.py). Supports resuming: already-successful entries in
the output JSONL are skipped on re-run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_runner import LLMRunnerConfig  # noqa: E402
from src.llm import create_llm, create_provider  # noqa: E402
from src.utils import logger, setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optis Benchmark - Paper Review Data Filter (Markdown -> JSONL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter markdown reviews in a folder with default GPT config
  python utils/paper_review_filter.py -i dataset/paper_review/md

  # Custom prompt, config, output and concurrency
  python utils/paper_review_filter.py -i dataset/paper_review/md \\
      -c configs/llm/GPT_OpenAI.yaml -o results/paper_review_filter.jsonl \\
      --concurrency 4 --recursive
        """,
    )

    # Input/Output
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        # required=True,
        default="dataset/paper_review/raw_reviews_md",
        help="Folder containing markdown review files (*.md)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/paper_review_filter.jsonl",
        help="Output JSONL path (default: results/paper_review_filter.jsonl)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search markdown files recursively",
    )

    # Prompt / LLM configuration
    parser.add_argument(
        "-c",
        "--llm-config",
        type=str,
        default="configs/llm/qwen_openai.yaml",
        help="Path to LLM config YAML (default: configs/llm/qwen.yaml)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompts/paper_review/data_filter.txt",
        help="Prompt file path (default: prompts/paper_review/data_filter.txt)",
    )

    # Execution
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of parallel LLM requests (default: 1)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=2,
        help="Retries per file on call/JSON-parse failure (default: 2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of markdown files to process (for testing)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (if not specified, only console logging)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input / prompt loading
# ---------------------------------------------------------------------------


def collect_markdown_files(input_dir: str, recursive: bool) -> list[Path]:
    """Collect *.md files under input_dir as absolute paths, sorted by path."""
    # Resolve so every returned path (and the "location" sent to the LLM) is
    # absolute and comparable with the resolved input root used for resume.
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(root.rglob("*.md") if recursive else root.glob("*.md"))
    files = [f for f in files if f.is_file()]
    logger.info(f"Found {len(files)} markdown files under {input_dir}")
    return files


def file_key(md_file: Path, input_root: Path) -> str:
    """Return md_file path relative to input_root, falling back to its name."""
    try:
        return str(md_file.resolve().relative_to(input_root))
    except ValueError:
        logger.warning(f"File outside input dir, using its name: {md_file}")
        return md_file.name


def load_prompt(prompt_file: str) -> str:
    """Load prompt file.

    Consistent with LLMPredRunner._load_prompt: skip the first two lines
    (comment line + blank line).
    """
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    content = path.read_text(encoding="utf-8")

    lines = content.split("\n")
    if len(lines) > 2:
        content = "\n".join(lines[2:])
    else:
        logger.warning(
            f"Prompt file '{prompt_file}' has only {len(lines)} lines, "
            f"expected at least 3. Using raw content."
        )

    return content


# ---------------------------------------------------------------------------
# Output / resume
# ---------------------------------------------------------------------------


def load_done(output_path: Path) -> tuple[dict[str, int], int]:
    """Read existing output JSONL and return (done_map, max_id).

    done_map maps each successful record's "file" (path relative to the input
    directory) to its auto-increment id, so re-runs keep the ids already used
    and number new files from max_id + 1.

    Failed entries are dropped from the file (rewritten with successes only)
    so that re-runs retry them without duplicating ids.
    """
    if not output_path.exists():
        return {}, 0

    kept: list[dict[str, Any]] = []
    dropped = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            if record.get("error") is None and record.get("data") is not None:
                # Only records with a numeric id and a known file path can be
                # reused; legacy records (e.g. string ids) are re-queued so
                # every entry ends up with a unique auto-increment id.
                file_key = record.get("file")
                raw_id = record.get("id")
                try:
                    record["id"] = int(raw_id)
                except (TypeError, ValueError):
                    record["id"] = None
                if file_key is None or record["id"] is None:
                    logger.warning(
                        f"Resume: record '{raw_id}' has no numeric id or file "
                        f"path, re-processing it"
                    )
                    dropped += 1
                    continue
                kept.append(record)
            else:
                dropped += 1

    if dropped:
        with open(output_path, "w", encoding="utf-8") as f:
            for record in kept:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Resume: kept {len(kept)} successful records, "
                    f"re-queueing {dropped} failed records")

    done_map = {str(r["file"]): int(r["id"]) for r in kept}
    return done_map, max(done_map.values(), default=0)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse LLM response into a dict, stripping markdown code fences if any."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


async def process_one(
    md_file: Path,
    file_id: int,
    input_root: Path,
    prompt: str,
    llm: Any,
    provider: Any,
    setup: dict[str, Any],
    retry: int,
    semaphore: asyncio.Semaphore,
    writer: asyncio.Lock,
    out_path: Path,
) -> dict[str, Any]:
    """Run the filter prompt for a single markdown file and append the result."""
    label = f"#{file_id} {md_file.name}"
    start_time = time.time()
    messages = [
        {"location": str(md_file)},
        {"prompt": prompt},
    ]

    raw = ""
    data: dict[str, Any] | None = None
    error: str | None = None
    result: dict[str, Any] = {}

    async with semaphore:
        for attempt in range(1, retry + 2):
            try:
                result = await llm.chat(
                    messages=messages,
                    provider=provider,
                    setup=setup,
                )
                raw = result.get("content", "")
                data = parse_json_response(raw)
                error = None
                break
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"[{label}] attempt {attempt}/{retry + 1} failed: {error}"
                )
                if attempt <= retry:
                    await asyncio.sleep(2 * attempt)

    record = {
        "id": file_id,
        "file": file_key(md_file, input_root),
        "model": getattr(llm, "model_name", ""),
        "latency": round(time.time() - start_time, 2),
        "cost": result.get("cost", 0.0),
        "data": data,
        "error": error,
    }
    if error:
        record["raw_response"] = raw

    async with writer:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if error:
        logger.error(f"  {label} FAILED after {retry + 1} attempts")
    else:
        logger.info(
            f"  {label} completed ({record['latency']:.2f}s, "
            f"cost ${record['cost']:.4f})"
        )
    return record


async def run_filter(args: argparse.Namespace) -> int:
    """Filter markdown review files through the LLM and write JSONL results."""
    logger.info("=" * 60)
    logger.info("Paper Review Data Filter (Markdown -> JSONL)")
    logger.info("=" * 60)
    logger.info("+-- Configuration " + "-" * 42 + "+")
    logger.info(f"|  Input Dir:   {args.input_dir}")
    logger.info(f"|  Prompt:      {args.prompt_file}")
    logger.info(f"|  LLM Config:  {args.llm_config}")
    logger.info(f"|  Output:      {args.output}")
    logger.info(f"|  Concurrency: {args.concurrency}")
    logger.info("+--" + "-" * 57 + "+")

    # Collect inputs and prompt
    md_files = collect_markdown_files(args.input_dir, args.recursive)
    if not md_files:
        logger.error("No markdown files found")
        return 1

    if args.max_samples is not None:
        md_files = md_files[: args.max_samples]
        logger.info(f"Limited to {len(md_files)} files (--max-samples)")

    prompt = load_prompt(args.prompt_file)
    logger.info(f"Prompt loaded: {len(prompt)} chars")

    # Resume support: skip files already completed successfully and keep their
    # ids; new files are numbered from max_id + 1 (first run starts at 1)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input_dir).resolve()

    done_map, max_id = load_done(out_path)
    if done_map:
        logger.info(f"Resume: {len(done_map)} records already done, skipping them")
        md_files = [f for f in md_files if file_key(f, input_root) not in done_map]
        logger.info(f"{len(md_files)} files remaining to process")
        if not md_files:
            logger.info("Nothing to do")
            return 0

    tasks: list[tuple[Path, int]] = [
        (md_file, max_id + i) for i, md_file in enumerate(md_files, start=1)
    ]
    logger.info(f"Auto-increment ids: {tasks[0][1]} .. {tasks[-1][1]}")

    # Create LLM stack from config (same as LLMPredRunner.setup)
    config = LLMRunnerConfig.from_yaml(args.llm_config)
    provider = create_provider(config.provider_config)
    llm = create_llm(config.model_config)
    setup = config.setup_config
    logger.info(f"Provider: {type(provider).__name__}")
    logger.info(f"LLM: {llm.model_name}")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    writer = asyncio.Lock()

    eval_start = time.time()
    try:
        progress = tqdm(tasks, desc="  filter", unit="file", leave=True)

        async def run_one(md_file: Path, file_id: int) -> dict[str, Any]:
            try:
                record = await process_one(
                    md_file=md_file,
                    file_id=file_id,
                    input_root=input_root,
                    prompt=prompt,
                    llm=llm,
                    provider=provider,
                    setup=setup,
                    retry=args.retry,
                    semaphore=semaphore,
                    writer=writer,
                    out_path=out_path,
                )
            except Exception as e:
                # A single bad file must not abort the whole batch; record the
                # failure so a re-run can retry it.
                logger.error(f"#{file_id} {md_file.name} crashed: {e}")
                record = {
                    "id": file_id,
                    "file": file_key(md_file, input_root),
                    "model": getattr(llm, "model_name", ""),
                    "latency": 0.0,
                    "cost": 0.0,
                    "data": None,
                    "error": f"{type(e).__name__}: {e}",
                }
                async with writer:
                    try:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception as write_err:
                        logger.error(
                            f"#{file_id} failed to write error record: {write_err}"
                        )
            finally:
                progress.update(1)
                progress.set_postfix({"done": progress.n})
            return record

        records = await asyncio.gather(*(run_one(f, i) for f, i in tasks))
    finally:
        try:
            await provider.close()
        except Exception:
            pass

    total_time = time.time() - eval_start
    succeeded = [r for r in records if r["error"] is None]
    failed = [r for r in records if r["error"] is not None]

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FILTER COMPLETE")
    logger.info("=" * 60)
    separator = "+" + "-" * 11 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 12 + "+"
    logger.info(separator)
    logger.info(f"| {'Processed':>9} | {'OK':>6} | {'Fail':>6} | {'Time':>10} |")
    logger.info(separator)
    logger.info(
        f"| {len(records):>9} | {len(succeeded):>6} | {len(failed):>6} "
        f"| {total_time:>9.1f}s |"
    )
    logger.info(separator)

    if failed:
        logger.warning("Failed files:")
        for r in failed:
            logger.warning(f"  - #{r['id']} {r['file']}: {r['error']}")
        logger.warning("Re-run the same command to retry failed files.")

    logger.info(f"Results saved to: {out_path}")
    logger.info("")

    return 0 if not failed else 1


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    setup_logger(
        level=args.log_level,
        log_file=args.log_file,
        console=True,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    return await run_filter(args)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        exit_code = asyncio.run(main_async(args))
        return exit_code
    except KeyboardInterrupt:
        logger.warning("Filter interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
