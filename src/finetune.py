#!/usr/bin/env python3
"""
Optis Benchmark - Fine-tune Job Management Entry Point

CLI tool for managing OpenAI fine-tuning jobs.

Usage:
    # Create a job and wait for completion (--wait)
    python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --wait

    # Create a job only, print job id
    python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml

    # Validate config and data files (no network calls)
    python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --dry-run

    # Check job status
    python src/finetune.py --status ftjob-abc123

    # List recent fine-tuning jobs
    python src/finetune.py --list

    # Export job events log
    python src/finetune.py --events ftjob-abc123 -o results/finetune/events.json

    # Cancel a job
    python src/finetune.py --cancel ftjob-abc123
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.finetune_runner import (  # noqa: E402
    FineTuneJobStatus,
    FineTuneRunner,
    FineTuneRunnerConfig,
)
from src.utils import logger, setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optis Benchmark - Fine-tune Job Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a job and wait for completion
  python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --wait

  # Dry-run: validate config and data files without making API calls
  python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --dry-run

  # List recent fine-tuning jobs
  python src/finetune.py --list

  # Check status of a specific job
  python src/finetune.py --status ftjob-abc123
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--config",
        type=str,
        # default=None,
        default="configs/fine_tuning/ministral_mistral.yaml",
        help="Fine-tune config YAML file path",
    )
    group.add_argument(
        "--status",
        type=str,
        default=None,
        help="Query status of a specific job by job_id",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List recent fine-tuning jobs",
    )
    group.add_argument(
        "--events",
        type=str,
        default=None,
        help="Export events log for a job by job_id",
    )
    group.add_argument(
        "--cancel",
        type=str,
        default=None,
        help="Cancel a running fine-tuning job by job_id",
    )

    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion after creation (--config mode only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path (overrides config: status_output_path for -c mode, "
        "or event file path for --events)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for --list command (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and training files without making API calls",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task Functions
# ---------------------------------------------------------------------------


async def task_create(
    config_path: str,
    wait: bool = False,
    dry_run: bool = False,
    output_override: str | None = None,
) -> int:
    """Create a fine-tuning job.

    Args:
        config_path: Config file path
        wait: Whether to wait for job completion
        dry_run: Validate only, do not execute
        output_override: Override status_output_path

    Returns:
        Exit code
    """
    config = FineTuneRunnerConfig.from_yaml(config_path)
    if output_override:
        config.execution_config["status_output_path"] = output_override

    if dry_run:
        logger.info("Dry-run mode: validating config and data files")
        errors = config.validate()
        if errors:
            for e in errors:
                logger.error(f"  - {e}")
            return 1

        training_file = config.job_config.get("training_file", "")
        if training_file:
            jsonl_errors = FineTuneRunner.validate_jsonl(training_file)
            if jsonl_errors:
                for e in jsonl_errors:
                    logger.error(f"  - {e}")
                return 1
            logger.info(f"Training file OK: {training_file}")

        logger.info(f"Config validated: {config_path}")
        logger.info(f"  base_model : {config.job_config.get('base_model')}")
        logger.info(f"  method     : {config.job_config.get('method')}")
        logger.info(f"  suffix     : {config.job_config.get('suffix')}")
        return 0

    runner = FineTuneRunner(config)
    try:
        await runner.setup()
        status = await runner.create_job()
        logger.info(f"Job created: {status.job_id}")
    
        if wait:
            status = await runner.wait_for_completion(job_id=status.job_id)
            if status.status == "succeeded":
                logger.info("Done. Use the model name below to run benchmark evaluation:")
                logger.info(f"  {status.fine_tuned_model}")
                logger.info("  (set as model.name in configs/llm/GPT_OpenAI.yaml)")
                return 0
            return 1

        logger.info("Use --wait to monitor, or query later via:")
        logger.info(f"  python src/finetune.py --status {status.job_id}")
        return 0

    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1
    finally:
        await runner.teardown()


async def task_query_status(
    config_path: str | None,
    job_id: str,
    output_override: str | None = None,
) -> int:
    """Query job status.

    Args:
        config_path: Optional config file (used for provider credentials)
        job_id: Job ID
        output_override: Unused

    Returns:
        Exit code
    """
    config = await _load_provider_config(config_path)
    runner = FineTuneRunner(config)
    try:
        await runner.setup()
        status = await runner.retrieve_job(job_id)
        _print_status(status)
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1
    finally:
        await runner.teardown()


async def task_list(config_path: str | None, limit: int = 10) -> int:
    """List recent fine-tuning jobs.

    Args:
        config_path: Optional config file (used for provider credentials)
        limit: Maximum number of entries to return

    Returns:
        Exit code
    """
    config = await _load_provider_config(config_path)
    runner = FineTuneRunner(config)
    try:
        await runner.setup()
        await runner.list_jobs(limit=limit)
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1
    finally:
        await runner.teardown()


async def task_cancel(config_path: str | None, job_id: str) -> int:
    """Cancel a job.

    Args:
        config_path: Optional config file (used for provider credentials)
        job_id: Job ID

    Returns:
        Exit code
    """
    config = await _load_provider_config(config_path)
    runner = FineTuneRunner(config)
    try:
        await runner.setup()
        status = await runner.cancel_job(job_id)
        _print_status(status)
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1
    finally:
        await runner.teardown()


async def task_events(
    config_path: str | None,
    job_id: str,
    output_path: str | None,
    limit: int = 100,
) -> int:
    """Export job events log.

    Args:
        config_path: Optional config file (used for provider credentials)
        job_id: Job ID
        output_path: Output file path
        limit: Maximum number of events

    Returns:
        Exit code
    """
    import json as json_mod

    config = await _load_provider_config(config_path)
    runner = FineTuneRunner(config)
    try:
        await runner.setup()
        events = await runner.get_events(job_id, limit=limit)
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json_mod.dump(events, f, ensure_ascii=False, indent=2)
            logger.info(f"Events saved: {output_path} ({len(events)} entries)")
        else:
            for ev in events:
                logger.info(f"  [{ev['level']}] {ev['message']}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1
    finally:
        await runner.teardown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _load_provider_config(config_path: str | None) -> FineTuneRunnerConfig:
    """Load provider credentials from a config file, or fall back to a status file.

    Args:
        config_path: YAML config file path; when None, tries to fall back to the default status file

    Returns:
        FineTuneRunnerConfig (provider_config must be available)
    """
    if config_path:
        return FineTuneRunnerConfig.from_yaml(config_path)

    # Fallback: infer from the status file
    status_path = "results/finetune/job_status.json"
    raise FileNotFoundError(
        f"Provider config not found. Provide -c <config.yaml> or set up GPT_API_KEY. "
        f"Previously saved status: {status_path}"
    )


def _print_status(status: FineTuneJobStatus) -> None:
    """Print job status nicely."""
    logger.info("=" * 60)
    logger.info(f"Job ID          : {status.job_id}")
    logger.info(f"Status          : {status.status}")
    logger.info(f"Base model      : {status.base_model}")
    if status.fine_tuned_model:
        logger.info(f"Fine-tuned model: {status.fine_tuned_model}")
    logger.info(f"Trained tokens  : {status.trained_tokens:,}")
    if status.error:
        logger.info(f"Error           : {status.error}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    setup_logger(level=args.log_level)

    if args.config:
        return await task_create(
            config_path=args.config,
            wait=args.wait,
            dry_run=args.dry_run,
            output_override=args.output,
        )
    if args.status:
        return await task_query_status(args.config, args.status, args.output)
    if args.list:
        return await task_list(args.config, args.limit)
    if args.cancel:
        return await task_cancel(args.config, args.cancel)
    if args.events:
        return await task_events(args.config, args.events, args.output)
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
