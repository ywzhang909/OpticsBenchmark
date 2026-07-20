#!/usr/bin/env python3
"""
OptiS Benchmark - LLM Prediction Entry Point

Inference entry point based on Provider + LLM architecture.
Reads LLM config, loads dataset, calls LLM for inference, and saves results.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_runner import LLMPredRunner, LLMRunnerConfig  # noqa: E402
from src.utils import logger, setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OptiS Benchmark - LLM Prediction (Provider + LLM Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with config file
  python src/llm_pred.py -c configs/llm/qwen_openai.yaml

  # Override output path
  python src/llm_pred.py -c configs/llm/qwen_openai.yaml -o results/my_outputs.jsonl

  # Limit sample count
  python src/llm_pred.py -c configs/llm/qwen_openai.yaml -n 10

  # Dry run (show config only)
  python src/llm_pred.py -c configs/llm/qwen_openai.yaml --dry-run
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        # required=True,
        default="configs/llm/qwen_openai.yaml",
        help="LLM config file path (YAML)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/llm_pred.jsonl",
        help="Output file path (overrides output_path in config)",
    )
    parser.add_argument(
        "-n",
        "--max-samples",
        type=int,
        help="Max samples (overrides max_samples in config)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Concurrency (overrides concurrency in config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config only, do not run inference",
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
# Main Function
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    setup_logger(level=args.log_level)

    # Load config
    try:
        config = LLMRunnerConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # CLI overrides
    if args.output:
        config.output_path = args.output
    if args.max_samples is not None:
        config.task_config["max_samples"] = args.max_samples
    if args.concurrency is not None:
        config.max_concurrency = args.concurrency

    # Dry run
    if args.dry_run:
        logger.info("Dry Run mode - Config info:")
        logger.info(f"  Provider: {config.provider_config.get('type', 'N/A')}")
        logger.info(f"  Model: {config.model_config.get('name', 'N/A')}")
        logger.info(f"  Dataset: {config.task_config.get('dataset_path', 'N/A')}")
        logger.info(f"  Output: {config.output_path}")
        logger.info(f"  Concurrency: {config.max_concurrency}")
        return 0

    # Run inference
    try:
        runner = LLMPredRunner(config)
        outputs = await runner.run()
        LLMPredRunner.save_outputs(outputs, config.output_path)

        # Statistics
        success_count = sum(1 for o in outputs if not o.error)
        total_cost = sum(o.cost for o in outputs)
        avg_latency = (
            sum(o.latency for o in outputs) / len(outputs) if outputs else 0
        )

        logger.info("Inference completed")
        logger.info(f"Success: {success_count}/{len(outputs)}")
        logger.info(f"Total cost: ${total_cost:.4f}, Avg latency: {avg_latency:.2f}s")

        return 0

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        exit_code = asyncio.run(main_async(args))
        return exit_code
    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
