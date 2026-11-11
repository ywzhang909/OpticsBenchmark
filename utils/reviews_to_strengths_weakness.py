#!/usr/bin/env python3
"""Extract paper strengths, weaknesses and disputes from reviewer comments.

Every record combines three inputs in one request (per
``prompts/paper_review/strengths_weakness.txt``):

1. the prompt ``prompts/paper_review/strengths_weakness.txt``;
2. the reviewer comments built from ``dataset/paper_review/data_v1/reviews_data.jsonl``
   (the ``review`` array of each record, rendered as ``## Reviewer #N`` sections;
   this JSONL is the only review source);
3. the paper text ``dataset/paper_review/filted_papers/<title>.md`` and its figures
   ``dataset/paper_review/filted_papers/<title>_images/`` (all uploaded in one request).

The results are written to
``dataset/paper_review/data_v1/strengths-weaknesses_data.jsonl``.

Each output line follows::

    {"id": 1, "title": "...", "strengths-weaknesses": {
        "strengths": [{"content": "...", "quotes": ["Reviewer #1: \"...\""]}],
        "weaknesses": [{"content": "...", "quotes": ["Reviewer #2: \"...\""]}],
        "disputes": [{"topic": "...", "positions": [
            {"reviewer": "Reviewer #1", "stance": "positive", "viewpoint": "..."}}]}]}}

The prompt demands strict JSON with verbatim field names, so the parser only
accepts a JSON object (tolerating code fences and a nested/double-encoded
wrapper) and then validates the structure: every point needs a non-empty
``content`` and a non-empty ``quotes`` array whose elements start with
``Reviewer #N:`` naming a reviewer of the record; dispute positions need a
known reviewer, a stance of ``positive`` or ``negative``, and a non-empty
``viewpoint``. Empty arrays are
allowed; both strengths and weaknesses empty is treated as a failure.
Failed records are logged and retried on the next run; successful records are
skipped. ``--manifest`` records what was actually sent per record.

Usage:
    uv run python utils/reviews_to_strengths_weakness.py
    uv run python utils/reviews_to_strengths_weakness.py --max-samples 2
    uv run python utils/reviews_to_strengths_weakness.py --no-images   # text-only A/B
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
from src.utils.general import IMAGE_MIME_TYPES  # noqa: E402
from src.utils.md_image_filter import IMAGE_LINK_RE  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = REPO_ROOT / "dataset" / "paper_review" / "data_v1" / "reviews_data.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "paper_review" / "data_v1" / "strengths-weaknesses_data.jsonl"
DEFAULT_PROMPT = REPO_ROOT / "prompts" / "paper_review" / "strengths_weakness.txt"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "llm" / "qwen_openai.yaml"
DEFAULT_PAPERS_MD = REPO_ROOT / "dataset" / "paper_review" / "filted_papers"
DEFAULT_MANIFEST = REPO_ROOT / "logs" / "strengths_weakness_manifest.jsonl"
DEFAULT_LOG = REPO_ROOT / "logs" / "reviews_to_strengths_weakness.log"

RESULT_KEY = "strengths-weaknesses"
STRENGTHS = "strengths"
WEAKNESSES = "weaknesses"
DISPUTES = "disputes"
RESULT_KEYS = (STRENGTHS, WEAKNESSES, DISPUTES)
STANCES = ("positive", "negative")

# "Reviewer #12: \"...\"" prefix required on every quote; "Reviewer #12" as the
# whole string for a dispute position's reviewer (per the prompt OUTPUT schema).
QUOTE_RE = re.compile(r"^Reviewer\s+#(\d+)\s*:", re.IGNORECASE)
REVIEWER_RE = re.compile(r"^\s*Reviewer\s+#(\d+)\s*$", re.IGNORECASE)

FENCE_RE = re.compile(r"^```(?:json|markdown)?\s*(.*?)\s*```$", re.DOTALL)
# Marker writes page/footnote anchors as <span id="page-2-0"> ... </span>.
SPAN_TAG_RE = re.compile(r"</?span[^>]*>")
BLANK_RUN_RE = re.compile(r"\n{3,}")

REVIEW_COMMENTS_HEADING = "<review_comments>"
MANUSCRIPT_HEADING = "<paper_manuscript>"
FIGURES_HEADING = "<paper_figures>"
FIGURES_INTRO = (
    "The figures of the manuscript are attached below, in the same order as the "
    "``[figure: ...]`` placeholders of the manuscript text:"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract strengths/weaknesses/disputes from reviewer comments (JSONL -> JSONL).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input reviews JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-c",
        "--llm-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"LLM config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT,
        help=f"Strengths/weaknesses prompt (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=DEFAULT_PAPERS_MD,
        help=f"Directory of paper Markdown and <title>_images/ folders (default: {DEFAULT_PAPERS_MD})",
    )
    parser.add_argument(
        "--no-manuscript",
        action="store_true",
        help="Do not send the paper text (reviewer comments only)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not send the paper figures",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum figures per record, 0 sends all of them (default: 0)",
    )
    parser.add_argument(
        "--max-image-mb",
        type=float,
        default=8.0,
        help="Skip figures larger than this many MB (default: 8)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"JSONL audit trail of what was sent per record (default: {DEFAULT_MANIFEST})",
    )
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
        help="Retries per record on API/parse failure (default: 2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of records to process (for testing)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Log file path (default: {DEFAULT_LOG})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_prompt(prompt_file: Path) -> str:
    """Load the prompt, skipping the first two lines (title + blank line)."""
    if not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    content = prompt_file.read_text(encoding="utf-8")
    lines = content.split("\n")
    if len(lines) > 2:
        return "\n".join(lines[2:])
    logger.warning(f"Prompt file {prompt_file} has only {len(lines)} lines, using raw content")
    return content


def strip_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence, if present."""
    stripped = text.strip()
    match = FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def build_review_text(record: dict[str, Any]) -> tuple[str, list[int]]:
    """Render a record's reviewer comments as ``## Reviewer #N`` sections.

    Returns the review text and the ordered list of reviewer ids; entries with
    a missing ``reviewer_id`` fall back to their position in the array.
    """
    blocks: list[str] = []
    reviewer_ids: list[int] = []
    for position, review in enumerate(record.get("review", []), start=1):
        raw_id = review.get("reviewer_id", position)
        try:
            reviewer_id = int(raw_id)
        except (TypeError, ValueError):
            reviewer_id = position
        text = str(review.get("review_text", "")).strip()
        reviewer_ids.append(reviewer_id)
        blocks.append(f"## Reviewer #{reviewer_id}\n{text}")
    return "\n\n".join(blocks), reviewer_ids


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------


def _extract_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Pick the three canonical fields from a response object.

    Top-level keys are matched case-insensitively. When none is present the
    payload is searched one level deeper (a nested object or a double-encoded
    JSON string), which some models emit even under ``response_format``.
    """
    lowered = {str(key).strip().lower(): value for key, value in payload.items()}
    if any(name in lowered for name in RESULT_KEYS):
        return lowered

    for value in payload.values():
        nested: Any = value
        if isinstance(value, str):
            try:
                nested = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(nested, dict):
            try:
                return _extract_fields(nested)
            except ValueError:
                continue

    raise ValueError(f"Response JSON has none of the required fields {list(RESULT_KEYS)}: keys={list(payload)}")


def _validate_point(entry: Any, key: str, index: int, reviewer_ids: list[int] | None) -> dict[str, Any]:
    """Validate one strengths/weaknesses entry."""
    if not isinstance(entry, dict):
        raise ValueError(f"{key}[{index}] is not an object: {entry!r}")
    lowered = {str(name).strip().lower(): value for name, value in entry.items()}
    content = str(lowered.get("content", "")).strip()
    if not content:
        raise ValueError(f"{key}[{index}] has an empty content")
    quotes = lowered.get("quotes")
    if not isinstance(quotes, list) or not quotes:
        raise ValueError(f"{key}[{index}] quotes must be a non-empty array")
    cleaned_quotes: list[str] = []
    for quote in quotes:
        text = str(quote).strip()
        match = QUOTE_RE.match(text)
        if not match:
            raise ValueError(f"{key}[{index}] quote lacks the 'Reviewer #N:' prefix: {text[:80]!r}")
        if reviewer_ids is not None:
            quote_reviewer = int(match.group(1))
            if quote_reviewer not in reviewer_ids:
                raise ValueError(
                    f"{key}[{index}] quote references unknown reviewer #{quote_reviewer} "
                    f"(record has {reviewer_ids})"
                )
        cleaned_quotes.append(text)
    return {"content": content, "quotes": cleaned_quotes}


def _validate_dispute(entry: Any, index: int, reviewer_ids: list[int] | None) -> dict[str, Any]:
    """Validate one dispute entry."""
    if not isinstance(entry, dict):
        raise ValueError(f"{DISPUTES}[{index}] is not an object: {entry!r}")
    lowered = {str(name).strip().lower(): value for name, value in entry.items()}
    topic = str(lowered.get("topic", "")).strip()
    if not topic:
        raise ValueError(f"{DISPUTES}[{index}] has an empty topic")
    positions = lowered.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"{DISPUTES}[{index}] positions must be a non-empty array")

    cleaned_positions: list[dict[str, str]] = []
    for pos_index, position in enumerate(positions):
        if not isinstance(position, dict):
            raise ValueError(f"{DISPUTES}[{index}].positions[{pos_index}] is not an object")
        fields = {str(name).strip().lower(): value for name, value in position.items()}
        reviewer = str(fields.get("reviewer", "")).strip()
        match = REVIEWER_RE.match(reviewer)
        if not match:
            raise ValueError(
                f"{DISPUTES}[{index}].positions[{pos_index}] reviewer is not 'Reviewer #N': {reviewer!r}"
            )
        reviewer_id = int(match.group(1))
        if reviewer_ids is not None and reviewer_id not in reviewer_ids:
            raise ValueError(
                f"{DISPUTES}[{index}].positions[{pos_index}] references unknown reviewer #{reviewer_id} "
                f"(record has {reviewer_ids})"
            )
        stance = str(fields.get("stance", "")).strip().lower()
        if stance not in STANCES:
            raise ValueError(
                f"{DISPUTES}[{index}].positions[{pos_index}] stance must be positive/negative: {stance!r}"
            )
        viewpoint = str(fields.get("viewpoint", "")).strip()
        if not viewpoint:
            raise ValueError(f"{DISPUTES}[{index}].positions[{pos_index}] has an empty viewpoint")
        cleaned_positions.append({"reviewer": f"Reviewer #{reviewer_id}", "stance": stance, "viewpoint": viewpoint})

    return {"topic": topic, "positions": cleaned_positions}


def validate_analysis(fields: dict[str, Any], reviewer_ids: list[int] | None = None) -> dict[str, Any]:
    """Validate and normalize the three result fields against the prompt schema."""
    analysis: dict[str, Any] = {}
    for key in (STRENGTHS, WEAKNESSES):
        value = fields.get(key)
        if value is None or not isinstance(value, list):
            raise ValueError(f"Missing or non-array field: {key}")
        analysis[key] = [_validate_point(entry, key, index, reviewer_ids) for index, entry in enumerate(value)]
    disputes = fields.get(DISPUTES)
    if disputes is None or not isinstance(disputes, list):
        raise ValueError(f"Missing or non-array field: {DISPUTES}")
    analysis[DISPUTES] = [_validate_dispute(entry, index, reviewer_ids) for index, entry in enumerate(disputes)]
    if not analysis[STRENGTHS] and not analysis[WEAKNESSES]:
        raise ValueError("Both strengths and weaknesses are empty")
    return analysis


def parse_response(raw: str, reviewer_ids: list[int] | None = None) -> dict[str, Any]:
    """Parse an LLM response into the validated strengths/weaknesses schema.

    Only JSON objects are accepted (the prompt forbids Markdown wrappers and
    the default LLM config forces ``response_format=json_object``); code fences
    and one level of nested/double-encoded wrapping are tolerated.
    """
    text = strip_fences(raw)
    if not text:
        raise ValueError("Empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Response JSON is not an object: {type(payload).__name__}")
    return validate_analysis(_extract_fields(payload), reviewer_ids)


# ---------------------------------------------------------------------------
# Input assembly (prompt + reviews + manuscript + figures)
# ---------------------------------------------------------------------------


def figures_dir_for(paper_md_path: Path) -> Path:
    """Return the image folder that belongs to a converted paper."""
    return paper_md_path.with_name(f"{paper_md_path.stem}_images")


def find_figures(paper_md_path: Path) -> list[Path]:
    """Return the figures of a paper in document order.

    Order follows the ``![](...)`` links inside the converted Markdown so every
    image keeps its place in the text; papers without links fall back to the
    file names of ``<title>_images/``.
    """
    if not paper_md_path.is_file():
        return []
    try:
        text = paper_md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Cannot read manuscript {paper_md_path}: {exc}")
        return []

    figures_dir = figures_dir_for(paper_md_path)
    figures: list[Path] = []
    seen: set[Path] = set()
    for target in IMAGE_LINK_RE.findall(text):
        target = target.strip().strip("<>")
        candidate = paper_md_path.parent / target
        if not candidate.is_file():
            candidate = figures_dir / Path(target).name
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            figures.append(candidate)
    if figures:
        return figures

    if figures_dir.is_dir():
        return sorted(
            path
            for path in figures_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_MIME_TYPES
        )
    return []


def clean_manuscript(text: str) -> str:
    """Make converted manuscript Markdown readable for the LLM.

    Image links become ``[figure: <file name>]`` placeholders so they line up
    with the uploaded figures, Marker's ``<span ...>`` anchors are dropped and
    blank runs are collapsed.
    """
    def _placeholder(match: re.Match[str]) -> str:
        target = match.group(1).strip().strip("<>")
        return f"[figure: {Path(target).name}]"

    cleaned = IMAGE_LINK_RE.sub(_placeholder, text)
    cleaned = SPAN_TAG_RE.sub("", cleaned)
    return BLANK_RUN_RE.sub("\n\n", cleaned).strip()


def select_figures(
    figures: list[Path],
    max_images: int,
    max_image_mb: float,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Split figures into the ones to upload and the ones to skip with a reason."""
    kept: list[Path] = []
    skipped: list[dict[str, str]] = []
    limit_bytes = int(max_image_mb * 1024 * 1024)
    for path in figures:
        try:
            size = path.stat().st_size
        except OSError:
            skipped.append({"file": path.name, "reason": "unreadable"})
            continue
        if size > limit_bytes:
            skipped.append({"file": path.name, "reason": f"larger than {max_image_mb:g} MB"})
        elif max_images and len(kept) >= max_images:
            skipped.append({"file": path.name, "reason": f"beyond --max-images={max_images}"})
        else:
            kept.append(path)
    return kept, skipped


def build_messages(
    prompt: str,
    reviews_text: str,
    manuscript_text: str | None,
    figures: list[Path],
) -> list[dict[str, Any]]:
    """Assemble the input parts into LLM messages, tagged per the prompt.

    The LLM layer merges them into a single user turn, keeping this order:
    prompt, ``<review_comments>``, ``<paper_manuscript>``, ``<paper_figures>``
    listing and finally the images.
    """
    messages: list[dict[str, Any]] = [
        {"prompt": prompt},
        {"prompt": f"{REVIEW_COMMENTS_HEADING}\n{reviews_text}\n</review_comments>"},
    ]
    if manuscript_text:
        messages.append({"prompt": f"{MANUSCRIPT_HEADING}\n{manuscript_text}\n</paper_manuscript>"})
    if figures:
        listing = "\n".join(f"- {path.name}" for path in figures)
        messages.append(
            {
                "prompt": f"{FIGURES_HEADING}\n{FIGURES_INTRO}\n{listing}\n</paper_figures>",
                "images": [str(path) for path in figures],
            }
        )
    return messages


def prepare_inputs(record: dict[str, Any], args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    """Collect the LLM input of one record and describe what was sent."""
    record_id = record.get("id")
    title = record.get("title", "")
    reviews_text, reviewer_ids = build_review_text(record)

    manuscript_text: str | None = None
    manuscript_path: Path | None = None
    figures: list[Path] = []
    skipped: list[dict[str, str]] = []

    if args.no_manuscript and args.no_images:
        logger.info(f"#{record_id} manuscript and figures disabled, text-only review input")
    else:
        candidate = args.papers_dir.resolve() / f"{title}.md"
        if candidate.is_file():
            manuscript_path = candidate
            try:
                raw = candidate.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(f"#{record_id} cannot read manuscript {candidate}: {exc}")
                raw = ""
            figures = find_figures(candidate)
            if not args.no_manuscript and raw:
                manuscript_text = clean_manuscript(raw)
            if args.no_images:
                skipped.extend({"file": path.name, "reason": "--no-images"} for path in figures)
                figures = []
            else:
                figures, skipped = select_figures(figures, args.max_images, args.max_image_mb)
            if not figures and not args.no_images:
                logger.warning(f"#{record_id} no figure uploaded ({title[:70]})")
        elif not args.no_manuscript:
            logger.warning(f"#{record_id} manuscript md not found: {candidate}")

    return {
        "messages": build_messages(
            prompt=prompt,
            reviews_text=reviews_text,
            manuscript_text=manuscript_text,
            figures=figures,
        ),
        "reviewer_ids": reviewer_ids,
        "manuscript": str(manuscript_path) if manuscript_path else None,
        "manuscript_chars": len(manuscript_text) if manuscript_text else 0,
        "figures": [path.name for path in figures],
        "skipped_images": skipped,
    }


# ---------------------------------------------------------------------------
# Output bookkeeping
# ---------------------------------------------------------------------------


def record_is_done(record: Any) -> bool:
    """Return True when a record already holds a validated analysis."""
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("id"), int):
        return False
    analysis = record.get(RESULT_KEY)
    if not isinstance(analysis, dict):
        return False
    try:
        validate_analysis(analysis)
    except ValueError:
        return False
    return True


def load_done(output_path: Path) -> dict[int, dict[str, Any]]:
    """Load successful records from an existing output file, keyed by id."""
    done: dict[int, dict[str, Any]] = {}
    if not output_path.is_file():
        return done

    with output_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Ignoring malformed output line {line_no}")
                continue
            if record_is_done(record):
                done[record["id"]] = record
    return done


def write_records(output_path: Path, records: dict[int, dict[str, Any]]) -> None:
    """Atomically rewrite the output file with records sorted by id."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name("." + output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record_id in sorted(records):
            handle.write(json.dumps(records[record_id], ensure_ascii=False) + "\n")
        handle.flush()
    tmp_path.replace(output_path)


async def append_manifest(
    manifest_path: Path | None,
    entry: dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    """Append one audit entry describing what was sent for a record."""
    if manifest_path is None:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    async with lock:
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


async def generate_one(
    record: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    llm: Any,
    provider: Any,
    setup: dict[str, Any],
    semaphore: asyncio.Semaphore,
    writer: asyncio.Lock,
    output_path: Path,
    done: dict[int, dict[str, Any]],
    manifest_lock: asyncio.Lock,
) -> bool:
    """Call the LLM for one record and append its result on success."""
    record_id = record["id"]
    title = record.get("title", "")
    prepared = prepare_inputs(record, args, prompt)
    messages = prepared["messages"]
    reviewer_ids = prepared["reviewer_ids"]
    figures_sent = len(prepared["figures"])

    async with semaphore:
        error: str | None = None
        for attempt in range(1, args.retry + 2):
            start = time.time()
            try:
                result = await llm.chat(messages=messages, provider=provider, setup=setup)
                if not result.get("content"):
                    raise RuntimeError(result.get("error") or "empty LLM response")
                analysis = parse_response(result.get("content", ""), reviewer_ids)
                output = {"id": record_id, "title": title, RESULT_KEY: analysis}
                async with writer:
                    done[record_id] = output
                    with output_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                usage = result.get("usage") or {}
                elapsed = time.time() - start
                await append_manifest(
                    args.manifest,
                    {
                        "id": record_id,
                        "title": title,
                        "status": "ok",
                        "counts": {key: len(analysis[key]) for key in RESULT_KEYS},
                        **{k: v for k, v in prepared.items() if k != "messages"},
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "seconds": round(elapsed, 1),
                    },
                    manifest_lock,
                )
                logger.info(
                    f"#{record_id} {title[:70]} completed ({elapsed:.2f}s, "
                    f"{len(analysis[STRENGTHS])} strengths, {len(analysis[WEAKNESSES])} weaknesses, "
                    f"{len(analysis[DISPUTES])} disputes, {figures_sent} figure(s), "
                    f"{usage.get('prompt_tokens', '?')} prompt tokens, "
                    f"cost ${result.get('cost', 0.0):.4f})"
                )
                return True
            except Exception as exc:  # noqa: BLE001 - retry on any failure
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    f"#{record_id} {title[:70]} attempt {attempt}/{args.retry + 1} "
                    f"failed: {error}"
                )
                if attempt <= args.retry:
                    await asyncio.sleep(2 * attempt)

    logger.error(f"#{record_id} {title[:70]} FAILED after {args.retry + 1} attempts: {error}")
    await append_manifest(
        args.manifest,
        {
            "id": record_id,
            "title": title,
            "status": "failed",
            "error": error,
            **{k: v for k, v in prepared.items() if k != "messages"},
        },
        manifest_lock,
    )
    return False


async def run(args: argparse.Namespace) -> None:
    """Process all pending records."""
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    args.papers_dir = args.papers_dir.resolve()
    args.manifest = args.manifest.resolve() if args.manifest else None

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not args.papers_dir.is_dir() and not (args.no_manuscript and args.no_images):
        logger.warning(f"Papers directory not found: {args.papers_dir} (papers will be skipped)")

    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    done = load_done(output_path)
    if done:
        # Compact duplicates left behind by an interrupted run.
        write_records(output_path, done)
        logger.info(f"Resume: {len(done)} successful records already done")

    pending = [record for record in records if record.get("id") not in done]
    if args.max_samples is not None:
        pending = pending[: args.max_samples]
    logger.info(f"Records: {len(records)} total, {len(pending)} pending")
    if not pending:
        logger.info("Nothing to do")
        return

    prompt = load_prompt(args.prompt_file.resolve())
    logger.info(f"Prompt loaded: {len(prompt)} chars")
    logger.info(
        f"Input: reviews={input_path} papers={args.papers_dir} "
        f"manuscript={'off' if args.no_manuscript else 'on'} "
        f"figures={'off' if args.no_images else 'on'}"
    )

    config = LLMRunnerConfig.from_yaml(args.llm_config.resolve())
    provider = create_provider(config.provider_config)
    llm = create_llm(config.model_config)
    setup = config.setup_config
    logger.info(f"Provider: {type(provider).__name__} | LLM: {llm.model_name}")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    writer = asyncio.Lock()
    manifest_lock = asyncio.Lock()
    failures = 0

    try:
        progress = tqdm(pending, desc="strengths-weaknesses", unit="record", leave=True)

        async def worker(record: dict[str, Any]) -> bool:
            ok = await generate_one(
                record=record,
                prompt=prompt,
                args=args,
                llm=llm,
                provider=provider,
                setup=setup,
                semaphore=semaphore,
                writer=writer,
                output_path=output_path,
                done=done,
                manifest_lock=manifest_lock,
            )
            progress.update(1)
            return ok

        results = await asyncio.gather(*(worker(record) for record in pending))
        failures = sum(1 for ok in results if not ok)
    finally:
        try:
            await provider.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass

    # Final pass: collapse any duplicates and order by id.
    write_records(output_path, done)

    logger.info("=" * 60)
    logger.info("STRENGTHS/WEAKNESSES COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed this run: {len(pending)}")
    logger.info(f"Succeeded this run: {len(pending) - failures}")
    logger.info(f"Failed this run:   {failures}")
    logger.info(f"Total saved:       {len(done)}")
    logger.info(f"Output:            {output_path}")
    if args.manifest:
        logger.info(f"Manifest:          {args.manifest}")
    logger.info("Re-run the same command to retry failed records.")
    logger.info("=" * 60)


def main() -> int:
    """Main entry point; always returns 0, failures are logged."""
    args = parse_args()
    log_file = args.log_file.resolve()
    setup_logger(
        level=args.log_level,
        log_file=str(log_file),
        console=True,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as exc:  # noqa: BLE001 - always exit 0 per workflow
        logger.error(f"Fatal error: {exc}")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
