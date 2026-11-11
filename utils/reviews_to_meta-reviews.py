#!/usr/bin/env python3
"""Generate meta-reviews from reviewer comments with an LLM.

Every record combines four inputs in one request:

1. the prompt ``prompts/paper_review/meta-review.txt``;
2. the reviewer comments ``dataset/paper_review/filted_reviews_md/<title>.md``;
3. the paper text ``dataset/paper_review/filted_papers/<title>.md``;
4. the figures of that paper ``dataset/paper_review/filted_papers/<title>_images/``
   (one paper owns several images, all of them are uploaded in the same request).

``dataset/paper_review/data_v1/reviews_data.jsonl`` provides the record list
(``id`` + ``title``) and is the fallback source for reviewer comments. The
resulting meta-reviews are written to
``dataset/paper_review/data_v1/meta-reviews_data.jsonl``.

Each output line follows::

    {"id": 1, "title": "...", "meta-review": {
        "Summary of the paper": "...",
        "Core disputes and arbitration": "...",
        "Strengths of the paper": ["...", "..."],
        "Weaknesses of the paper": ["...", "..."]}}

The model is asked to answer with Markdown sections (per the prompt). Since the
default LLM config forces JSON output, the parser accepts both a JSON object and
Markdown headings. The strengths and weaknesses values are stored as lists with
one bullet point per element (a JSON array or one ``-``/``*``/numbered bullet per
line; continuation lines are merged into the preceding bullet).
Failed records are logged and retried on the next run;
successful records are skipped. ``--manifest`` records what was actually sent
(manuscript, kept and skipped figures) per record.

Usage:
    uv run python utils/reviews_to_meta-reviews.py
    uv run python utils/reviews_to_meta-reviews.py --max-samples 2
    uv run python utils/reviews_to_meta-reviews.py --no-images   # text-only A/B
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
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "paper_review" / "data_v1" / "meta-reviews_data.jsonl"
DEFAULT_PROMPT = REPO_ROOT / "prompts" / "paper_review" / "meta-review.txt"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "llm" / "qwen_openai.yaml"
DEFAULT_REVIEWS_MD = REPO_ROOT / "dataset" / "paper_review" / "filted_reviews_md"
DEFAULT_PAPERS_MD = REPO_ROOT / "dataset" / "paper_review" / "filted_papers"
DEFAULT_MANIFEST = REPO_ROOT / "logs" / "meta_reviews_manifest.jsonl"
DEFAULT_LOG = REPO_ROOT / "logs" / "reviews_to_meta_reviews.log"

SUMMARY = "Summary of the paper"
DISPUTES = "Core disputes and arbitration"
STRENGTHS = "Strengths of the paper"
WEAKNESSES = "Weaknesses of the paper"

META_FIELDS = (SUMMARY, DISPUTES, STRENGTHS, WEAKNESSES)
# Fields whose value is stored as a list of bullet points (one per element).
LIST_FIELDS = (STRENGTHS, WEAKNESSES)
REQUIRED_FIELDS = (SUMMARY, STRENGTHS, WEAKNESSES)

FENCE_RE = re.compile(r"^```(?:json|markdown)?\s*(.*?)\s*```$", re.DOTALL)
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.*?)\s*$")
# Bullet markers used for strengths/weaknesses: "-", "*", "•" or "1." / "1)" / "1、".
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)、])\s+(.+)")
# Marker writes page/footnote anchors as <span id="page-2-0"> ... </span>.
SPAN_TAG_RE = re.compile(r"</?span[^>]*>")
BLANK_RUN_RE = re.compile(r"\n{3,}")

FIGURES_HEADING = "## Figures of the manuscript"
FIGURES_INTRO = (
    "The figures of the manuscript are attached below, in the same order as the "
    "``[figure: ...]`` placeholders of the manuscript text:\n"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate meta-reviews from reviewer comments (JSONL -> JSONL).",
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
        help=f"Output meta-reviews JSONL (default: {DEFAULT_OUTPUT})",
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
        help=f"Meta-review prompt (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        default=DEFAULT_REVIEWS_MD,
        help=f"Directory of reviewer comment Markdown (default: {DEFAULT_REVIEWS_MD})",
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


def normalize_key(value: str) -> str:
    """Normalize a heading/key for tolerant matching."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


CANONICAL_KEYS = {normalize_key(name): name for name in META_FIELDS}


def canonical_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a response object's keys onto the canonical meta-review fields."""
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = CANONICAL_KEYS.get(normalize_key(str(key)))
        if canonical:
            mapped[canonical] = value
        elif normalize_key(str(key)) == normalize_key("meta-review"):
            if isinstance(value, dict):
                mapped.update(canonical_mapping(value))
    return mapped


def strip_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence, if present."""
    stripped = text.strip()
    match = FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def heading_title(line: str) -> str | None:
    """Return a heading title for Markdown ATX, bold, or numbered headings."""
    match = HEADING_RE.match(line)
    if match:
        raw = match.group(2)
    else:
        bold = re.match(r"^\s*\*\*(.+?)\*\*\s*:?\s*$", line)
        if not bold:
            return None
        raw = bold.group(1)
    title = raw.strip().strip("*").strip().rstrip(":").strip()
    title = re.sub(r"^\d+[.)、]?\s*", "", title)
    return title or None


def parse_markdown_sections(text: str) -> dict[str, str]:
    """Extract the four meta-review sections from Markdown headings."""
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        title = heading_title(line)
        if title is None:
            continue
        canonical = CANONICAL_KEYS.get(normalize_key(title))
        if canonical:
            headings.append((index, canonical))

    if not headings:
        raise ValueError("No meta-review headings found in response")

    preamble = "\n".join(lines[: headings[0][0]]).strip()
    if preamble:
        logger.warning(
            f"Response contains {len(preamble)} chars before the first meta-review "
            f"heading: {preamble[:160]!r}"
        )

    sections: dict[str, str] = {}
    for pos, (index, canonical) in enumerate(headings):
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        sections[canonical] = "\n".join(lines[index + 1 : end]).strip()
    return sections


def to_bulleted_list(value: Any) -> list[str]:
    """Convert a strengths/weaknesses payload into a list of bullet points.

    Accepts a JSON array (each element becomes one point) or a Markdown block
    where every point starts with a ``-``/``*``/``•`` or numbered bullet;
    continuation lines are appended to the preceding point. A non-empty text
    without any bullet marker is kept as a single-element list.
    """
    if isinstance(value, (list, tuple)):
        points: list[str] = []
        for item in value:
            points.extend(to_bulleted_list(item))
        return points

    text = str(value).strip()
    if not text:
        return []

    points: list[str] = []
    for line in text.splitlines():
        match = BULLET_RE.match(line)
        if match:
            points.append(match.group(1).strip())
        elif points and line.strip():
            points[-1] = f"{points[-1]} {line.strip()}"
        elif line.strip():
            points.append(line.strip())
    return [point for point in points if point] or [text]


def parse_response(raw: str) -> dict[str, Any]:
    """Parse an LLM response into the canonical meta-review fields.

    Accepts either a JSON object (forced by response_format) or Markdown
    sections as requested by the prompt. Strengths and weaknesses are returned
    as lists of bullet points; the other fields stay plain strings.
    """
    text = strip_fences(raw)
    if not text:
        raise ValueError("Empty response")

    sections: dict[str, Any] = {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        mapped = canonical_mapping(payload)
        for field, value in mapped.items():
            if field in LIST_FIELDS and isinstance(value, (list, tuple)):
                sections[field] = [str(item) for item in value]
            elif isinstance(value, str):
                sections[field] = value.strip()
            elif isinstance(value, list):
                sections[field] = "\n".join(str(item) for item in value).strip()
        # Some models wrap the whole Markdown answer in a single JSON string.
        if not any(sections.get(field) for field in REQUIRED_FIELDS):
            for value in payload.values():
                if isinstance(value, str) and "#" in value:
                    try:
                        sections.update(parse_markdown_sections(value))
                        break
                    except ValueError:
                        continue

    if not sections:
        sections = parse_markdown_sections(text)

    result: dict[str, Any] = {}
    for field in META_FIELDS:
        value = sections.get(field, "")
        if field in LIST_FIELDS:
            result[field] = to_bulleted_list(value)
        elif isinstance(value, (list, tuple)):
            result[field] = "\n".join(str(item) for item in value).strip()
        else:
            result[field] = str(value).strip()
    missing = [field for field in REQUIRED_FIELDS if not result[field]]
    if missing:
        raise ValueError(f"Missing or empty meta-review sections: {missing}")
    return result


def build_review_block(record: dict[str, Any]) -> str:
    """Render a record's reviewer comments as a labelled input block."""
    blocks: list[str] = []
    for review in record.get("review", []):
        text = str(review.get("review_text", "")).strip()
        blocks.append(f"## Reviewer #{review.get('reviewer_id', '?')}\n{text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Input assembly (prompt + reviews + manuscript + figures)
# ---------------------------------------------------------------------------


def load_review_text(record: dict[str, Any], reviews_dir: Path) -> tuple[str, str]:
    """Return ``(reviewer comments, source)`` for one record.

    Prefers ``<reviews_dir>/<title>.md``; falls back to the ``review`` array of
    the JSONL record when that file is missing.
    """
    record_id = record.get("id")
    title = record.get("title", "")
    md_path = reviews_dir / f"{title}.md"
    if md_path.is_file():
        try:
            return md_path.read_text(encoding="utf-8").strip(), str(md_path)
        except OSError as exc:
            logger.warning(f"#{record_id} cannot read reviews md {md_path}: {exc}")
    else:
        logger.warning(f"#{record_id} reviews md not found under {reviews_dir}, using JSONL text")
    return build_review_block(record), "<reviews jsonl>"


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
    """Assemble the four input parts into LLM messages.

    The LLM layer merges them into a single user turn, keeping this order:
    prompt, reviewer comments, manuscript, figure list and finally the images.
    """
    messages: list[dict[str, Any]] = [
        {"prompt": prompt},
        {"prompt": f"## Reviewer comments\n\n{reviews_text}"},
    ]
    if manuscript_text:
        messages.append({"prompt": f"## Original manuscript\n\n{manuscript_text}"})
    if figures:
        listing = "\n".join(f"- {path.name}" for path in figures)
        messages.append(
            {
                "prompt": f"{FIGURES_HEADING}\n\n{FIGURES_INTRO}\n{listing}",
                "images": [str(path) for path in figures],
            }
        )
    return messages


def prepare_inputs(record: dict[str, Any], args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    """Collect the LLM input of one record and describe what was sent."""
    record_id = record.get("id")
    title = record.get("title", "")
    reviews_text, reviews_source = load_review_text(record, args.reviews_dir.resolve())

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
        "reviews_source": reviews_source,
        "manuscript": str(manuscript_path) if manuscript_path else None,
        "manuscript_chars": len(manuscript_text) if manuscript_text else 0,
        "figures": [path.name for path in figures],
        "skipped_images": skipped,
    }


# ---------------------------------------------------------------------------
# Output bookkeeping
# ---------------------------------------------------------------------------


def record_is_done(record: Any) -> bool:
    """Return True when a record already holds a complete meta-review.

    Matches the list format: ``Summary`` is a non-empty string and
    ``Strengths``/``Weaknesses`` are non-empty lists of non-empty points;
    records in the old all-strings format are not considered done.
    """
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("id"), int):
        return False
    meta = record.get("meta-review")
    if not isinstance(meta, dict):
        return False
    if not isinstance(meta.get(SUMMARY), str) or not meta.get(SUMMARY, "").strip():
        return False
    for field in LIST_FIELDS:
        points = meta.get(field)
        if not isinstance(points, list) or not points:
            return False
        if not all(isinstance(point, str) and point.strip() for point in points):
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
                meta = record["meta-review"]
                meta.setdefault(DISPUTES, "")
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
    """Call the LLM for one record and append its meta-review on success."""
    record_id = record["id"]
    title = record.get("title", "")
    prepared = prepare_inputs(record, args, prompt)
    messages = prepared["messages"]
    figures_sent = len(prepared["figures"])

    async with semaphore:
        error: str | None = None
        for attempt in range(1, args.retry + 2):
            start = time.time()
            try:
                result = await llm.chat(messages=messages, provider=provider, setup=setup)
                if not result.get("content"):
                    raise RuntimeError(result.get("error") or "empty LLM response")
                meta = parse_response(result.get("content", ""))
                output = {"id": record_id, "title": title, "meta-review": meta}
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
                        **{k: v for k, v in prepared.items() if k != "messages"},
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "seconds": round(elapsed, 1),
                    },
                    manifest_lock,
                )
                logger.info(
                    f"#{record_id} {title[:70]} completed ({elapsed:.2f}s, "
                    f"{figures_sent} figure(s), "
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
    args.reviews_dir = args.reviews_dir.resolve()
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
        f"Input dirs: reviews={args.reviews_dir} papers={args.papers_dir} "
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
        progress = tqdm(pending, desc="meta-review", unit="record", leave=True)

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
    logger.info("META-REVIEW COMPLETE")
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
