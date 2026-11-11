#!/usr/bin/env python3
"""Convert filtered markdown review files into a JSONL dataset.

Each markdown file is split into reviewer sections and written as one JSON
object per line, following the schema of
``dataset/paper_review/data_v1/raw_reviews.json``::

    {"id": 1, "title": "...", "review": [
        {"reviewer_id": 1, "review_text": "...", "version": "v1"}
    ]}

The conversion is lossless: ``review_text`` is the exact markdown section
except that runs of blank lines are collapsed into a single blank line and
leading/trailing whitespace is trimmed. Every file is verified while writing;
any content that does not end up in the JSON, any structural anomaly, and any
author reply detected inside a reviewer section is recorded in the log file.

Usage:
    uv run python utils/review_convert_jsonl.py
    uv run python utils/review_convert_jsonl.py \
        --input-dir dataset/paper_review/filted_reviews_md \
        --output dataset/paper_review/data_v1/reviews_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = REPO_ROOT / "dataset" / "paper_review" / "filted_reviews_md"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "paper_review" / "data_v1" / "reviews_data.jsonl"
DEFAULT_LOG_FILE = REPO_ROOT / "logs" / "review_convert_jsonl.log"

# Handles "### Reviewer #1", "#### Reviewer #2", "# Reviewer #3",
# "### Reviewer #3 (Remarks to the Author)".
REVIEW_HEADING_RE = re.compile(
    r"^\s{0,3}(#{1,6})\s*(.*?reviewer\s*#?\s*(\d+).*)$",
    re.IGNORECASE,
)

# Structural header lines that carry no reviewer content:
# "Reviewer comments:", "!Reviewer comments:", "Reviewer's comments:" ...
HEADER_BOILERPLATE_RE = re.compile(
    r"^\s*[!！]?\s*reviewer'?s?\s+comments?\s*:?\s*$",
    re.IGNORECASE,
)

# Paragraph starts that indicate an author reply inside a reviewer section.
AUTHOR_REPLY_RE = re.compile(
    r"(?im)^\s*(?:we\s+thank\s+the\s+(?:reviewers?|editor)"
    r"|authors?'?\s+responses?"
    r"|response\s+to\s+the\s+reviewers?"
    r"|reply\s+to\s+the\s+reviewers?)\b",
)

logger = logging.getLogger("review_convert_jsonl")


# ---------------------------------------------------------------------------
# CLI / logging
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert filtered markdown reviews into a JSONL dataset.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing markdown review files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Log file path (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the summary to stdout",
    )
    return parser.parse_args()


def setup_logging(log_file: Path) -> None:
    """Configure console + file logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines into one and trim outer whitespace."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_sections(lines: list[str]) -> list[tuple[int, int, int]]:
    """Return (heading_idx, reviewer_id, end_idx) section tuples.

    ``end_idx`` is exclusive: it is the next heading line index, or
    ``len(lines)`` for the last section.
    """
    headings: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        match = REVIEW_HEADING_RE.match(line)
        if match:
            headings.append((idx, int(match.group(3))))

    sections: list[tuple[int, int, int]] = []
    for pos, (idx, reviewer_id) in enumerate(headings):
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        sections.append((idx, reviewer_id, end))
    return sections


def check_header(
    lines: list[str], first_heading: int, file_name: str
) -> list[int]:
    """Log unmapped non-structural header lines; return their line indices."""
    unmapped: list[int] = []
    for idx in range(first_heading):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if HEADER_BOILERPLATE_RE.match(stripped):
            continue
        unmapped.append(idx)
        logger.warning(
            "[%s] line %d: header content not represented in JSON: %r",
            file_name,
            idx + 1,
            stripped[:160],
        )
    return unmapped


def parse_file(path: Path) -> dict[str, Any]:
    """Parse one markdown file into its JSON record plus verification metadata."""
    file_name = path.name
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = split_sections(lines)

    if not sections:
        logger.warning("[%s] no reviewer heading found", file_name)

    unmapped_header = check_header(lines, sections[0][0] if sections else len(lines), file_name)

    reviews: list[dict[str, Any]] = []
    seen_ids: list[int] = []
    empty_sections: list[int] = []
    author_replies: list[tuple[int, str]] = []

    for heading_idx, reviewer_id, end in sections:
        body = collapse_blank_lines("\n".join(lines[heading_idx + 1 : end]))
        seen_ids.append(reviewer_id)

        if not body:
            empty_sections.append(reviewer_id)
            logger.warning(
                "[%s] reviewer #%d has an empty review_text",
                file_name,
                reviewer_id,
            )

        reply = AUTHOR_REPLY_RE.search(body)
        if reply:
            preview = body[reply.start() : reply.start() + 120].replace("\n", " ")
            author_replies.append((reviewer_id, preview))
            logger.warning(
                "[%s] reviewer #%d contains an author reply (kept in full): %r",
                file_name,
                reviewer_id,
                preview,
            )

        reviews.append(
            {
                "reviewer_id": reviewer_id,
                "review_text": body,
                "version": "v1",
            }
        )

    duplicates = sorted({rid for rid in seen_ids if seen_ids.count(rid) > 1})
    structure_warnings = len(duplicates) + len(empty_sections)
    if duplicates:
        logger.warning("[%s] duplicate reviewer ids: %s", file_name, duplicates)
    if seen_ids and sorted(seen_ids) != list(range(1, max(seen_ids) + 1)):
        structure_warnings += 1
        logger.warning("[%s] non-contiguous reviewer ids: %s", file_name, seen_ids)

    logger.info("[%s] reviewers=%d chars=%d", file_name, len(reviews), sum(
        len(r["review_text"]) for r in reviews
    ))

    return {
        "record": {
            "id": 0,  # assigned later in sorted file order
            "title": path.stem,
            "review": reviews,
        },
        "expected": [
            (r["reviewer_id"], r["review_text"]) for r in reviews
        ],
        "line_count": len(lines),
        "header_lines": sections[0][0] if sections else len(lines),
        "unmapped_header": unmapped_header,
        "empty_sections": empty_sections,
        "author_replies": author_replies,
        "structure_warnings": structure_warnings,
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def main() -> int:
    """Convert markdown reviews into JSONL and verify the result."""
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    setup_logging(args.log_file.resolve())

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 0

    md_files = sorted(
        (p for p in input_dir.glob("*.md") if p.is_file()),
        key=lambda p: p.name,
    )
    if not md_files:
        logger.error("No markdown files found in %s", input_dir)
        return 0

    logger.info("Input directory: %s", input_dir)
    logger.info("Markdown files: %d", len(md_files))

    parsed = [parse_file(path) for path in md_files]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name("." + output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for new_id, item in enumerate(parsed, start=1):
            item["record"]["id"] = new_id
            handle.write(json.dumps(item["record"], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, output_path)

    # ------------------------------------------------------------------
    # Verification: reload the file and compare every review_text with the
    # markdown section it came from.
    # ------------------------------------------------------------------
    warnings = 0
    reloaded: list[dict[str, Any]] = []
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                reloaded.append(json.loads(line))

    if len(reloaded) != len(parsed):
        logger.warning(
            "Record count mismatch after write: expected %d, got %d",
            len(parsed),
            len(reloaded),
        )
        warnings += 1

    for expected_item, record in zip(parsed, reloaded):
        expected_pairs = expected_item["expected"]
        actual_pairs = [
            (r["reviewer_id"], r["review_text"]) for r in record.get("review", [])
        ]
        if expected_pairs != actual_pairs:
            logger.warning(
                "[%s] round-trip mismatch: reviewer sections differ after serialization",
                record.get("title"),
            )
            warnings += 1
        if set(record) != {"id", "title", "review"}:
            logger.warning(
                "[%s] unexpected top-level keys: %s",
                record.get("title"),
                sorted(record),
            )
            warnings += 1
        for review in record.get("review", []):
            if set(review) != {"reviewer_id", "review_text", "version"}:
                logger.warning(
                    "[%s] reviewer %s has unexpected keys: %s",
                    record.get("title"),
                    review.get("reviewer_id"),
                    sorted(review),
                )
                warnings += 1

    ids = [rec.get("id") for rec in reloaded]
    if ids != list(range(1, len(reloaded) + 1)):
        logger.warning("Ids are not contiguous 1..%d: %s", len(reloaded), ids[:10])
        warnings += 1

    titles = {rec.get("title") for rec in reloaded}
    expected_titles = {path.stem for path in md_files}
    if titles != expected_titles:
        logger.warning(
            "Title set mismatch: missing=%s extra=%s",
            sorted(expected_titles - titles)[:3],
            sorted(titles - expected_titles)[:3],
        )
        warnings += 1

    total_reviews = sum(len(item["record"]["review"]) for item in parsed)
    log_warnings = sum(
        len(item["unmapped_header"])
        + len(item["empty_sections"])
        + len(item["author_replies"])
        for item in parsed
    )
    structure_warnings = sum(item["structure_warnings"] for item in parsed) + warnings
    summary = (
        f"Wrote {len(reloaded)} records, {total_reviews} reviewer sections "
        f"to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB); "
        f"content warnings={log_warnings}, structural warnings={structure_warnings}"
    )
    logger.info(summary)
    logger.info("Log file: %s", args.log_file.resolve())

    if not args.quiet:
        print("=" * 60)
        print("Review Markdown -> JSONL")
        print("=" * 60)
        print(f"Files:            {len(md_files)}")
        print(f"Reviewer sections:{total_reviews}")
        print(f"Output:           {output_path}")
        print(f"Log:              {args.log_file.resolve()}")
        print(f"Warnings:         {log_warnings} content, {structure_warnings} structural (see log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
