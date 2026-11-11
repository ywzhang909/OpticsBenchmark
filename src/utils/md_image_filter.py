#!/usr/bin/env python3
"""
Optis Benchmark - Markdown Image Filter

Prunes the images that are *not* real document figures out of a Markdown tree
produced by :mod:`src.utils.pdf_convert_md_marker` (Markdown files flat in one
directory, each with a ``<md name>_images/`` folder).

Marker crops every block the layout model labels as ``Picture`` / ``Figure``,
so publisher logos, license badges and other page furniture end up on disk next
to the real figures. An image counts as **valid** when it is a content figure:

    valid = (area >= min_area or figure number) and not bare boilerplate

``bare boilerplate`` means boilerplate text next to a crop that has nothing else
to recommend it: it is either small or has no figure number of its own. Journal
pages repeat their license footer within one line of the figure that sits at the
bottom of the page, so boilerplate alone must not veto a large, captioned figure.

* ``area`` — pixel area of the saved crop (``width * height``). At marker's
  192 DPI page rendering a real figure spans percent of a page while logos and
  icons stay small, so the default ``min_area = 60000`` px (roughly 1.9% of an
  A4 page) separates them with wide margin. ``--min-area 0`` disables the area
  rule entirely, so only a figure number can keep an image.
* ``figure number`` — a caption line (``Figure 4:``, ``Fig. S7``, ``Fig. R1-6``,
  ``Supplementary Figure 3``, ``Extended Data Fig. 2``, ...) directly before or
  after the image, or any ``Figure``/``Fig.`` reference inside the surrounding
  ``window`` lines. Keeps genuinely small figures that carry a caption.
* ``boilerplate`` — publisher boilerplate text adjacent to the image
  (``Open Access``, ``Creative Commons``, ``creativecommons.org``,
  ``Peer Review File``, ``ORCID``, the article ``https://doi.org/...`` banner,
  ...). This vetoes a crop that the area and figure-number rules do not claim on
  their own, because journal mastheads are wide enough to pass the area test; a
  large crop with its own caption survives a boilerplate line that merely
  happens to be the page footer.

Invalid images are deleted together with their Markdown link line (a link is
never left dangling), and the files are moved to ``<md name>_images_removed/``
so a wrong threshold can be rolled back. Every image gets a recorded verdict.

Usage::

    # see what would be removed, change nothing
    python -m src.utils.md_image_filter --dir path/to/markdowns --dry-run

    # prune, keep removed files in <md name>_images_removed/, write a JSONL log
    python -m src.utils.md_image_filter --dir path/to/markdowns \
        --report logs/md_image_prune.jsonl --log-file logs/md_image_prune.log

    # stricter: only a figure number keeps an image, no area allowance
    python -m src.utils.md_image_filter --dir path/to/markdowns --min-area 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.logger import get_logger, setup_logger

logger = get_logger("md_image_filter")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MIN_AREA = 60_000  # px; ~1.9% of a 192 DPI page (~1584x2016)
DEFAULT_WINDOW = 3  # context lines inspected around an image link
REMOVED_SUFFIX = "_images_removed"

#: Markdown link destination: a ``<...>`` form, or a bare target that may
#: contain spaces and one level of balanced parentheses.
_MD_DESTINATION = r"<[^<>\n]+>|[^()\s](?:[^()]|\([^()]*\))*"

#: A Markdown image link and the target it points to. Marker writes unquoted
#: targets that contain spaces (the document title) and, when the title itself
#: has parentheses (``... NbOX2 (X = Br, I)_images/...``), those balanced
#: parentheses are part of the target too.
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(\s*(" + _MD_DESTINATION + r")\)")

#: A line that holds nothing but a single image link (dropped as a whole).
IMAGE_ONLY_LINE_RE = re.compile(
    r"^\s*!\[[^\]]*\]\(\s*(?:" + _MD_DESTINATION + r")\)\s*$"
)

#: marker's image names: ``_page_14_Picture_1.jpeg`` / ``_page_3_Figure_0.png``.
IMAGE_NAME_RE = re.compile(
    r"^_?page_(?P<page>\d+)_(?P<kind>[A-Za-z]+)_(?P<index>\d+)\.(?P<ext>[A-Za-z0-9]+)$"
)

#: Caption start: ``Figure 4:``, ``*Fig. S7:*``, ``**Fig. R1.**``, ``New Fig.4``,
#: ``Supplementary Figure 3``, ``Extended Data Fig. 2``, ``Graphic 1``.
CAPTION_LINE_RE = re.compile(
    r"(?i)"
    r"(?:supplementary|extended\s+data|online\s+|supporting\s+information\s+)?"
    r"(?:figure|fig\.?|graphic)\s*[.:]?\s*(?:\d|[A-Z]\b)"
)

#: Any figure reference inside the context window, including ``Fig. 4(c)``.
FIGURE_REF_RE = re.compile(r"(?i)\b(?:fig(?:ure)?\.?)\s*\d+")

#: Publisher / journal boilerplate that marks a non-content image.
BOILERPLATE_RE = re.compile(
    r"(?i)("
    r"open\s+access"
    r"|creative\s+commons"
    r"|creativecommons\.org"
    r"|is\s+licensed\s+under"
    r"|peer\s+review\s+file"
    r"|editor'?s\s+note"
    r"|publisher'?s\s+disclaimer"
    r"|downloaded\s+from"
    r"|https?://doi\.org/"  # the article DOI banner beside the journal masthead
    r"|\borcid\b"
    r")"
)

_EMPHASIS_RE = re.compile(r"[*_`~]+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LINK_TITLE_RE = re.compile(r"\s+[\"'][^\"']*[\"']\s*$")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class ImageVerdict:
    """Classification result for a single extracted image."""

    name: str  # file name inside the image folder
    link: str  # link text as written in the Markdown ("" when orphan)
    line: int  # 1-based Markdown line holding the link (0 when orphan)
    page: int | None = None  # page index parsed from the file name
    kind: str = ""  # ``Picture`` / ``Figure`` / ...
    width: int = 0
    height: int = 0
    area: int = 0
    has_caption: bool = False
    figure_refs: int = 0
    boilerplate: bool = False
    duplicate_of: str = ""  # file name of the byte-identical earlier image
    readable: bool = True  # False when the crop cannot be opened
    valid: bool = True
    reason: str = ""


@dataclass
class PruneResult:
    """Per-Markdown-file bookkeeping of a prune pass."""

    markdown: str
    total: int = 0
    kept: int = 0
    removed: int = 0
    orphans: int = 0
    missing: int = 0  # links whose file is absent on disk (left untouched)
    external: int = 0  # http(s) / data: links (left untouched)
    verdicts: list[ImageVerdict] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)


# =============================================================================
# Private Functions
# =============================================================================


def _setup_logging(log_file: str | None, verbose: bool) -> None:
    """Configure the project logger for a prune run.

    Args:
        log_file: Optional path for a rotating log file (for traceability).
        verbose: When True, use DEBUG level instead of INFO.
    """
    setup_logger(log_file=log_file, level="DEBUG" if verbose else "INFO")


def _clean_context(text: str) -> str:
    """Strip Markdown emphasis and HTML tags so caption patterns can match."""
    return _EMPHASIS_RE.sub("", _HTML_TAG_RE.sub("", text)).strip()


def _is_external(target: str) -> bool:
    """True for links that do not point at a local file."""
    return bool(re.match(r"(?i)^(https?:|data:|mailto:|//)", target))


def _normalize_target(raw: str) -> str:
    """Clean a link target: trim spaces and drop an optional ``\"title\"``."""
    return _LINK_TITLE_RE.sub("", raw.strip())


def _strip_link_from_line(line: str, target: str) -> str:
    """Remove the image markup (label + target) that pointed at ``target``."""
    pattern = re.compile(r"!\[[^\]]*\]\(\s*" + re.escape(target) + r"\s*\)")
    return pattern.sub("", line)


def _image_size(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` of an image file, or None when unreadable."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as exc:  # corrupt or truncated crop
        logger.warning(f"Cannot read image {path}: {exc}")
        return None


def _file_digest(path: Path) -> str:
    """MD5 of a file, used only to spot byte-identical duplicate crops."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_lines(
    lines: list[str], index: int, window: int
) -> tuple[list[str], list[str]]:
    """Non-empty, cleaned lines around ``index`` (0-based), before then after."""
    before: list[str] = []
    for offset in range(1, window + 1):
        if index - offset < 0:
            break
        text = _clean_context(lines[index - offset])
        if text:
            before.append(text)
    after: list[str] = []
    for offset in range(1, window + 1):
        if index + offset >= len(lines):
            break
        text = _clean_context(lines[index + offset])
        if text:
            after.append(text)
    before.reverse()
    return before, after


def _matches_boilerplate(
    texts: Iterable[str], patterns: list[re.Pattern[str]]
) -> bool:
    """True when any adjacent line looks like publisher boilerplate."""
    return any(pattern.search(text) for text in texts for pattern in patterns)


def _compile_boilerplate(extra: Iterable[str] | None) -> list[re.Pattern[str]]:
    """Build the boilerplate matchers, appending user-supplied patterns."""
    patterns = [BOILERPLATE_RE]
    for raw in extra or []:
        try:
            patterns.append(re.compile(raw, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid --boilerplate regex {raw!r}: {exc}") from exc
    return patterns


# =============================================================================
# Public Functions
# =============================================================================


def classify_markdown_images(
    md_path: str | Path,
    min_area: int = DEFAULT_MIN_AREA,
    window: int = DEFAULT_WINDOW,
    extra_boilerplate: Iterable[str] | None = None,
    dedupe: bool = False,
) -> tuple[list[ImageVerdict], list[str], list[str]]:
    """Classify every extracted image of one Markdown file.

    Args:
        md_path: Markdown file whose ``<stem>_images/`` folder is inspected.
        min_area: Pixel area at or above which an image is a content figure.
        window: Context lines inspected around each image link.
        extra_boilerplate: Additional regexes that veto an image.
        dedupe: Also flag byte-identical duplicates (default keeps them).

    Returns:
        ``(verdicts, lines, link_order)`` where ``verdicts`` lists every image
        (linked ones in document order, then orphans), ``lines`` is the
        Markdown split by line and ``link_order`` the link targets in order.

    Raises:
        FileNotFoundError: If ``md_path`` does not exist.
        ValueError: If an ``extra_boilerplate`` regex does not compile.
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file does not exist: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    image_dir = path.parent / f"{path.stem}_images"
    boilerplates = _compile_boilerplate(extra_boilerplate)

    # link target -> 0-based line index (first occurrence wins)
    link_lines: dict[str, int] = {}
    for index, line in enumerate(lines):
        for raw_target in IMAGE_LINK_RE.findall(line):
            target = _normalize_target(raw_target)
            link_lines.setdefault(target, index)

    on_disk = sorted(p.name for p in image_dir.glob("*")) if image_dir.is_dir() else []
    on_disk_set = set(on_disk)

    linked_names: list[str] = []
    for target in link_lines:
        if _is_external(target):
            continue
        name = Path(target).name
        if name in on_disk_set and name not in linked_names:
            linked_names.append(name)
    orphans = [name for name in on_disk if name not in linked_names]

    verdicts: list[ImageVerdict] = []
    seen_digests: dict[str, str] = {}
    for name in linked_names + orphans:
        file_path = image_dir / name
        target = next((t for t in link_lines if Path(t).name == name), "")
        line_index = link_lines.get(target, -1)
        parsed = IMAGE_NAME_RE.match(name)
        size = _image_size(file_path)

        has_caption = False
        figure_refs = 0
        boiler = False
        if line_index >= 0:
            before, after = _context_lines(lines, line_index, window)
            has_caption = bool(CAPTION_LINE_RE.match(after[0])) if after else False
            if not has_caption and before:
                has_caption = bool(CAPTION_LINE_RE.match(before[-1]))
            figure_refs = sum(len(FIGURE_REF_RE.findall(line)) for line in before + after)
            boiler = _matches_boilerplate(before[-1:] + after[:1], boilerplates)

        width, height = size if size else (0, 0)
        area_ok = min_area > 0 and width * height >= min_area
        verdict = ImageVerdict(
            name=name,
            link=target,
            line=line_index + 1,
            page=int(parsed.group("page")) if parsed else None,
            kind=parsed.group("kind") if parsed else "",
            width=width,
            height=height,
            area=width * height,
            has_caption=has_caption,
            figure_refs=figure_refs,
            boilerplate=boiler,
            readable=size is not None,
        )

        content_figure = area_ok and (has_caption or figure_refs)

        if not verdict.readable:
            verdict.valid = False
            verdict.reason = "unreadable image file"
        elif not verdict.link:
            verdict.valid = False
            verdict.reason = "orphan: not referenced in the Markdown"
        elif boiler and not content_figure:
            verdict.valid = False
            verdict.reason = (
                "publisher boilerplate (license banner / masthead) adjacent to it"
            )
        elif area_ok:
            verdict.reason = f"area {verdict.area} >= {min_area}"
            if boiler:
                verdict.reason += ", own figure number overrides the boilerplate line"
        elif has_caption or figure_refs:
            verdict.reason = (
                f"figure number nearby (caption={int(has_caption)}, "
                f"refs={figure_refs})"
            )
        elif min_area > 0:
            verdict.valid = False
            verdict.reason = (
                f"area {verdict.area} < {min_area} and no figure number nearby"
            )
        else:
            verdict.valid = False
            verdict.reason = (
                "no figure number nearby (--min-area 0 requires a caption)"
            )

        if dedupe and verdict.valid:
            digest = _file_digest(file_path)
            if digest in seen_digests:
                verdict.valid = False
                verdict.duplicate_of = seen_digests[digest]
                verdict.reason = f"duplicate of {seen_digests[digest]}"
            else:
                seen_digests[digest] = name

        verdicts.append(verdict)

    return verdicts, lines, list(link_lines)


def prune_markdown_images(
    md_path: str | Path,
    min_area: int = DEFAULT_MIN_AREA,
    window: int = DEFAULT_WINDOW,
    extra_boilerplate: Iterable[str] | None = None,
    dedupe: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> PruneResult:
    """Remove non-content images and their Markdown links from one file.

    Args:
        md_path: Markdown file to clean; its ``<stem>_images/`` folder holds the
            extracted images.
        min_area: Pixel area threshold (see :func:`classify_markdown_images`).
        window: Context lines inspected around each image link.
        extra_boilerplate: Additional veto regexes for publisher furniture.
        dedupe: Also remove byte-identical duplicate crops.
        backup: Move removed files to ``<stem>_images_removed/`` rather than
            deleting them.
        dry_run: Classify only; never touch Markdown or image files.

    Returns:
        A :class:`PruneResult` with per-image verdicts and counters.
    """
    path = Path(md_path)
    verdicts, lines, link_order = classify_markdown_images(
        path,
        min_area=min_area,
        window=window,
        extra_boilerplate=extra_boilerplate,
        dedupe=dedupe,
    )
    image_dir = path.parent / f"{path.stem}_images"
    result = PruneResult(markdown=str(path), total=len(verdicts), verdicts=verdicts)

    linked_names = {verdict.name for verdict in verdicts if verdict.link}
    for target in link_order:
        if _is_external(target):
            result.external += 1
        elif Path(target).name not in linked_names:
            result.missing += 1
            logger.warning(f"{path.name}: broken link, file missing: {target}")
    result.orphans = sum(1 for verdict in verdicts if not verdict.link)

    invalid = [verdict for verdict in verdicts if not verdict.valid]
    result.kept = result.total - len(invalid)
    result.removed = len(invalid)
    result.changes = [f"{verdict.name}: {verdict.reason}" for verdict in invalid]
    if not invalid:
        return result

    if dry_run:
        logger.info(
            f"[dry-run] {path.name}: would remove {result.removed} of "
            f"{result.total} image(s)"
        )
        return result

    by_line: dict[int, list[ImageVerdict]] = {}
    for verdict in invalid:
        if verdict.link:
            by_line.setdefault(verdict.line, []).append(verdict)

    kept_lines: list[str] = []
    dropped_lines = 0
    dropped_inline = 0
    for index, line in enumerate(lines, start=1):
        hits = by_line.get(index, [])
        if not hits:
            kept_lines.append(line)
            continue
        if IMAGE_ONLY_LINE_RE.match(line):
            dropped_lines += 1
            continue
        cleaned = line
        for verdict in hits:
            cleaned = _strip_link_from_line(cleaned, verdict.link)
        if cleaned != line:
            dropped_inline += 1
        kept_lines.append(cleaned)

    text = _BLANK_RUN_RE.sub("\n\n", "\n".join(kept_lines)).rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")

    removed_dir = path.parent / f"{path.stem}{REMOVED_SUFFIX}"
    for verdict in invalid:
        source = image_dir / verdict.name
        if not source.exists():
            continue
        if backup:
            removed_dir.mkdir(parents=True, exist_ok=True)
            target = removed_dir / verdict.name
            if target.exists():
                target.unlink()
            source.replace(target)
        else:
            source.unlink()
    if removed_dir.is_dir() and not any(removed_dir.iterdir()):
        removed_dir.rmdir()

    logger.info(
        f"{path.name}: removed {result.removed} of {result.total} image(s) "
        f"({dropped_lines} link line(s), {dropped_inline} inline link(s))"
    )
    return result


def prune_directory(
    root: str | Path,
    recursive: bool = False,
    min_area: int = DEFAULT_MIN_AREA,
    window: int = DEFAULT_WINDOW,
    extra_boilerplate: Iterable[str] | None = None,
    dedupe: bool = False,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prune non-content images across every Markdown file of a directory.

    Args:
        root: Directory containing the Markdown files.
        recursive: Also walk Markdown files in sub-directories.
        min_area: Pixel area threshold.
        window: Context lines inspected around each image link.
        extra_boilerplate: Additional veto regexes.
        dedupe: Also remove byte-identical duplicate crops.
        backup: Move removed files to ``<stem>_images_removed/``.
        dry_run: Report only, change nothing.

    Returns:
        Summary dict with ``files``, ``changed_files``, ``images``, ``removed``,
        ``orphans``, ``missing``, ``external``, the effective thresholds and a
        per-file ``details`` list (each entry holds the verdicts).

    Raises:
        NotADirectoryError: If ``root`` is not a directory.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    md_files = sorted(root_path.rglob("*.md") if recursive else root_path.glob("*.md"))
    summary: dict[str, Any] = {
        "root": str(root_path),
        "files": len(md_files),
        "changed_files": 0,
        "images": 0,
        "removed": 0,
        "orphans": 0,
        "missing": 0,
        "external": 0,
        "min_area": min_area,
        "window": window,
        "dedupe": dedupe,
        "backup": backup,
        "dry_run": dry_run,
        "details": [],
    }

    for md_path in md_files:
        result = prune_markdown_images(
            md_path,
            min_area=min_area,
            window=window,
            extra_boilerplate=extra_boilerplate,
            dedupe=dedupe,
            backup=backup,
            dry_run=dry_run,
        )
        if not result.verdicts and not result.missing:
            continue
        summary["images"] += result.total
        summary["removed"] += result.removed
        summary["orphans"] += result.orphans
        summary["missing"] += result.missing
        summary["external"] += result.external
        if result.removed and not dry_run:
            summary["changed_files"] += 1
        entry = asdict(result)
        entry["removed_images"] = [
            {
                key: getattr(verdict, key)
                for key in (
                    "name",
                    "line",
                    "page",
                    "kind",
                    "width",
                    "height",
                    "area",
                    "reason",
                )
            }
            for verdict in result.verdicts
            if not verdict.valid
        ]
        summary["details"].append(entry)

    logger.info(
        f"Scanned {summary['files']} Markdown file(s) in {root_path}: "
        f"{summary['images']} image(s), {summary['removed']} removed"
        f"{' (dry-run)' if dry_run else ''}"
    )
    return summary


def write_report(summary: dict[str, Any], report_path: str | Path) -> None:
    """Write a JSONL prune report (one line per Markdown file with images)."""
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in summary["details"]:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Report written: {target}")


def format_verdicts(entry: dict[str, Any]) -> list[str]:
    """Human-readable verdict lines for one scanned Markdown file."""
    rows = [f"  {Path(entry['markdown']).name}"]
    for verdict in entry["verdicts"]:
        flag = "keep " if verdict["valid"] else "PRUNE"
        rows.append(
            f"    {flag} {verdict['name']:<28} "
            f"{verdict['width']:>4}x{verdict['height']:<4} "
            f"area={verdict['area']:>7} cap={int(verdict['has_caption'])} "
            f"refs={verdict['figure_refs']} "
            f"boiler={int(verdict['boilerplate'])} | {verdict['reason']}"
        )
    return rows


def main() -> int:
    """Command-line entry point: prune non-content images from a Markdown tree.

    Returns:
        Process exit code; 0 when the pass completed, 1 on a fatal error.
    """
    parser = argparse.ArgumentParser(
        description="Remove non-content images (logos, banners, icons) and their "
        "Markdown links from a marker-converted Markdown tree"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory holding the .md files and their <md name>_images folders",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process Markdown files in sub-directories",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=f"Pixel area at or above which an image is kept "
        f"(default: {DEFAULT_MIN_AREA}; 0 requires a figure number)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Context lines inspected around each image "
        f"(default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--boilerplate",
        action="append",
        default=None,
        metavar="REGEX",
        help="Extra regex that vetoes an image when its adjacent line matches "
        "(repeatable)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Also remove byte-identical duplicate crops (keeps the first one)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the verdicts without changing Markdown or image files",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help=f"Delete removed images instead of moving them to "
        f"<md name>{REMOVED_SUFFIX}/",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write a JSONL report of every scanned Markdown file to this path",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a rotating log file for traceability",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging and print the full verdict table",
    )
    args = parser.parse_args()

    _setup_logging(args.log_file, args.verbose)

    try:
        summary = prune_directory(
            root=args.dir,
            recursive=args.recursive,
            min_area=args.min_area,
            window=args.window,
            extra_boilerplate=args.boilerplate,
            dedupe=args.dedupe,
            backup=args.backup,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    if args.verbose:
        for entry in summary["details"]:
            for row in format_verdicts(entry):
                print(row)

    if args.report:
        write_report(summary, args.report)

    logger.info(
        f"Summary: {summary['changed_files']} of {summary['files']} file(s) "
        f"changed, {summary['removed']} of {summary['images']} image(s) removed "
        f"({summary['orphans']} orphan(s), {summary['missing']} broken link(s), "
        f"{summary['external']} external link(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
