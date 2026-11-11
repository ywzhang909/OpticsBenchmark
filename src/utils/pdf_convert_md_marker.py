#!/usr/bin/env python3
"""
Optis Benchmark - PDF to Markdown Converter (Marker)

Batch-converts every PDF in an input directory into Markdown using
`marker-pdf <https://github.com/datalab-to/marker>`_. The model weights are
loaded once and reused for the whole batch, and per-file progress plus a final
summary are logged so a long run stays traceable.

Conversion options are read from a YAML config file (default:
``configs/dataset/marker_pdf2md.yaml``) and passed to marker's
``ConfigParser``. See ``ref_document/marker_config_reference.zh.md`` for the
full parameter reference, or run ``marker_single --help``.

When ``extract_images`` is enabled (the marker default, and the repo config),
the images marker crops out of each document are written next to the Markdown
into a per-document ``<md name>_images/`` folder and the Markdown links are
rewritten to point there, so Markdown files can stay flat in one directory
without image names colliding across documents. Pass ``--no-images`` to keep
Markdown-only output.

Environment variables (set them in ``.env`` or the shell; they are read *before*
``marker`` is imported, which is required by surya):
    SURYA_INFERENCE_URL  Optional. Base URL of a remote surya inference server.
    HF_ENDPOINT          Optional. Hugging Face mirror used for model downloads.

.. note::
    This module intentionally contains no hard-coded host addresses. Configure
    the variables above in your own ``.env`` instead of editing the source.

.. note::
    VLM work needs a surya inference server: ``mode: balanced`` (VLM layout and
    full-page OCR) and ``mode: fast`` with OCR enabled both call it — verified
    with marker-pdf 2.0.0, where ``fast`` fails immediately without a server even
    for a clean text-layer PDF. With ``SURYA_INFERENCE_URL`` set, marker attaches
    to that server and fails fast when it is unreachable; with the variable
    unset, surya launches its own temporary ``vllm`` docker container instead.
    Only ``mode: fast`` combined with ``disable_ocr: true`` runs without a server.

.. note::
    The cropped images are the *layout model's* Picture/Figure blocks, which
    include publisher logos and license badges. ``src.utils.md_image_filter``
    prunes those non-content images (and their Markdown links) afterwards.

Usage::

    python -m src.utils.pdf_convert_md_marker --input-dir path/to/pdfs
    python -m src.utils.pdf_convert_md_marker --input-dir path/to/pdfs --output-dir out
    python -m src.utils.pdf_convert_md_marker --input-dir path/to/pdfs --output-dir out \\
        --recursive --skip-existing --log-file logs/pdf2md.log
    python -m src.utils.pdf_convert_md_marker --input-dir path/to/pdfs \\
        --marker-config configs/dataset/marker_pdf2md.yaml
    python -m src.utils.pdf_convert_md_marker --input-dir path/to/pdfs --output-dir out \\
        --overwrite            # re-convert files that already have .md (adds images)
    python -m src.utils.pdf_convert_md_marker --input-dir path/to/pdfs --no-images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time
from typing import Any

from tqdm import tqdm

from src.utils.logger import get_logger, setup_logger
from src.utils.parser import ConfigParser as ConfigLoader

# =============================================================================
# Constants
# =============================================================================
#
# Marker conversion options are loaded from a YAML config file whose keys are
# marker's ConfigParser option names (the CLI flag name without the leading
# "--"). The most common / important options:
#
#   mode                        "balanced" (VLM layout + full-page OCR, best on
#                               GPU) | "fast" (lightweight CPU detectors).
#                               Defaults by device: balanced on GPU, fast on CPU.
#   output_format               "markdown" | "json" | "html" | "chunks".
#   page_range                  e.g. "0,5-10,20" (0-based); null = all pages.
#   extract_images              Extract figures from the document (True/False).
#                               Saved images land in "<md name>_images/" next to
#                               the .md file (see convert_pdf / _save_images).
#   lowres_image_dpi            DPI for layout detection (default 96).
#   highres_image_dpi           DPI for OCR (default 192).
#   disable_ocr                 Disable OCR entirely (text-layer only).
#   force_ocr                   Force full-document OCR (skips layout).
#   ocr_full_page               One full-page OCR request per page (default True).
#   keep_page_in_output   Keep running headers in the output (False).
#   keep_pagefooter_in_output   Keep running footers in the output (False).
#   paginate_output             Insert a separator between pages (markdown/html).
#   page_separator              The page separator string (MarkdownRenderer).
#   html_tables_in_markdown     Emit tables as HTML instead of Markdown pipes.
#   filter_blank_pages          Drop blank pages detected as images.
#   pdftext_workers             Number of pdftext extraction workers (default 4).
#   use_llm                     Enable LLM-assisted cleanup (tables/equations).
#   llm_service                 Full import path of the LLM service to use.
#   debug                       Dump debug images/JSON into the output folder.
#
# Component-specific overrides use the "<Component>_<param>" form, e.g.
# "MarkdownRenderer_page_separator" or "PdfProvider_pdftext_workers".
#
# Full reference (Chinese): ref_document/marker_config_reference.zh.md
# Authoritative list:        marker_single --help

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "dataset" / "marker_pdf2md.yaml"

_PDF_SUFFIX = ".pdf"

logger = get_logger("pdf_convert_md_marker")


# =============================================================================
# Private Functions
# =============================================================================


def _build_output_path(
    pdf_path: Path,
    input_dir: Path,
    output_dir: Path | None,
    recursive: bool,
) -> Path:
    """Resolve the Markdown output path for a single PDF.

    When ``output_dir`` is omitted the Markdown file is written next to the
    source PDF. With ``recursive`` the relative sub-directory structure below
    ``input_dir`` is mirrored under the output directory.

    Args:
        pdf_path: Source PDF path.
        input_dir: Root directory the PDF was discovered under.
        output_dir: Destination directory, or None to write next to the PDF.
        recursive: Whether the input tree is traversed recursively.

    Returns:
        The target ``.md`` path (parent directories are not created here).
    """
    target_dir = output_dir if output_dir is not None else pdf_path.parent
    if recursive:
        target_dir = target_dir / pdf_path.relative_to(input_dir).parent
    return target_dir / f"{pdf_path.stem}.md"


def _setup_logging(log_file: str | None, verbose: bool) -> None:
    """Configure the project logger for a conversion run.

    Args:
        log_file: Optional path for a rotating log file (for traceability).
        verbose: When True, use DEBUG level instead of INFO.
    """
    setup_logger(log_file=log_file, level="DEBUG" if verbose else "INFO")


# =============================================================================
# Public Functions
# =============================================================================


def iter_pdf_files(input_dir: str | Path, recursive: bool = False) -> list[Path]:
    """List the PDF files contained in a directory.

    Args:
        input_dir: Directory to scan.
        recursive: If True, also scan sub-directories.

    Returns:
        Sorted list of PDF paths (extension matching is case-insensitive).

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
        NotADirectoryError: If ``input_dir`` is not a directory.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    pdfs = [
        p
        for p in candidates
        if p.is_file() and p.suffix.lower() == _PDF_SUFFIX
    ]
    return sorted(pdfs)


def load_marker_options(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load marker ConfigParser options from a YAML/JSON config file.

    The file keys are marker's option names (CLI flag names without the leading
    ``--``); ``${ENV_VAR}`` placeholders are expanded from the environment.

    Args:
        config_path: Path to the config file; defaults to
            ``configs/dataset/marker_pdf2md.yaml``.

    Returns:
        A dict of marker options ready to hand to
        ``marker.config.parser.ConfigParser``.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Marker config file not found: {path}")

    options = ConfigLoader.load_config(path)
    # ConfigParser.get_renderer() indexes cli_options["output_format"], so make
    # sure a valid format is always present (defaults to Markdown).
    if not options.get("output_format"):
        options["output_format"] = "markdown"
    return options


def build_converter(config_path: str | Path | None = None) -> Any:
    """Load the marker model weights once and return a configured converter.

    Options are read from ``config_path`` (default:
    ``configs/dataset/marker_pdf2md.yaml``). ``load_dotenv()`` is called before
    importing ``marker`` so that ``SURYA_INFERENCE_URL`` / ``HF_ENDPOINT`` (and
    any LLM credentials) are visible to surya at import time.

    The converter is built exactly like marker's own CLI, so ``output_format``,
    ``processors``, ``use_llm`` and ``llm_service`` from the config take effect.

    Args:
        config_path: Optional path to the marker options config file.

    Returns:
        A ready-to-use converter (``PdfConverter`` by default).
    """
    from dotenv import load_dotenv

    load_dotenv()

    from marker.config.parser import ConfigParser as MarkerConfigParser
    from marker.models import create_model_dict

    options = load_marker_options(config_path)
    config_parser = MarkerConfigParser(options)

    logger.info("Loading marker model weights (first run may download them)...")
    load_start = time()
    converter = config_parser.get_converter_cls()(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    logger.info(f"Model weights ready (took {time() - load_start:.1f}s)")
    return converter


def _save_images(
    images: dict[str, Any], markdown_text: str, md_path: Path
) -> tuple[str, str, int, list[str]]:
    """Write marker's extracted images next to the Markdown and rewrite links.

    Images keep marker's own names (``_page_14_Picture_1.jpeg``) but are placed
    in a per-document ``<markdown stem>_images`` folder, so Markdown files can
    stay flat in one directory without image names colliding across documents.
    Only successfully saved images get their ``](name)`` link rewritten.

    Args:
        images: ``{name: PIL.Image}`` returned by ``text_from_rendered``.
        markdown_text: Rendered Markdown whose image links get rewritten.
        md_path: Destination Markdown path; decides the image folder name.

    Returns:
        ``(markdown_text, image_subdir, saved_count, failed_names)`` where
        ``markdown_text`` has its saved image links rewritten and
        ``image_subdir`` is empty when nothing was saved.
    """
    from marker.output import convert_if_not_rgb
    from marker.settings import settings

    if not images:
        return markdown_text, "", 0, []

    image_subdir = f"{md_path.stem}_images"
    image_dir = md_path.parent / image_subdir
    image_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed: list[str] = []
    for name, image in images.items():
        try:
            convert_if_not_rgb(image).save(
                image_dir / name, settings.OUTPUT_IMAGE_FORMAT
            )
        except Exception as exc:  # keep the Markdown even if one image fails
            failed.append(name)
            logger.warning(
                f"Failed to save image {name} for {md_path.name}: {exc} "
                f"(link left unchanged)"
            )
            continue
        markdown_text = markdown_text.replace(f"]({name})", f"]({image_subdir}/{name})")
        saved += 1

    if not saved:
        # nothing was written — do not leave an empty folder behind
        if not any(image_dir.iterdir()):
            image_dir.rmdir()
        return markdown_text, "", 0, failed

    return markdown_text, image_subdir, saved, failed


def convert_pdf(
    converter: Any,
    pdf_path: str | Path,
    output_path: str | Path,
    save_images: bool = True,
) -> dict[str, Any]:
    """Convert a single PDF into Markdown and write it to ``output_path``.

    Args:
        converter: A converter created by :func:`build_converter`.
        pdf_path: Source PDF path.
        output_path: Destination ``.md`` path; parent directories are created.
        save_images: Also write the images marker extracted from the document
            into ``<output stem>_images/`` and rewrite their Markdown links.

    Returns:
        A dict with ``path`` (written Markdown), ``pages``, ``images``
        (saved image count), ``image_dir`` and ``image_failures``.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If marker returns empty content.
    """
    from marker.output import text_from_rendered

    src = Path(pdf_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"PDF file does not exist: {src}")

    size_mb = src.stat().st_size / (1024 * 1024)
    logger.info(f"Converting: {src.name} ({size_mb:.2f} MB)")
    convert_start = time()

    rendered = converter(str(src))
    markdown_text, _, images = text_from_rendered(rendered)
    if not markdown_text:
        raise RuntimeError(f"marker returned empty Markdown for: {src}")

    page_count = len(rendered.pages) if hasattr(rendered, "pages") else "unknown"
    extracted = len(images) if images else 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    image_dir, saved_images, image_failures = "", 0, []
    if save_images and images:
        markdown_text, image_dir, saved_images, image_failures = _save_images(
            images, markdown_text, dst
        )
    elif images:
        logger.info(f"{src.name}: {extracted} image(s) found, saving disabled")

    dst.write_text(markdown_text, encoding="utf-8")

    logger.info(
        f"Done: {src.name} -> {dst} "
        f"({page_count} pages, {saved_images}/{extracted} images"
        f"{f' -> {image_dir}/' if image_dir else ''}, "
        f"{time() - convert_start:.1f}s, {dst.stat().st_size / 1024:.1f} KB)"
    )
    return {
        "path": dst,
        "pages": page_count,
        "images": saved_images,
        "image_dir": image_dir,
        "image_failures": image_failures,
    }


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    skip_existing: bool = False,
    config_path: str | Path | None = None,
    converter: Any | None = None,
    save_images: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert every PDF in a directory into Markdown.

    Args:
        input_dir: Directory holding the PDF files.
        output_dir: Destination directory; when None, write next to each PDF.
        recursive: If True, also convert PDFs in sub-directories.
        skip_existing: If True, skip PDFs whose Markdown already exists.
            Ignored when ``overwrite`` is True.
        config_path: Marker options config file passed to :func:`build_converter`.
        converter: Optional pre-built converter to reuse; built on demand.
        save_images: Save marker's extracted images into ``<md stem>_images/``
            and rewrite their Markdown links.
        overwrite: Re-convert PDFs even when their Markdown already exists
            (used to add images to previously converted files).

    Returns:
        A summary dict with keys ``total``, ``succeeded``, ``skipped``,
        ``failed``, ``images``, ``image_failures``, ``failures`` (list of
        ``{file, error}``) and ``elapsed``.
    """
    input_root = Path(input_dir)
    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    if overwrite and skip_existing:
        logger.warning("--overwrite takes precedence over --skip-existing")
        skip_existing = False

    pdfs = iter_pdf_files(input_root, recursive=recursive)
    summary: dict[str, Any] = {
        "total": len(pdfs),
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
        "images": 0,
        "image_failures": [],
        "failures": [],
        "elapsed": 0.0,
    }
    if not pdfs:
        logger.warning(f"No PDF files found in: {input_root}")
        return summary

    if converter is None:
        converter = build_converter(config_path)

    logger.info(f"Found {len(pdfs)} PDF file(s) in {input_root}")

    start = time()
    progress = tqdm(pdfs, desc="PDF -> MD", unit="file", disable=None)
    for index, pdf_path in enumerate(progress, start=1):
        progress.set_postfix_str(pdf_path.name, refresh=False)
        target = _build_output_path(pdf_path, input_root, output_root, recursive)

        if skip_existing and target.exists():
            summary["skipped"] += 1
            logger.info(f"[{index}/{len(pdfs)}] Skip (exists): {target}")
            continue

        try:
            result = convert_pdf(converter, pdf_path, target, save_images=save_images)
            summary["succeeded"] += 1
            summary["images"] += result["images"]
            summary["image_failures"].extend(f"{target.name}:{n}" for n in result["image_failures"])
        except Exception as exc:  # keep going: one bad file must not abort the batch
            summary["failed"] += 1
            summary["failures"].append({"file": str(pdf_path), "error": str(exc)})
            logger.error(f"[{index}/{len(pdfs)}] Failed: {pdf_path} -> {exc}")

    summary["elapsed"] = time() - start
    return summary


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Command-line entry point: batch-convert a directory of PDFs to Markdown.

    Returns:
        Process exit code; 0 when every file succeeded (or was skipped),
        1 when at least one conversion failed.
    """
    parser = argparse.ArgumentParser(
        description="Convert every PDF in a directory to Markdown using marker-pdf"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        # default="/path/to/raw_review_files",  # machine-local path, keep --input-dir required
        help="Directory containing the PDF files to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/paper_review/raw_reviews_md",
        help="Destination directory for the .md files (default: next to each PDF)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also convert PDFs in sub-directories (mirrors the relative tree)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs whose Markdown output already exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-convert PDFs even when their Markdown already exists "
            "(use it to add images to previously converted files)"
        ),
    )
    parser.add_argument(
        "--no-images",
        dest="save_images",
        action="store_false",
        help="Do not save images marker extracts (Markdown links stay unchanged)",
    )
    parser.add_argument(
        "--marker-config",
        type=str,
        default="configs/dataset/marker_pdf2md.yaml",
        help=(
            "Marker ConfigParser options file (default: "
            "configs/dataset/marker_pdf2md.yaml)"
        ),
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
        help="Enable DEBUG level logging",
    )

    args = parser.parse_args()
    _setup_logging(args.log_file, args.verbose)

    try:
        summary = convert_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
            config_path=args.marker_config,
            save_images=args.save_images,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(str(exc))
        return 1

    logger.info(
        f"Summary: {summary['succeeded']} succeeded, "
        f"{summary['skipped']} skipped, {summary['failed']} failed "
        f"out of {summary['total']} in {summary['elapsed']:.1f}s; "
        f"{summary['images']} image(s) saved"
    )
    for failure in summary["failures"]:
        logger.error(f"  FAILED {failure['file']}: {failure['error']}")
    for image_failure in summary["image_failures"]:
        logger.warning(f"  IMAGE FAILED {image_failure}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
