"""
Optis Benchmark - Tests for the Markdown image filter

Covers ``src.utils.md_image_filter``: which extracted crops count as real
figures, and how pruning rewrites the Markdown and moves the removed files.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.utils.md_image_filter import (
    classify_markdown_images,
    prune_directory,
    prune_markdown_images,
    write_report,
)


def _write_image(directory: Path, name: str, size=(400, 300), color="red") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def _build_tree(
    tmp_path: Path,
    markdown: str,
    images: dict[str, tuple[int, int]],
    md_name: str = "Paper",
) -> Path:
    """Write ``md_name.md`` plus its ``_images`` folder and return the md path."""
    md_path = tmp_path / f"{md_name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    image_dir = tmp_path / f"{md_name}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, size in images.items():
        _write_image(image_dir, name, size=size)
    return md_path


LOGO = "![](Paper_images/_page_0_Picture_4.jpeg)"
FIGURE = "![](Paper_images/_page_4_Figure_1.jpeg)"
TINY = "![](Paper_images/_page_6_Picture_2.jpeg)"
CAPTIONED = "![](Paper_images/_page_7_Figure_3.jpeg)"


def _sample_markdown() -> str:
    return (
        "Open Access This file is licensed under a Creative Commons Attribution "
        "4.0 license.\n\n"
        f"{LOGO}\n\n"
        "# Peer Review File\n\n"
        "The scattering response is shown below.\n\n"
        f"{FIGURE}\n\n"
        "A decorative mark follows.\n\n"
        f"{TINY}\n\n"
        f"{CAPTIONED}\n\n"
        "*Figure 3: Waterfall plot of the transmission curves.*\n"
    )


def _sample_images() -> dict[str, tuple[int, int]]:
    return {
        "_page_0_Picture_4.jpeg": (292, 103),  # 30k px, logo under license text
        "_page_4_Figure_1.jpeg": (1100, 800),  # 880k px, real figure
        "_page_6_Picture_2.jpeg": (120, 90),  # 10.8k px, icon without a number
        "_page_7_Figure_3.jpeg": (300, 150),  # 45k px, small but captioned
    }


def _verdicts_by_name(md_path: Path, **kwargs):
    verdicts, _, _ = classify_markdown_images(md_path, **kwargs)
    return {verdict.name: verdict for verdict in verdicts}


def test_logo_and_icon_are_pruned_figures_kept(tmp_path: Path):
    md_path = _build_tree(tmp_path, _sample_markdown(), _sample_images())

    verdicts = _verdicts_by_name(md_path)

    assert verdicts["_page_0_Picture_4.jpeg"].valid is False
    assert "boilerplate" in verdicts["_page_0_Picture_4.jpeg"].reason
    assert verdicts["_page_6_Picture_2.jpeg"].valid is False
    assert "no figure number" in verdicts["_page_6_Picture_2.jpeg"].reason
    assert verdicts["_page_4_Figure_1.jpeg"].valid is True
    # small, but the caption line right after it keeps it
    assert verdicts["_page_7_Figure_3.jpeg"].valid is True
    assert verdicts["_page_7_Figure_3.jpeg"].has_caption is True
    assert (
        verdicts["_page_0_Picture_4.jpeg"].page == 0
        and verdicts["_page_0_Picture_4.jpeg"].kind == "Picture"
    )


def test_prune_removes_link_line_and_backs_up_file(tmp_path: Path):
    md_path = _build_tree(tmp_path, _sample_markdown(), _sample_images())

    result = prune_markdown_images(md_path)

    assert (result.total, result.removed, result.kept) == (4, 2, 2)
    text = md_path.read_text(encoding="utf-8")
    assert "_page_0_Picture_4.jpeg" not in text
    assert "_page_6_Picture_2.jpeg" not in text
    assert "_page_4_Figure_1.jpeg" in text and "_page_7_Figure_3.jpeg" in text
    # the whole link line is gone and no triple newline was left behind
    assert "\n\n\n" not in text
    assert "# Peer Review File" in text

    image_dir = tmp_path / "Paper_images"
    removed_dir = tmp_path / "Paper_images_removed"
    assert sorted(p.name for p in image_dir.iterdir()) == [
        "_page_4_Figure_1.jpeg",
        "_page_7_Figure_3.jpeg",
    ]
    assert sorted(p.name for p in removed_dir.iterdir()) == [
        "_page_0_Picture_4.jpeg",
        "_page_6_Picture_2.jpeg",
    ]


def test_prune_is_idempotent(tmp_path: Path):
    md_path = _build_tree(tmp_path, _sample_markdown(), _sample_images())
    before = md_path.read_text(encoding="utf-8")

    prune_markdown_images(md_path)
    after = md_path.read_text(encoding="utf-8")
    second = prune_markdown_images(md_path)

    assert second.removed == 0
    assert (second.total, second.kept) == (2, 2)
    assert md_path.read_text(encoding="utf-8") == after
    assert after != before


def test_no_backup_deletes_files(tmp_path: Path):
    md_path = _build_tree(tmp_path, _sample_markdown(), _sample_images())

    prune_markdown_images(md_path, backup=False)

    assert not (tmp_path / "Paper_images" / "_page_0_Picture_4.jpeg").exists()
    assert not (tmp_path / "Paper_images_removed").exists()


def test_orphan_image_is_pruned(tmp_path: Path):
    md_path = _build_tree(
        tmp_path,
        f"Body text.\n\n{FIGURE}\n",
        {
            "_page_4_Figure_1.jpeg": (1100, 800),
            "_page_9_Picture_0.jpeg": (900, 700),  # saved but never referenced
        },
    )

    verdicts = _verdicts_by_name(md_path)

    assert verdicts["_page_9_Picture_0.jpeg"].valid is False
    assert verdicts["_page_9_Picture_0.jpeg"].link == ""
    assert "orphan" in verdicts["_page_9_Picture_0.jpeg"].reason
    assert verdicts["_page_9_Picture_0.jpeg"].line == 0


def test_broken_link_and_external_link_are_left_alone(tmp_path: Path):
    markdown = (
        "Body text.\n\n"
        "![](Paper_images/_page_1_Figure_0.jpeg)\n\n"
        "![cover](https://example.org/cover.png)\n\n"
        f"{FIGURE}\n"
    )
    md_path = _build_tree(
        tmp_path, markdown, {"_page_4_Figure_1.jpeg": (1100, 800)}
    )

    result = prune_markdown_images(md_path)

    assert (result.total, result.removed) == (1, 0)
    assert (result.missing, result.external) == (1, 1)
    assert md_path.read_text(encoding="utf-8") == markdown


def test_targets_with_spaces_and_inline_links(tmp_path: Path):
    markdown = (
        "Open Access licensed under Creative Commons.\n\n"
        "![](Some Paper_images/_page_0_Picture_1.jpeg)\n\n"
        "As shown in ![](Some Paper_images/_page_2_Picture_9.jpeg) the lattice "
        "response peaks.\n"
    )
    md_path = _build_tree(
        tmp_path,
        markdown,
        {
            "_page_0_Picture_1.jpeg": (60, 60),  # logo: small + boilerplate
            "_page_2_Picture_9.jpeg": (80, 60),  # inline icon: small, no number
        },
        md_name="Some Paper",
    )

    verdicts = _verdicts_by_name(md_path)

    assert sorted(verdicts) == ["_page_0_Picture_1.jpeg", "_page_2_Picture_9.jpeg"]
    assert verdicts["_page_0_Picture_1.jpeg"].boilerplate is True
    assert verdicts["_page_2_Picture_9.jpeg"].valid is False

    result = prune_markdown_images(md_path)

    assert (result.total, result.removed, result.kept) == (2, 2, 0)
    text = md_path.read_text(encoding="utf-8")
    assert "_page_0_Picture_1.jpeg" not in text
    # the inline case strips only the image markup and keeps the sentence intact
    assert "As shown in  the lattice response peaks." in text
    removed_dir = tmp_path / "Some Paper_images_removed"
    assert sorted(p.name for p in removed_dir.iterdir()) == [
        "_page_0_Picture_1.jpeg",
        "_page_2_Picture_9.jpeg",
    ]


def test_min_area_zero_requires_a_figure_number(tmp_path: Path):
    md_path = _build_tree(tmp_path, _sample_markdown(), _sample_images())

    strict = _verdicts_by_name(md_path, min_area=0)

    # 1100x800 figure has no caption here, so it only survives via the area rule
    assert strict["_page_4_Figure_1.jpeg"].valid is False
    assert strict["_page_7_Figure_3.jpeg"].valid is True


def test_dedupe_removes_identical_crops(tmp_path: Path):
    markdown = (
        "Body text.\n\n"
        "![](Paper_images/_page_12_Picture_3.jpeg)\n\n"
        "![](Paper_images/_page_12_Figure_4.jpeg)\n"
    )
    md_path = tmp_path / "Paper.md"
    md_path.write_text(markdown, encoding="utf-8")
    image_dir = tmp_path / "Paper_images"
    _write_image(image_dir, "_page_12_Picture_3.jpeg", size=(800, 500))
    _write_image(image_dir, "_page_12_Figure_4.jpeg", size=(800, 500))

    keep_both = _verdicts_by_name(md_path)
    assert all(verdict.valid for verdict in keep_both.values())

    with_dupes = _verdicts_by_name(md_path, dedupe=True)
    assert with_dupes["_page_12_Picture_3.jpeg"].valid is True
    assert with_dupes["_page_12_Figure_4.jpeg"].valid is False
    assert with_dupes["_page_12_Figure_4.jpeg"].duplicate_of == (
        "_page_12_Picture_3.jpeg"
    )


def test_extra_boilerplate_regex_vetoes_large_image(tmp_path: Path):
    markdown = (
        "Downloaded from the journal portal.\n\n"
        "![](Paper_images/_page_2_Picture_1.jpeg)\n"
    )
    md_path = _build_tree(
        tmp_path, markdown, {"_page_2_Picture_1.jpeg": (1200, 900)}
    )

    default = _verdicts_by_name(md_path)
    assert default["_page_2_Picture_1.jpeg"].boilerplate is True

    md_path.write_text(
        "Scanned stamp of the submission office.\n\n"
        "![](Paper_images/_page_2_Picture_1.jpeg)\n",
        encoding="utf-8",
    )
    extra = _verdicts_by_name(md_path, extra_boilerplate=[r"submission office"])
    assert extra["_page_2_Picture_1.jpeg"].valid is False


def test_doi_banner_masthead_is_pruned_despite_size(tmp_path: Path):
    markdown = (
        "![](Paper_images/_page_0_Picture_0.jpeg)\n\n"
        "Article https://doi.org/10.1038/s41467-025-66990-4\n\n"
        "# Paper title\n"
    )
    md_path = _build_tree(tmp_path, markdown, {"_page_0_Picture_0.jpeg": (900, 600)})

    verdicts = _verdicts_by_name(md_path)

    assert verdicts["_page_0_Picture_0.jpeg"].boilerplate is True
    assert verdicts["_page_0_Picture_0.jpeg"].valid is False

    result = prune_markdown_images(md_path)

    assert result.removed == 1
    text = md_path.read_text(encoding="utf-8")
    assert "![](" not in text
    # only the image markup goes; the DOI text line stays part of the document
    assert "Article https://doi.org/10.1038/s41467-025-66990-4" in text


def test_page_footer_boilerplate_does_not_veto_a_captioned_figure(tmp_path: Path):
    """Nature-style pages: the masthead footer sits right under the last figure."""
    markdown = (
        "Fig. 2 Meta-optics characterization. The learned metalens is fabricated "
        "with electron-beam lithography.\n\n"
        "![](Paper_images/_page_5_Figure_2.jpeg)\n\n"
        "6 NATURE COMMUNICATIONS | (2021) 12:6493 | "
        "https://doi.org/10.1038/s41467-021-26443-0 | www.nature.com/naturecommunications\n"
    )
    md_path = _build_tree(
        tmp_path, markdown, {"_page_5_Figure_2.jpeg": (1008, 919)}
    )

    verdicts = _verdicts_by_name(md_path)
    verdict = verdicts["_page_5_Figure_2.jpeg"]

    assert verdict.boilerplate is True and verdict.has_caption is True
    assert verdict.valid is True
    assert "overrides the boilerplate" in verdict.reason
    assert prune_markdown_images(md_path).removed == 0


def test_small_badge_next_to_license_text_is_still_pruned(tmp_path: Path):
    markdown = (
        "![](Paper_images/_page_0_Picture_1.jpeg)\n\n"
        "Open Access This article is licensed under a Creative Commons license.\n"
    )
    md_path = _build_tree(tmp_path, markdown, {"_page_0_Picture_1.jpeg": (220, 60)})

    verdicts = _verdicts_by_name(md_path)

    assert verdicts["_page_0_Picture_1.jpeg"].valid is False
    assert "boilerplate" in verdicts["_page_0_Picture_1.jpeg"].reason


def test_parentheses_in_link_target_are_part_of_the_file(tmp_path: Path):
    """Marker titles like ``NbOX2 (X = Br, I)`` keep parentheses in the target."""
    title = "Nonlinear source NbOX2 (X = Br, I)"
    markdown = (
        f"![]({title}_images/_page_2_Figure_2.jpeg)\n\n"
        "Figure 2 | Microscopy of the flake.\n"
    )
    md_path = _build_tree(
        tmp_path, markdown, {"_page_2_Figure_2.jpeg": (900, 700)}, md_name=title
    )

    verdicts = _verdicts_by_name(md_path)

    assert list(verdicts) == ["_page_2_Figure_2.jpeg"]
    assert verdicts["_page_2_Figure_2.jpeg"].valid is True
    # the target resolves: nothing is reported as a missing link
    assert prune_markdown_images(md_path).missing == 0


def test_unreadable_image_is_pruned(tmp_path: Path):
    md_path = _build_tree(
        tmp_path, f"Body.\n\n{FIGURE}\n", {"_page_4_Figure_1.jpeg": (100, 100)}
    )
    (tmp_path / "Paper_images" / "_page_4_Figure_1.jpeg").write_bytes(b"not a jpeg")

    verdicts = _verdicts_by_name(md_path)

    assert verdicts["_page_4_Figure_1.jpeg"].readable is False
    assert verdicts["_page_4_Figure_1.jpeg"].valid is False
    assert "unreadable" in verdicts["_page_4_Figure_1.jpeg"].reason


def test_directory_pass_and_json_report(tmp_path: Path):
    _build_tree(tmp_path / "tree", _sample_markdown(), _sample_images())
    quiet = tmp_path / "tree" / "NoImages.md"
    quiet.parent.mkdir(parents=True, exist_ok=True)
    quiet.write_text("Just text, no images at all.\n", encoding="utf-8")

    summary = prune_directory(tmp_path / "tree")

    assert summary["files"] == 2
    assert summary["changed_files"] == 1
    assert (summary["images"], summary["removed"]) == (4, 2)
    assert len(summary["details"]) == 1  # the Markdown without images is skipped

    report = tmp_path / "report.jsonl"
    write_report(summary, report)
    lines = [
        json.loads(line)
        for line in report.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    assert {item["name"] for item in lines[0]["removed_images"]} == {
        "_page_0_Picture_4.jpeg",
        "_page_6_Picture_2.jpeg",
    }


def test_invalid_boilerplate_regex_raises(tmp_path: Path):
    md_path = _build_tree(tmp_path, _sample_markdown(), _sample_images())

    with pytest.raises(ValueError, match="--boilerplate"):
        classify_markdown_images(md_path, extra_boilerplate=["([unclosed"])
