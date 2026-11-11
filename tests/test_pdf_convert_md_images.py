"""
Optis Benchmark - Tests for marker image extraction in pdf_convert_md_marker

Covers ``_save_images``: images marker crops out of a document are written into
a per-document ``<md stem>_images/`` folder and only the saved images get their
Markdown links rewritten.
"""

from pathlib import Path

from PIL import Image

from src.utils.pdf_convert_md_marker import _save_images


class _BrokenImage:
    """Stands in for a crop the encoder cannot write (simulates a save error)."""

    def save(self, *args, **kwargs):
        raise OSError("simulated encoder failure")


def _image(mode: str = "RGB", color: str = "red") -> Image.Image:
    return Image.new(mode, (40, 30), color)


def test_no_images_returns_markdown_unchanged(tmp_path: Path):
    markdown = "# Paper\n\nbody\n"

    text, image_dir, saved, failed = _save_images({}, markdown, tmp_path / "Paper.md")

    assert (text, image_dir, saved, failed) == (markdown, "", 0, [])
    assert not (tmp_path / "Paper_images").exists()


def test_saves_images_and_rewrites_links(tmp_path: Path):
    markdown = (
        "# Paper\n\n"
        "![](_page_1_Picture_1.jpeg)\n\n"
        "![](_page_20_Figure_8.jpeg)\n"
        "![](_page_20_Figure_80.jpeg)\n"
    )
    images = {
        "_page_1_Picture_1.jpeg": _image(),
        "_page_20_Figure_8.jpeg": _image(color="green"),
        # RGBA crops have to be converted before JPEG saving
        "_page_20_Figure_80.jpeg": _image(mode="RGBA"),
    }

    text, image_dir, saved, failed = _save_images(
        images, markdown, tmp_path / "Some Paper.md"
    )

    assert image_dir == "Some Paper_images"
    assert (saved, failed) == (3, [])
    assert sorted(p.name for p in (tmp_path / image_dir).iterdir()) == sorted(images)
    for name in images:
        assert f"]({image_dir}/{name})" in text
    assert "](_page_" not in text
    # the 8 / 80 prefix pair must not shadow each other
    assert f"]({image_dir}/_page_20_Figure_8.jpeg)" in text
    with Image.open(tmp_path / image_dir / "_page_20_Figure_80.jpeg") as img:
        assert img.format == "JPEG"


def test_failed_image_keeps_original_link(tmp_path: Path):
    markdown = "![](_page_3_Figure_1.jpeg)\n![](_page_3_Figure_2.jpeg)\n"
    images = {
        "_page_3_Figure_1.jpeg": _BrokenImage(),
        "_page_3_Figure_2.jpeg": _image(color="blue"),
    }

    text, image_dir, saved, failed = _save_images(
        images, markdown, tmp_path / "Paper.md"
    )

    assert saved == 1
    assert failed == ["_page_3_Figure_1.jpeg"]
    assert f"]({image_dir}/_page_3_Figure_2.jpeg)" in text
    assert "](_page_3_Figure_1.jpeg)" in text
    assert not (tmp_path / image_dir / "_page_3_Figure_1.jpeg").exists()


def test_all_images_fail_leaves_no_empty_folder(tmp_path: Path):
    markdown = "![](_page_3_Figure_1.jpeg)\n"

    text, image_dir, saved, failed = _save_images(
        {"_page_3_Figure_1.jpeg": _BrokenImage()}, markdown, tmp_path / "Paper.md"
    )

    assert (text, image_dir, saved) == (markdown, "", 0)
    assert failed == ["_page_3_Figure_1.jpeg"]
    assert not (tmp_path / "Paper_images").exists()
