"""
Optis Benchmark - Tests for the meta-review input assembly and parsing

Covers both sides of ``utils/reviews_to_meta-reviews.py``: the four parts
sent per record (prompt, reviewer comments, manuscript, figures), figure order,
manuscript clean-up, the skip rules, and response parsing into the canonical
fields (strengths/weaknesses as lists of bullet points). No LLM calls.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "utils" / "reviews_to_meta-reviews.py"

TITLE = "High NA lens for lithography"


@pytest.fixture(scope="module")
def meta():
    """Load the hyphenated meta-review script as a module."""
    spec = importlib.util.spec_from_file_location("reviews_to_meta_reviews", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _image(directory: Path, name: str, size=(120, 80)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", size, "navy").save(path, "JPEG")
    return path


def _paper(tmp_path: Path, links: list[str], images: list[str], md_name: str = TITLE) -> Path:
    """Write ``<md_name>.md`` with image links plus its ``_images`` folder."""
    md_path = tmp_path / f"{md_name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"Paragraph about {name}.\n\n![]({md_name}_images/{name})" for name in links)
    md_path.write_text(body or "Plain manuscript without figures.", encoding="utf-8")
    for name in images:
        _image(md_path.with_name(f"{md_name}_images"), name)
    return md_path


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    options = {
        "reviews_dir": tmp_path / "reviews",
        "papers_dir": tmp_path / "papers",
        "no_manuscript": False,
        "no_images": False,
        "max_images": 0,
        "max_image_mb": 8.0,
    }
    options.update(overrides)
    return argparse.Namespace(**options)


class TestFindFigures:
    """Tests for figure discovery."""

    def test_follows_document_order(self, tmp_path, meta):
        md_path = _paper(
            tmp_path,
            links=["_page_5_Figure_2.jpeg", "_page_2_Figure_1.jpeg"],
            images=["_page_2_Figure_1.jpeg", "_page_5_Figure_2.jpeg", "_page_9_Figure_3.jpeg"],
        )
        figures = meta.find_figures(md_path)
        assert [path.name for path in figures] == ["_page_5_Figure_2.jpeg", "_page_2_Figure_1.jpeg"]

    def test_falls_back_to_directory_listing(self, tmp_path, meta):
        md_path = _paper(tmp_path, links=[], images=["b.jpeg", "a.jpeg"])
        figures = meta.find_figures(md_path)
        assert [path.name for path in figures] == ["a.jpeg", "b.jpeg"]

    def test_links_without_files_are_dropped(self, tmp_path, meta):
        md_path = _paper(tmp_path, links=["missing.jpeg", "present.jpeg"], images=["present.jpeg"])
        figures = meta.find_figures(md_path)
        assert [path.name for path in figures] == ["present.jpeg"]

    def test_duplicate_links_keep_first_position(self, tmp_path, meta):
        md_path = _paper(tmp_path, links=["a.jpeg", "a.jpeg"], images=["a.jpeg"])
        assert len(meta.find_figures(md_path)) == 1

    def test_missing_paper_returns_empty(self, tmp_path, meta):
        assert meta.find_figures(tmp_path / "absent.md") == []


class TestCleanManuscript:
    """Tests for the manuscript clean-up sent to the LLM."""

    def test_image_links_become_placeholders(self, tmp_path, meta):
        md_path = _paper(tmp_path, links=["_page_2_Figure_1.jpeg"], images=["_page_2_Figure_1.jpeg"])
        cleaned = meta.clean_manuscript(md_path.read_text(encoding="utf-8"))
        assert "[figure: _page_2_Figure_1.jpeg]" in cleaned
        assert "![](" not in cleaned

    def test_span_anchors_removed_text_kept(self, meta):
        raw = 'Result<span id="page-3-1">1</span>\n\n\n\nNext paragraph.'
        cleaned = meta.clean_manuscript(raw)
        assert cleaned == "Result1\n\nNext paragraph."

    def test_blank_runs_collapsed(self, meta):
        assert meta.clean_manuscript("a\n\n\n\n\nb") == "a\n\nb"


class TestBuildMessages:
    """Tests for the four-part message layout."""

    def test_all_four_parts_in_order(self, tmp_path, meta):
        figures = [_image(tmp_path / "imgs", "_page_2_Figure_1.jpeg")]
        messages = meta.build_messages("PROMPT", "REVIEWS", "MANUSCRIPT", figures)

        assert [sorted(message) for message in messages] == [
            ["prompt"],
            ["prompt"],
            ["prompt"],
            ["images", "prompt"],
        ]
        assert messages[0]["prompt"] == "PROMPT"
        assert messages[1]["prompt"].startswith("## Reviewer comments")
        assert messages[2]["prompt"].startswith("## Original manuscript")
        assert messages[3]["prompt"].startswith(meta.FIGURES_HEADING)
        assert "_page_2_Figure_1.jpeg" in messages[3]["prompt"]
        assert messages[3]["images"] == [str(figures[0])]

    def test_no_manuscript_skips_that_part(self, tmp_path, meta):
        figures = [_image(tmp_path / "imgs", "a.jpeg")]
        messages = meta.build_messages("PROMPT", "REVIEWS", None, figures)
        assert len(messages) == 3
        assert "## Original manuscript" not in messages[1]["prompt"]
        assert messages[2]["prompt"].startswith(meta.FIGURES_HEADING)

    def test_no_figures_leaves_text_only(self, meta):
        messages = meta.build_messages("PROMPT", "REVIEWS", "MANUSCRIPT", [])
        assert len(messages) == 3
        assert all("images" not in message for message in messages)


class TestSelectFigures:
    """Tests for the upload limits."""

    def test_max_images_cap(self, tmp_path, meta):
        figures = [_image(tmp_path / "imgs", f"f{i}.jpeg") for i in range(4)]
        kept, skipped = meta.select_figures(figures, max_images=2, max_image_mb=8.0)
        assert [path.name for path in kept] == ["f0.jpeg", "f1.jpeg"]
        assert [item["file"] for item in skipped] == ["f2.jpeg", "f3.jpeg"]
        assert "max-images" in skipped[0]["reason"]

    def test_oversized_images_skipped(self, tmp_path, meta):
        small = _image(tmp_path / "imgs", "small.jpeg", size=(20, 20))
        big = _image(tmp_path / "imgs", "big.jpeg", size=(600, 600))
        assert big.stat().st_size > 2 * small.stat().st_size
        limit_mb = 2 * small.stat().st_size / 1024 / 1024

        kept, skipped = meta.select_figures([small, big], max_images=0, max_image_mb=limit_mb)

        assert [path.name for path in kept] == ["small.jpeg"]
        assert skipped[0]["file"] == "big.jpeg" and "MB" in skipped[0]["reason"]

    def test_unreadable_figure_skipped(self, tmp_path, meta):
        figures = [_image(tmp_path / "imgs", "ok.jpeg"), tmp_path / "imgs" / "gone.jpeg"]
        kept, skipped = meta.select_figures(figures, max_images=0, max_image_mb=8.0)
        assert [path.name for path in kept] == ["ok.jpeg"]
        assert skipped == [{"file": "gone.jpeg", "reason": "unreadable"}]


class TestLoadReviewText:
    """Tests for the reviewer comment source selection."""

    def test_markdown_preferred(self, tmp_path, meta):
        reviews = tmp_path / "reviews"
        reviews.mkdir()
        (reviews / f"{TITLE}.md").write_text("  Reviewer one: weak.  ", encoding="utf-8")
        record = {"id": 1, "title": TITLE, "review": [{"reviewer_id": 1, "review_text": "jsonl"}]}

        text, source = meta.load_review_text(record, reviews)
        assert text == "Reviewer one: weak."
        assert source.endswith(f"{TITLE}.md")

    def test_falls_back_to_jsonl(self, tmp_path, meta):
        record = {
            "id": 2,
            "title": TITLE,
            "review": [
                {"reviewer_id": 1, "review_text": "first"},
                {"reviewer_id": 2, "review_text": "second"},
            ],
        }
        text, source = meta.load_review_text(record, tmp_path / "absent")
        assert "## Reviewer #1\nfirst" in text and "## Reviewer #2\nsecond" in text
        assert source == "<reviews jsonl>"


class TestPrepareInputs:
    """Tests for the assembled input of a whole record."""

    def _tree(self, tmp_path, meta):
        papers = tmp_path / "papers"
        reviews = tmp_path / "reviews"
        reviews.mkdir(parents=True, exist_ok=True)
        (reviews / f"{TITLE}.md").write_text("Reviewer comments body", encoding="utf-8")
        md_path = _paper(
            papers,
            links=["_page_2_Figure_1.jpeg", "_page_4_Figure_2.jpeg"],
            images=["_page_2_Figure_1.jpeg", "_page_4_Figure_2.jpeg"],
        )
        args = _args(tmp_path, reviews_dir=reviews, papers_dir=papers)
        return md_path, args, reviews

    def test_full_record(self, tmp_path, meta):
        md_path, args, reviews = self._tree(tmp_path, meta)
        record = {"id": 7, "title": TITLE, "review": [{"reviewer_id": 1, "review_text": "x"}]}

        prepared = meta.prepare_inputs(record, args, "PROMPT")

        assert prepared["reviews_source"] == str(reviews.resolve() / f"{TITLE}.md")
        assert prepared["manuscript"] == str(md_path.resolve())
        assert prepared["figures"] == ["_page_2_Figure_1.jpeg", "_page_4_Figure_2.jpeg"]
        assert prepared["skipped_images"] == []
        assert prepared["manuscript_chars"] > 0
        assert len(prepared["messages"]) == 4

    def test_missing_manuscript_keeps_reviews(self, tmp_path, meta):
        _, args, _ = self._tree(tmp_path, meta)
        args.papers_dir = tmp_path / "nowhere"
        record = {"id": 8, "title": TITLE, "review": [{"reviewer_id": 1, "review_text": "x"}]}

        prepared = meta.prepare_inputs(record, args, "PROMPT")

        assert prepared["manuscript"] is None
        assert prepared["figures"] == []
        assert len(prepared["messages"]) == 2
        assert prepared["messages"][1]["prompt"].endswith("Reviewer comments body")

    def test_no_images_flag_records_skips(self, tmp_path, meta):
        _, args, _ = self._tree(tmp_path, meta)
        args.no_images = True
        record = {"id": 9, "title": TITLE, "review": [{"reviewer_id": 1, "review_text": "x"}]}

        prepared = meta.prepare_inputs(record, args, "PROMPT")

        assert prepared["figures"] == []
        assert {item["reason"] for item in prepared["skipped_images"]} == {"--no-images"}
        assert len(prepared["messages"]) == 3

    def test_no_manuscript_still_sends_figures(self, tmp_path, meta):
        _, args, _ = self._tree(tmp_path, meta)
        args.no_manuscript = True
        record = {"id": 10, "title": TITLE, "review": [{"reviewer_id": 1, "review_text": "x"}]}

        prepared = meta.prepare_inputs(record, args, "PROMPT")

        assert prepared["manuscript_chars"] == 0
        assert len(prepared["messages"]) == 3
        assert prepared["messages"][2]["prompt"].startswith(meta.FIGURES_HEADING)


class TestToBulletedList:
    """Tests for the strengths/weaknesses bullet splitting."""

    def test_dash_star_and_numbered_bullets(self, meta):
        text = "- Novel mask design\n* High throughput\n3. Well written"
        assert meta.to_bulleted_list(text) == ["Novel mask design", "High throughput", "Well written"]

    def test_continuation_lines_merge_into_previous_point(self, meta):
        text = "- First point\n  spans two lines\n- Second point"
        assert meta.to_bulleted_list(text) == ["First point spans two lines", "Second point"]

    def test_list_elements_stay_separate(self, meta):
        assert meta.to_bulleted_list(["  Point A  ", "Point B"]) == ["Point A", "Point B"]

    def test_list_element_with_bullets_is_split(self, meta):
        assert meta.to_bulleted_list(["- a\n- b"]) == ["a", "b"]

    def test_plain_text_without_bullets_is_single_point(self, meta):
        assert meta.to_bulleted_list("Just one paragraph.") == ["Just one paragraph."]

    def test_empty_values(self, meta):
        assert meta.to_bulleted_list("") == []
        assert meta.to_bulleted_list("   \n\n  ") == []
        assert meta.to_bulleted_list([]) == []


class TestParseResponse:
    """Tests for parsing LLM responses into canonical fields."""

    MARKDOWN = (
        "## Summary of the paper\nSolid study.\n\n"
        "## Strengths of the paper\n- Novel mask design\n- High throughput\n\n"
        "## Weaknesses of the paper\n* Missing error bars on Fig. 3\n"
        "with no baseline comparison"
    )

    def test_markdown_sections_split_bullets(self, meta):
        result = meta.parse_response(self.MARKDOWN)

        assert result[meta.SUMMARY] == "Solid study."
        assert result[meta.DISPUTES] == ""
        assert result[meta.STRENGTHS] == ["Novel mask design", "High throughput"]
        assert result[meta.WEAKNESSES] == ["Missing error bars on Fig. 3 with no baseline comparison"]

    def test_json_object_with_array_fields(self, meta):
        payload = {
            "Summary of the paper": "S.",
            "Core disputes and arbitration": "No disputes.",
            "Strengths of the paper": ["Point A", "Point B"],
            "Weaknesses of the paper": ["Only one"],
        }

        result = meta.parse_response(json.dumps(payload))

        assert result[meta.STRENGTHS] == ["Point A", "Point B"]
        assert result[meta.WEAKNESSES] == ["Only one"]
        assert result[meta.DISPUTES] == "No disputes."

    def test_json_markdown_wrapper_splits_bullets(self, meta):
        payload = {"meta_review": self.MARKDOWN}

        result = meta.parse_response(json.dumps(payload))

        assert result[meta.STRENGTHS] == ["Novel mask design", "High throughput"]

    def test_missing_strengths_raises(self, meta):
        text = "## Summary of the paper\nS.\n\n## Weaknesses of the paper\n- only this"
        with pytest.raises(ValueError, match="Strengths"):
            meta.parse_response(text)

    def test_empty_strengths_raises(self, meta):
        payload = {
            "Summary of the paper": "S.",
            "Strengths of the paper": [],
            "Weaknesses of the paper": ["W"],
        }
        with pytest.raises(ValueError, match="Strengths"):
            meta.parse_response(json.dumps(payload))


class TestRecordIsDone:
    """Tests for the resume check against the list output format."""

    @staticmethod
    def _record(meta, strengths=None, summary="S."):
        return {
            "id": 1,
            "title": TITLE,
            "meta-review": {
                meta.SUMMARY: summary,
                meta.DISPUTES: "",
                meta.STRENGTHS: ["a"] if strengths is None else strengths,
                meta.WEAKNESSES: ["w"],
            },
        }

    def test_complete_list_record(self, meta):
        assert meta.record_is_done(self._record(meta))

    def test_old_string_format_is_not_done(self, meta):
        record = self._record(meta)
        record["meta-review"][meta.STRENGTHS] = "- a"
        assert not meta.record_is_done(record)

    def test_empty_strengths_list_is_not_done(self, meta):
        assert not meta.record_is_done(self._record(meta, strengths=[]))

    def test_blank_summary_is_not_done(self, meta):
        assert not meta.record_is_done(self._record(meta, summary="  "))
