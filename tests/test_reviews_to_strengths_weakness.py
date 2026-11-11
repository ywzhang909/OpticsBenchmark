"""
Optis Benchmark - Tests for the strengths/weaknesses extraction script

Covers ``utils/reviews_to_strengths_weakness.py``: review text assembly from
``reviews_data.jsonl`` records, XML-tagged message layout, strict-JSON parsing
and the schema validation rules (quote prefixes, stance enum, known reviewers).
No LLM calls.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "utils" / "reviews_to_strengths_weakness.py"

TITLE = "High NA lens for lithography"


@pytest.fixture(scope="module")
def sw():
    """Load the strengths/weaknesses script as a module."""
    spec = importlib.util.spec_from_file_location("reviews_to_strengths_weakness", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _analysis(**overrides):
    base = {
        "strengths": [{"content": "Novel mask design.", "quotes": ['Reviewer #1: "a novel mask design"']}],
        "weaknesses": [{"content": "No error bars.", "quotes": ['Reviewer #2: "no error bars are shown"']}],
        "disputes": [
            {
                "topic": "Novelty of the method",
                "positions": [
                    {"reviewer": "Reviewer #1", "stance": "positive", "viewpoint": '"novel strategy"'},
                    {"reviewer": "Reviewer #2", "stance": "negative", "viewpoint": '"identical to FLICE"'},
                ],
            }
        ],
    }
    base.update(overrides)
    return base


def _image(directory: Path, name: str, size=(120, 80)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", size, "navy").save(path, "JPEG")
    return path


class TestBuildReviewText:
    """Tests for the JSONL-only review assembly."""

    def test_sections_and_ids(self, sw):
        record = {
            "id": 1,
            "title": TITLE,
            "review": [
                {"reviewer_id": 1, "review_text": " first ", "score": 3},
                {"reviewer_id": 3, "review_text": "third"},
            ],
        }
        text, ids = sw.build_review_text(record)
        assert ids == [1, 3]
        assert text == "## Reviewer #1\nfirst\n\n## Reviewer #3\nthird"

    def test_missing_id_falls_back_to_position(self, sw):
        record = {"id": 2, "title": TITLE, "review": [{"review_text": "a"}, {"reviewer_id": "x", "review_text": "b"}]}
        text, ids = sw.build_review_text(record)
        assert ids == [1, 2]
        assert text.startswith("## Reviewer #1\na")


class TestParseResponse:
    """Tests for strict-JSON parsing and schema validation."""

    def test_valid_payload(self, sw):
        result = sw.parse_response(json.dumps(_analysis()), [1, 2, 3])
        assert result["strengths"][0]["content"] == "Novel mask design."
        assert result["disputes"][0]["positions"][1]["stance"] == "negative"

    def test_code_fence_is_stripped(self, sw):
        raw = "```json\n" + json.dumps(_analysis()) + "\n```"
        assert sw.parse_response(raw)["weaknesses"][0]["quotes"]

    def test_double_encoded_wrapper(self, sw):
        payload = {"answer": json.dumps(_analysis())}
        assert sw.parse_response(json.dumps(payload))["strengths"]

    def test_case_insensitive_keys(self, sw):
        analysis = _analysis()
        payload = {key.upper(): value for key, value in analysis.items()}
        assert sw.parse_response(json.dumps(payload))["disputes"]

    def test_markdown_response_rejected(self, sw):
        with pytest.raises(ValueError, match="not valid JSON"):
            sw.parse_response("## Strengths\n- a point")

    def test_missing_fields_rejected(self, sw):
        with pytest.raises(ValueError, match="none of the required fields"):
            sw.parse_response(json.dumps({"summary": "only that"}))

    def test_quote_prefix_required(self, sw):
        analysis = _analysis(
            strengths=[{"content": "Good.", "quotes": ["a novel mask design"]}],
        )
        with pytest.raises(ValueError, match="prefix"):
            sw.parse_response(json.dumps(analysis))

    def test_empty_quotes_rejected(self, sw):
        analysis = _analysis(weaknesses=[{"content": "Bad.", "quotes": []}])
        with pytest.raises(ValueError, match="non-empty array"):
            sw.parse_response(json.dumps(analysis))

    def test_quote_reviewer_membership(self, sw):
        analysis = json.dumps(_analysis())
        # #1 and #2 exist -> accepted; omit #2 -> the weaknesses quote fails.
        assert sw.parse_response(analysis, [1, 2])["strengths"]
        with pytest.raises(ValueError, match="unknown reviewer"):
            sw.parse_response(analysis, [1])
        # Without a reviewer list (resume re-validation) only the format is checked.
        assert sw.parse_response(analysis)["weaknesses"]

    def test_both_empty_lists_rejected(self, sw):
        analysis = _analysis(strengths=[], weaknesses=[])
        with pytest.raises(ValueError, match="Both strengths and weaknesses are empty"):
            sw.parse_response(json.dumps(analysis))

    def test_empty_disputes_allowed(self, sw):
        result = sw.parse_response(json.dumps(_analysis(disputes=[])))
        assert result["disputes"] == []

    def test_stance_enum_normalized(self, sw):
        analysis = _analysis()
        analysis["disputes"][0]["positions"][0]["stance"] = " Positive "
        result = sw.parse_response(json.dumps(analysis))
        assert result["disputes"][0]["positions"][0]["stance"] == "positive"

    def test_invalid_stance_rejected(self, sw):
        analysis = _analysis()
        analysis["disputes"][0]["positions"][0]["stance"] = "neutral"
        with pytest.raises(ValueError, match="stance"):
            sw.parse_response(json.dumps(analysis))

    def test_reviewer_format_and_membership(self, sw):
        analysis = _analysis()
        analysis["disputes"][0]["positions"][0]["reviewer"] = " reviewer #2 "
        result = sw.parse_response(json.dumps(analysis), [1, 2])
        assert result["disputes"][0]["positions"][0]["reviewer"] == "Reviewer #2"

        analysis["disputes"][0]["positions"][0]["reviewer"] = "Reviewer #7"
        with pytest.raises(ValueError, match="unknown reviewer"):
            sw.parse_response(json.dumps(analysis), [1, 2])

    def test_empty_viewpoint_rejected(self, sw):
        analysis = _analysis()
        analysis["disputes"][0]["positions"][1]["viewpoint"] = "  "
        with pytest.raises(ValueError, match="viewpoint"):
            sw.parse_response(json.dumps(analysis))


class TestRecordIsDone:
    """Tests for the resume check."""

    def test_valid_record(self, sw):
        record = {"id": 1, "title": TITLE, sw.RESULT_KEY: _analysis()}
        assert sw.record_is_done(record)

    def test_missing_key(self, sw):
        assert not sw.record_is_done({"id": 1, "title": TITLE})

    def test_invalid_analysis_not_done(self, sw):
        analysis = _analysis()
        analysis["strengths"][0]["quotes"] = ["no prefix"]
        record = {"id": 1, "title": TITLE, sw.RESULT_KEY: analysis}
        assert not sw.record_is_done(record)


class TestBuildMessages:
    """Tests for the XML-tagged message layout."""

    def test_all_parts_in_order(self, tmp_path, sw):
        figures = [_image(tmp_path / "imgs", "_page_2_Figure_1.jpeg")]
        messages = sw.build_messages("PROMPT", "REVIEWS", "MANUSCRIPT", figures)

        assert [sorted(message) for message in messages] == [
            ["prompt"],
            ["prompt"],
            ["prompt"],
            ["images", "prompt"],
        ]
        assert messages[1]["prompt"] == "<review_comments>\nREVIEWS\n</review_comments>"
        assert messages[2]["prompt"] == "<paper_manuscript>\nMANUSCRIPT\n</paper_manuscript>"
        assert messages[3]["prompt"].startswith("<paper_figures>")
        assert "_page_2_Figure_1.jpeg" in messages[3]["prompt"]
        assert messages[3]["prompt"].rstrip().endswith("</paper_figures>")
        assert messages[3]["images"] == [str(figures[0])]

    def test_optional_parts(self, tmp_path, sw):
        figures = [_image(tmp_path / "imgs", "a.jpeg")]
        assert len(sw.build_messages("P", "R", None, figures)) == 3
        assert len(sw.build_messages("P", "R", "M", [])) == 3
        assert len(sw.build_messages("P", "R", None, [])) == 2


class TestPrepareInputs:
    """Tests for the assembled input of a whole record."""

    def _record(self):
        return {
            "id": 7,
            "title": TITLE,
            "review": [
                {"reviewer_id": 1, "review_text": "first"},
                {"reviewer_id": 2, "review_text": "second"},
            ],
        }

    def _args(self, tmp_path, sw, **overrides):
        import argparse

        papers = tmp_path / "papers"
        reviews_dir = papers.parent / "reviews"  # unused: reviews come from JSONL
        options = {
            "papers_dir": papers,
            "no_manuscript": False,
            "no_images": False,
            "max_images": 0,
            "max_image_mb": 8.0,
        }
        options.update(overrides)
        return argparse.Namespace(**options), papers, reviews_dir

    def test_full_record(self, tmp_path, sw):
        args, papers, _ = self._args(tmp_path, sw)
        md_path = papers / f"{TITLE}.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("Text.\n\n![](x_images/_page_2_Figure_1.jpeg)", encoding="utf-8")
        _image(md_path.with_name(f"{TITLE}_images"), "_page_2_Figure_1.jpeg")

        prepared = sw.prepare_inputs(self._record(), args, "PROMPT")

        assert prepared["reviewer_ids"] == [1, 2]
        assert prepared["manuscript"] == str(md_path.resolve())
        assert prepared["figures"] == ["_page_2_Figure_1.jpeg"]
        assert prepared["skipped_images"] == []
        assert prepared["manuscript_chars"] > 0
        assert len(prepared["messages"]) == 4
        assert "[figure: _page_2_Figure_1.jpeg]" in prepared["messages"][2]["prompt"]

    def test_missing_manuscript_keeps_reviews(self, tmp_path, sw):
        args, _, _ = self._args(tmp_path, sw)
        args.papers_dir = tmp_path / "nowhere"

        prepared = sw.prepare_inputs(self._record(), args, "PROMPT")

        assert prepared["manuscript"] is None
        assert prepared["figures"] == []
        assert len(prepared["messages"]) == 2

    def test_no_images_records_skips(self, tmp_path, sw):
        args, papers, _ = self._args(tmp_path, sw)
        md_path = papers / f"{TITLE}.md"
        md_path.parent.mkdir(parents=True)
        md_path.write_text("![](x_images/a.jpeg)", encoding="utf-8")
        _image(md_path.with_name(f"{TITLE}_images"), "a.jpeg")
        args.no_images = True

        prepared = sw.prepare_inputs(self._record(), args, "PROMPT")

        assert prepared["figures"] == []
        assert {item["reason"] for item in prepared["skipped_images"]} == {"--no-images"}
