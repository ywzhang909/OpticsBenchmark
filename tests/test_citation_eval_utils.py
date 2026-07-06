"""
OptiS Benchmark - Citation Evaluation Utils Tests

Tests for citation text processing functions (remove_citations, extract_citations).
GPU-dependent functions (get_max_memory, _run_nli_autoais, compute_citation_f1)
are skipped when torch/transformers are unavailable.

NOTE: Tests import citation_eval_utils inside fixtures rather than at module
level because the module has eager top-level dependencies (nltk, torch,
transformers) that may not resolve under pytest's import machinery.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(scope="module")
def citation_utils():
    """Import citation_eval_utils once per module via sys.path."""
    sys.path.insert(0, SCRIPTS_DIR)
    from algorithm.citation_eval_utils import extract_citations, remove_citations

    return remove_citations, extract_citations


class TestRemoveCitations:
    """Tests for remove_citations function."""

    def test_remove_single_citation(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations(
            "AI is evolving rapidly [1], which will impact society."
        )
        assert result == "AI is evolving rapidly, which will impact society."

    def test_remove_multiple_citations_in_one_bracket(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("Prior work [1, 2, 3] shows promising results.")
        assert result == "Prior work shows promising results."

    def test_remove_multiple_brackets(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations(
            "Method A [1] and method B [2] were compared [3]."
        )
        assert result == "Method A and method B were compared."

    def test_no_citations(self, citation_utils):
        remove_citations, _ = citation_utils
        text = "This is a plain sentence without any citations."
        result = remove_citations(text)
        assert result == text

    def test_empty_string(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("")
        assert result == ""

    def test_citation_at_start(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("[1] This is an important finding.")
        assert result == "This is an important finding."

    def test_citation_at_end(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("This is a finding [1].")
        assert result == "This is a finding."

    def test_only_citations(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("[1, 2, 3]")
        assert result == ""

    def test_fixes_period_spacing(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("End of sentence [1] . Next sentence.")
        assert result == "End of sentence. Next sentence."

    def test_fixes_comma_spacing(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("Items [1], were tested.")
        assert result == "Items, were tested."

    def test_large_citation_numbers(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations(
            "Survey [1, 2, 3, 4, 5] covers multiple topics."
        )
        assert result == "Survey covers multiple topics."

    def test_text_with_numeric_values(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations("In 2024, 95% of participants agreed [1].")
        assert result == "In 2024, 95% of participants agreed."

    def test_two_digit_citation_numbers(self, citation_utils):
        remove_citations, _ = citation_utils
        result = remove_citations(
            "Results from [10, 11, 12] confirm the hypothesis."
        )
        assert result == "Results from confirm the hypothesis."


class TestExtractCitations:
    """Tests for extract_citations function."""

    def test_extract_single_citation(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("According to [1], this is true.")
        assert result == ["[1]"]

    def test_extract_multiple_citations_in_one_bracket(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("See [1, 2, 3] for details.")
        assert result == ["[1]", "[2]", "[3]"]

    def test_extract_from_multiple_brackets(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("Method [1] and result [2] show [3].")
        assert result == ["[1]", "[2]", "[3]"]

    def test_no_citations(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("This text has no citations.")
        assert result == []

    def test_empty_string(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("")
        assert result == []

    def test_citations_with_spaces(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("Results [1,2,3] confirm.")
        assert result == ["[1]", "[2]", "[3]"]

    def test_mixed_single_and_multi(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("Study [1] and review [2, 3] agree [4].")
        assert result == ["[1]", "[2]", "[3]", "[4]"]

    def test_large_bracket(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations(
            "Survey [1, 2, 3, 10, 11] covers topics."
        )
        assert result == ["[1]", "[2]", "[3]", "[10]", "[11]"]

    def test_preserves_order(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations("Start [3] middle [1] end [2].")
        assert result == ["[3]", "[1]", "[2]"]

    def test_text_with_citation_like_numbers(self, citation_utils):
        _, extract_citations = citation_utils
        result = extract_citations(
            "2024 study [1] had 95 participants [2]."
        )
        assert result == ["[1]", "[2]"]


class TestCitationModuleImports:
    """Verify the citation module can be imported."""

    def test_imports_available(self):
        """Verify the module and function names exist."""
        sys.path.insert(0, SCRIPTS_DIR)
        from utils import citation_eval_utils as ceu

        assert hasattr(ceu, "compute_citation_f1")
        assert hasattr(ceu, "get_max_memory")
        assert hasattr(ceu, "_run_nli_autoais")
