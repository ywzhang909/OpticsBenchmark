"""
Optis Benchmark - Consistency Tests

Tests for the Pearson-correlation consistency utilities in src/algorithm/consistency.py,
the metadata cross-check in src/utils/eval_consistency.py and the loaders in
src/utils/score_io.py.
"""

import json

import pytest

from src.algorithm.consistency import evaluate_consistency, pearson_correlation
from src.utils.eval_consistency import _cross_check, load_scores
from src.utils.score_io import records_by_id, records_by_key, summarize_records

# =============================================================================
# Helpers
# =============================================================================


def _judge_record(
    entry_id: int,
    score: float,
    output_model: str = "gpt-4o",
    context: str = "薄透镜近轴成像",
    judge_model: str = "gpt-4o",
) -> dict:
    """构造一条规范裁判打分记录。"""
    return {
        "id": entry_id,
        "output_model": output_model,
        "context": context,
        "judge_model": judge_model,
        "score": score,
    }


# =============================================================================
# Classes
# =============================================================================


class TestPearsonCorrelation:
    """Tests for pearson_correlation."""

    def test_perfect_positive(self):
        """Identical sequences give r = 1."""
        assert pearson_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Inverted sequences give r = -1."""
        assert pearson_correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(
            -1.0
        )

    def test_known_value(self):
        """Classic example gives r = 0.8."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 1.0, 4.0, 3.0, 5.0]
        assert pearson_correlation(xs, ys) == pytest.approx(0.8)

    def test_too_few_samples(self):
        """A single sample leaves the correlation undefined."""
        assert pearson_correlation([0.5], [0.5]) is None

    def test_constant_input(self):
        """Zero variance leaves the correlation undefined."""
        assert pearson_correlation([0.5, 0.5, 0.5], [0.1, 0.4, 0.9]) is None
        assert pearson_correlation([0.1, 0.4, 0.9], [0.5, 0.5, 0.5]) is None

    def test_length_mismatch(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError):
            pearson_correlation([0.1, 0.2], [0.1])


class TestEvaluateConsistency:
    """Tests for evaluate_consistency."""

    def test_paired_by_id(self):
        """Entries are paired by id intersection, order independent."""
        list_a = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}, {"id": 3, "score": 0.7}]
        list_b = [{"id": 3, "score": 0.6}, {"id": 1, "score": 0.85}, {"id": 2, "score": 0.3}]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 3
        assert result["defined"] is True
        assert result["pearson_r"] == pytest.approx(1.0, abs=0.02)
        assert result["mean_a"] == pytest.approx(0.6667, abs=1e-3)

    def test_identical_lists(self):
        """Identical lists give r = 1."""
        list_a = [{"id": i, "score": i / 10} for i in range(1, 6)]
        result = evaluate_consistency(list_a, list_a)

        assert result["pearson_r"] == pytest.approx(1.0)
        assert result["num_pairs"] == 5
        assert result["unpaired_ids"] == []

    def test_disjoint_ids(self):
        """Disjoint ids leave nothing to pair, r is undefined and reported as 0."""
        list_a = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}]
        list_b = [{"id": 3, "score": 0.6}, {"id": 4, "score": 0.2}]

        result = evaluate_consistency(list_a, list_b)

        assert result["defined"] is False
        assert result["pearson_r"] == 0.0
        assert result["pearson_p"] is None
        assert result["num_pairs"] == 0
        assert result["unpaired_ids"] == [1, 2, 3, 4]

    def test_partial_overlap(self):
        """Unmatched ids are reported on both sides."""
        list_a = [{"id": 1, "score": 0.1}, {"id": 2, "score": 0.2}, {"id": 5, "score": 0.9}]
        list_b = [{"id": 1, "score": 0.3}, {"id": 2, "score": 0.4}, {"id": 6, "score": 0.8}]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 2
        assert result["num_only_a"] == 1
        assert result["num_only_b"] == 1
        assert result["unpaired_ids"] == [5, 6]

    def test_single_pair_undefined(self):
        """One paired sample is not enough for a correlation."""
        list_a = [{"id": 1, "score": 0.5}, {"id": 2, "score": 0.7}]
        list_b = [{"id": 1, "score": 0.6}, {"id": 3, "score": 0.8}]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 1
        assert result["defined"] is False
        assert result["pearson_r"] == 0.0

    def test_constant_scores_undefined(self):
        """Constant scores on one side leave the correlation undefined."""
        list_a = [{"id": i, "score": 0.5} for i in range(1, 4)]
        list_b = [{"id": i, "score": i / 10} for i in range(1, 4)]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 3
        assert result["defined"] is False
        assert "variance is 0" in result["undefined_reason"]

    def test_undefined_reason_reported(self):
        """An undefined correlation carries the reason it is undefined."""
        list_a = [{"id": 1, "score": 0.5}, {"id": 2, "score": 0.7}]
        list_b = [{"id": 1, "score": 0.6}, {"id": 9, "score": 0.8}]

        result = evaluate_consistency(list_a, list_b)

        assert result["defined"] is False
        assert "fewer than" in result["undefined_reason"]

    def test_duplicate_ids_last_wins(self):
        """Duplicate ids are deduplicated, the last occurrence wins."""
        list_a = [{"id": 1, "score": 0.1}, {"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}]
        list_b = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.4}]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 2
        assert result["mean_a"] == pytest.approx(0.7)

    def test_string_and_int_ids(self):
        """Numeric strings and int ids are both coerced."""
        list_a = [{"id": "1", "score": "0.9"}, {"id": 2, "score": 0.2}, {"id": 3, "score": 0.5}]
        list_b = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.1}, {"id": 3, "score": 0.4}]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 3
        assert result["pearson_r"] == pytest.approx(1.0)

    def test_malformed_entries_skipped(self):
        """Malformed entries are skipped instead of raising."""
        list_a = [
            {"id": 1, "score": 0.1},
            {"id": 2, "score": "not-a-number"},
            "bad-entry",
            {"id": 3},
            {"id": 4, "score": 0.4},
        ]
        list_b = [{"id": 1, "score": 0.2}, {"id": 3, "score": 0.3}, {"id": 4, "score": 0.5}]

        result = evaluate_consistency(list_a, list_b)

        assert result["num_pairs"] == 2
        assert result["unpaired_ids"] == [3]

    def test_empty_inputs(self):
        """Empty inputs produce an undefined, zero-valued result."""
        result = evaluate_consistency([], [{"id": 1, "score": 0.5}])

        assert result["num_pairs"] == 0
        assert result["defined"] is False
        assert result["pearson_r"] == 0.0
        assert result["mean_a"] == 0.0
        assert result["mean_b"] == 0.0

    def test_non_list_input(self):
        """A non-list input is handled without raising."""
        result = evaluate_consistency({"id": 1, "score": 0.5}, [])

        assert result["num_pairs"] == 0
        assert result["defined"] is False


class TestLoadScores:
    """Tests for the load_scores reader."""

    def test_load_jsonl(self, tmp_path):
        """JSONL with one record per line is loaded."""
        path = tmp_path / "scores.jsonl"
        path.write_text(
            '{"id": 1, "score": 0.9}\n\n{"id": 2, "score": 0.4}\n',
            encoding="utf-8",
        )

        loaded = load_scores(path)

        assert loaded == [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}]

    def test_load_json_list(self, tmp_path):
        """A JSON top-level list is loaded."""
        path = tmp_path / "scores.json"
        path.write_text(
            json.dumps([{"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}]),
            encoding="utf-8",
        )

        loaded = load_scores(path)

        assert len(loaded) == 2
        assert loaded[0]["id"] == 1

    def test_load_json_wrapped(self, tmp_path):
        """A JSON object wrapping a list under 'scores' is unwrapped."""
        path = tmp_path / "scores.json"
        path.write_text(
            json.dumps({"scores": [{"id": 1, "score": 0.9}]}),
            encoding="utf-8",
        )

        loaded = load_scores(path)

        assert loaded == [{"id": 1, "score": 0.9}]

    def test_load_eval_results_records(self, tmp_path):
        """Evaluation result records are converted to id/score pairs."""
        path = tmp_path / "eval_results.jsonl"
        path.write_text(
            '{"evaluator": "exact_match", "task_id": 1, '
            '"metrics": {"exact_match_avg": 1.0, "num_entries": 8}}\n'
            '{"evaluator": "exact_match", "task_id": 2, '
            '"metrics": {"exact_match_avg": 0.0, "num_entries": 8}}\n',
            encoding="utf-8",
        )

        loaded = load_scores(path)

        assert loaded == [{"id": 1, "score": 1.0}, {"id": 2, "score": 0.0}]

    def test_load_eval_results_with_metric(self, tmp_path):
        """The metric argument selects which metric is extracted."""
        path = tmp_path / "eval_results.jsonl"
        path.write_text(
            '{"task_id": 1, "metrics": {"rouge1": 0.7, "rouge2": 0.2}}\n',
            encoding="utf-8",
        )

        loaded = load_scores(path, metric="rouge2")

        assert loaded == [{"id": 1, "score": 0.2}]

    def test_missing_file(self, tmp_path):
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_scores(tmp_path / "absent.json")

    def test_invalid_jsonl_line(self, tmp_path):
        """A broken JSONL line raises ValueError."""
        path = tmp_path / "broken.jsonl"
        path.write_text('{"id": 1, "score": 0.5}\n{not json}\n', encoding="utf-8")

        with pytest.raises(ValueError):
            load_scores(path)

    def test_unsupported_json_structure(self, tmp_path):
        """A JSON object without a list field raises ValueError."""
        path = tmp_path / "unsupported.json"
        path.write_text(json.dumps({"pearson_r": 0.9}), encoding="utf-8")

        with pytest.raises(ValueError):
            load_scores(path)

    def test_judge_records_keep_metadata(self, tmp_path):
        """Judge records pass through with every metadata field preserved."""
        record = _judge_record(
            7, 4.2, output_model="qwen2.5-72b-instruct", judge_model="claude-3-5-sonnet"
        )
        path = tmp_path / "judge.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        loaded = load_scores(path)

        assert loaded == [record]

    def test_judge_records_evaluated_by_id_and_score(self, tmp_path):
        """Metadata fields do not disturb the correlation computation."""
        path = tmp_path / "a.jsonl"
        path.write_text(
            json.dumps(_judge_record(1, 1.0), ensure_ascii=False)
            + "\n"
            + json.dumps(_judge_record(2, 2.0), ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        other = tmp_path / "b.jsonl"
        other.write_text(
            json.dumps(_judge_record(1, 2.0, judge_model="claude-3-5-sonnet"), ensure_ascii=False)
            + "\n"
            + json.dumps(_judge_record(2, 4.0, judge_model="claude-3-5-sonnet"), ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        result = evaluate_consistency(load_scores(path), load_scores(other))

        assert result["num_pairs"] == 2
        assert result["pearson_r"] == pytest.approx(1.0)


class TestRecordsById:
    """Tests for the records_by_id index."""

    def test_keeps_full_record(self):
        """The index stores the whole record, not just id and score."""
        record = _judge_record(3, 2.5, context="光栅衍射方程")

        indexed = records_by_id([record])

        assert indexed == {3: record}

    def test_skips_invalid_entries(self):
        """Non-dict entries and unparsable ids are skipped."""
        indexed = records_by_id([_judge_record(1, 0.5), "bad", {"score": 0.9}, {"id": "x"}])

        assert indexed == {1: indexed[1]}

    def test_duplicate_id_last_wins(self):
        """A repeated id keeps the last occurrence, as in extract_scores."""
        first = _judge_record(1, 0.1, judge_model="judge-a")
        second = _judge_record(1, 0.9, judge_model="judge-b")

        indexed = records_by_id([first, second])

        assert indexed[1]["judge_model"] == "judge-b"

    def test_non_list_input(self):
        """A non-list input yields an empty index instead of raising."""
        assert records_by_id({"id": 1, "score": 0.5}) == {}


class TestRecordsByKey:
    """Tests for the records_by_key two-level index."""

    def test_indexes_by_id_and_field(self):
        """Repeated ids are grouped by the secondary field."""
        records = [
            _judge_record(1, 0.5, output_model="gpt-4o"),
            _judge_record(1, 0.6, output_model="qwen"),
            _judge_record(2, 0.7, output_model="gpt-4o"),
        ]

        indexed = records_by_key(records, "output_model")

        assert set(indexed) == {1, 2}
        assert set(indexed[1]) == {"gpt-4o", "qwen"}
        assert indexed[1]["qwen"]["score"] == 0.6
        assert indexed[2]["gpt-4o"]["score"] == 0.7

    def test_keeps_full_record(self):
        """The index stores the whole record, not just the key and score."""
        record = _judge_record(3, 2.5, context="光栅衍射方程")

        indexed = records_by_key([record], "output_model")

        assert indexed == {3: {"gpt-4o": record}}

    def test_duplicate_id_and_key_last_wins(self):
        """A repeated (id, field) pair keeps the last occurrence."""
        first = _judge_record(1, 0.1, output_model="gpt-4o")
        second = _judge_record(1, 0.9, output_model="gpt-4o")

        indexed = records_by_key([first, second], "output_model")

        assert indexed[1]["gpt-4o"]["score"] == 0.9

    def test_skips_invalid_entries(self):
        """Non-dict entries, unparsable ids and blank keys are skipped."""
        indexed = records_by_key(
            [
                "bad",
                {"score": 0.9},
                {"id": "x", "output_model": "gpt-4o"},
                _judge_record(1, 0.5, output_model="  "),
            ],
            "output_model",
        )

        assert indexed == {}

    def test_string_ids_converted(self):
        """String ids are converted to int, matching records_by_id."""
        indexed = records_by_key(
            [{"id": "7", "output_model": "gpt-4o", "score": 0.5}], "output_model"
        )

        assert indexed == {7: {"gpt-4o": {"id": "7", "output_model": "gpt-4o", "score": 0.5}}}

    def test_non_list_input(self):
        """A non-list input yields an empty index instead of raising."""
        assert records_by_key({"id": 1, "output_model": "gpt-4o"}, "output_model") == {}


class TestSummarizeRecords:
    """Tests for the summarize_records metadata summary."""

    def test_collects_deduplicated_values(self):
        """Judge and output model names are deduplicated and sorted."""
        records = [
            _judge_record(1, 0.5, output_model="gpt-4o", judge_model="judge-a"),
            _judge_record(2, 0.6, output_model="gpt-4o", judge_model="judge-a"),
            _judge_record(3, 0.7, output_model="qwen2.5-72b-instruct", judge_model="judge-a"),
        ]

        summary = summarize_records(records)

        assert summary["num_records"] == 3
        assert summary["judge_models"] == ["judge-a"]
        assert summary["output_models"] == ["gpt-4o", "qwen2.5-72b-instruct"]
        assert summary["has_context"] is True

    def test_legacy_records_have_no_metadata(self):
        """Legacy {id, score} records report empty metadata instead of failing."""
        summary = summarize_records([{"id": 1, "score": 0.5}, {"id": 2, "score": 0.6}])

        assert summary["num_records"] == 2
        assert summary["judge_models"] == []
        assert summary["output_models"] == []
        assert summary["has_context"] is False

    def test_blank_values_ignored(self):
        """Empty-string metadata values are treated as absent."""
        record = _judge_record(1, 0.5, output_model="", context="")
        record["judge_model"] = "  "

        summary = summarize_records([record])

        assert summary["judge_models"] == []
        assert summary["output_models"] == []
        assert summary["has_context"] is False

    def test_repeated_ids_counted_by_key(self):
        """A grid file is counted by (id, field), not by id."""
        records = [
            _judge_record(1, 0.5, output_model="gpt-4o"),
            _judge_record(1, 0.6, output_model="qwen"),
            _judge_record(2, 0.7, output_model="gpt-4o"),
            _judge_record(2, 0.8, output_model="qwen"),
        ]

        summary = summarize_records(records, key="output_model")

        assert summary["num_records"] == 4
        assert summary["output_models"] == ["gpt-4o", "qwen"]

    def test_repeated_ids_counted_once_by_id(self):
        """Without a key the id-index keeps one record per id."""
        records = [
            _judge_record(1, 0.5, output_model="gpt-4o"),
            _judge_record(1, 0.6, output_model="qwen"),
        ]

        summary = summarize_records(records)

        assert summary["num_records"] == 1
        assert summary["output_models"] == ["gpt-4o", "qwen"]

    def test_judge_models_survive_collapsed_cells(self):
        """Judge names are collected from every record, not from the index."""
        records = [
            _judge_record(1, 0.5, judge_model="judge-a"),
            _judge_record(1, 0.6, judge_model="judge-b"),
        ]

        summary = summarize_records(records, key="output_model")

        assert summary["judge_models"] == ["judge-a", "judge-b"]


class TestCrossCheck:
    """Tests for the metadata cross-check between two score files."""

    def test_matching_metadata_is_clean(self):
        """Two judges agreeing on metadata produce no mismatches."""
        list_a = [
            _judge_record(1, 0.5, judge_model="judge-a"),
            _judge_record(2, 0.6, judge_model="judge-a"),
        ]
        list_b = [
            _judge_record(1, 0.55, judge_model="judge-b"),
            _judge_record(2, 0.65, judge_model="judge-b"),
        ]

        meta = _cross_check(list_a, list_b)

        assert meta["same_judge_model"] is False
        assert meta["output_model_mismatch_ids"] == []
        assert meta["context_mismatch_ids"] == []

    def test_same_judge_model_flagged(self):
        """Comparing a judge against itself is reported."""
        list_a = [_judge_record(1, 0.5, judge_model="judge-a")]
        list_b = [_judge_record(1, 0.6, judge_model="judge-a")]

        meta = _cross_check(list_a, list_b)

        assert meta["same_judge_model"] is True

    def test_output_model_mismatch_detected(self):
        """Ids scored as different output models are listed."""
        list_a = [
            _judge_record(1, 0.5, output_model="gpt-4o", judge_model="judge-a"),
            _judge_record(2, 0.6, output_model="gpt-4o", judge_model="judge-a"),
        ]
        list_b = [
            _judge_record(1, 0.5, output_model="gpt-4o", judge_model="judge-b"),
            _judge_record(2, 0.6, output_model="qwen2.5-72b-instruct", judge_model="judge-b"),
        ]

        meta = _cross_check(list_a, list_b)

        assert meta["output_model_mismatch_ids"] == [2]

    def test_context_mismatch_detected(self):
        """Ids whose judges saw different material are listed."""
        list_a = [
            _judge_record(1, 0.5, context="第一段材料", judge_model="judge-a"),
            _judge_record(2, 0.6, context="第二段材料", judge_model="judge-a"),
        ]
        list_b = [
            _judge_record(1, 0.5, context="第一段材料", judge_model="judge-b"),
            _judge_record(2, 0.6, context="被改写的材料", judge_model="judge-b"),
        ]

        meta = _cross_check(list_a, list_b)

        assert meta["context_mismatch_ids"] == [2]

    def test_unpaired_ids_not_compared(self):
        """Only ids present in both files are compared field by field."""
        list_a = [_judge_record(1, 0.5, judge_model="judge-a"), _judge_record(2, 0.6)]
        list_b = [_judge_record(1, 0.5, judge_model="judge-b")]

        meta = _cross_check(list_a, list_b)

        assert meta["output_model_mismatch_ids"] == []
        assert meta["context_mismatch_ids"] == []

    def test_legacy_files_not_flagged(self):
        """Legacy {id, score} files are neither flagged nor crash the check."""
        list_a = [{"id": 1, "score": 0.5}, {"id": 2, "score": 0.6}]
        list_b = [{"id": 1, "score": 0.55}, {"id": 2, "score": 0.65}]

        meta = _cross_check(list_a, list_b)

        assert meta["same_judge_model"] is False
        assert meta["a"]["judge_models"] == []
        assert meta["a"]["has_context"] is False
        assert meta["output_model_mismatch_ids"] == []
        assert meta["context_mismatch_ids"] == []
