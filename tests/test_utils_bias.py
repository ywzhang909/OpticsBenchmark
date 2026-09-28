"""
Optis Benchmark - Bias Tests

Tests for the self-preference bias metric in src/algorithm/bias.py and the
single-file command-line entry point in src/utils/eval_bias.py.

The input is one judge's scores over a full grid: n samples x m model outputs,
paired by (id, output_model).
"""

import json

import pytest

from src.algorithm.bias import evaluate_judge_bias
from src.utils.eval_bias import _format_table, _log_report, main
from src.utils.logger import logger as loguru_logger

MODELS = ["gpt-4o", "qwen", "gemini"]


def grid(
    sample_scores: dict[int, dict[str, float]],
    judge_model: str = "gpt-4o",
    context: str = "薄透镜近轴成像",
) -> list[dict]:
    """把 ``{id: {output_model: score}}`` 展开成裁判打分记录列表。"""
    return [
        {
            "id": entry_id,
            "output_model": output_model,
            "context": context,
            "judge_model": judge_model,
            "score": score,
        }
        for entry_id, per_model in sample_scores.items()
        for output_model, score in per_model.items()
    ]


def write_records(path, records: list[dict]) -> str:
    """把记录写成 jsonl 并返回路径字符串。"""
    body = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(body + "\n", encoding="utf-8")
    return str(path)


def symmetric_grid(offset: float = 0.0) -> dict[int, dict[str, float]]:
    """构造自身输出与他人同分、或被抬高 ``offset`` 分的 n=2 网格。"""
    return {
        1: {"gpt-4o": 3.0 + offset, "qwen": 3.0, "gemini": 3.0},
        2: {"gpt-4o": 4.0 + offset, "qwen": 4.0, "gemini": 4.0},
    }


# =============================================================================
# Classes
# =============================================================================


class TestEvaluateJudgeBias:
    """Tests for evaluate_judge_bias."""

    def test_no_bias(self):
        """A judge scoring its own output like the others has zero bias."""
        result = evaluate_judge_bias(grid(symmetric_grid()))

        assert result["defined"] is True
        assert result["undefined_reason"] == ""
        assert result["num_samples"] == 2
        assert result["num_defined"] == 2
        assert result["num_undefined"] == 0
        assert result["mean_error_rate_se"] == pytest.approx(0.0, abs=1e-9)
        assert result["median_error_rate_se"] == pytest.approx(0.0, abs=1e-9)

    def test_positive_bias(self):
        """A judge inflating its own output by 20% shows +0.2."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))

        # 3.6/3 - 1 = 0.2 and 4.6/4 - 1 = 0.15
        assert result["mean_error_rate_se"] == pytest.approx(0.175)
        assert result["median_error_rate_se"] == pytest.approx(0.175)

    def test_negative_bias(self):
        """A judge scoring its own output lower shows negative bias."""
        result = evaluate_judge_bias(grid({1: {"gpt-4o": 1.0, "qwen": 2.0, "gemini": 2.0}}))

        assert result["mean_error_rate_se"] == pytest.approx(-0.5)

    def test_per_sample_values(self):
        """Per-sample y_self / y_other / ErrorRate_SE are reported."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))
        sample = result["per_sample"][0]

        assert sample["id"] == 1
        assert sample["output_model"] == "gpt-4o"
        assert sample["y_self"] == pytest.approx(3.6)
        assert sample["y_other"] == pytest.approx(3.0)
        assert sample["error_rate_se"] == pytest.approx(0.2)

    def test_y_other_is_mean_of_other_models(self):
        """y_other averages the judge's scores on the other m-1 outputs."""
        result = evaluate_judge_bias(grid({1: {"gpt-4o": 4.0, "qwen": 1.0, "gemini": 4.0}}))

        # (1.0 + 4.0) / 2 = 2.5
        assert result["per_sample"][0]["y_other"] == pytest.approx(2.5)
        assert result["per_sample"][0]["error_rate_se"] == pytest.approx(0.6)

    def test_one_bias_value_per_sample(self):
        """A full n x m grid yields exactly n bias values."""
        sample_scores = {sample_id: dict.fromkeys(MODELS, 2.0) for sample_id in range(1, 6)}
        sample_scores[3]["gpt-4o"] = 3.0

        result = evaluate_judge_bias(grid(sample_scores))

        assert result["num_samples"] == 5
        assert result["num_defined"] == 5
        assert len(result["per_sample"]) == 5
        assert [s["id"] for s in result["per_sample"]] == [1, 2, 3, 4, 5]
        assert result["per_sample"][2]["error_rate_se"] == pytest.approx(0.5)
        assert result["mean_error_rate_se"] == pytest.approx(0.1)

    def test_aggregates_reported(self):
        """Mean/median of the bias and of y_self / y_other are reported."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))

        assert result["mean_y_self"] == pytest.approx(4.1)
        assert result["mean_y_other"] == pytest.approx(3.5)
        assert result["mean_error_rate_se"] == pytest.approx(result["median_error_rate_se"])

    def test_output_models_and_counts(self):
        """The m output models and record counts are reported."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))

        assert result["output_models"] == ["gemini", "gpt-4o", "qwen"]
        assert result["judge_model"] == "gpt-4o"
        assert result["self_output_model"] == "gpt-4o"
        assert result["num_records"] == 6
        assert result["num_scored"] == 6
        assert result["num_invalid"] == 0
        assert result["incomplete_ids"] == []

    def test_missing_own_output_undefined(self):
        """A sample without the judge's own output leaves that sample undefined."""
        records = grid(
            {
                1: {"gpt-4o": 3.0, "qwen": 2.0, "gemini": 2.0},
                2: {"qwen": 3.0, "gemini": 3.0},
            }
        )

        result = evaluate_judge_bias(records)

        assert result["num_samples"] == 2
        assert result["num_defined"] == 1
        assert result["num_undefined"] == 1
        assert result["undefined_ids"] == [2]
        assert result["incomplete_ids"] == [2]

    def test_no_other_model_undefined(self):
        """A sample scored only by its own output has no y_other."""
        records = grid(
            {
                1: {"gpt-4o": 3.0},
                2: {"gpt-4o": 3.0, "qwen": 2.0, "gemini": 2.0},
            }
        )

        result = evaluate_judge_bias(records)

        assert result["undefined_ids"] == [1]
        assert result["num_defined"] == 1

    def test_zero_y_other_undefined(self):
        """A zero y_other leaves the ratio undefined."""
        result = evaluate_judge_bias(grid({1: {"gpt-4o": 1.0, "qwen": 0.0, "gemini": 0.0}}))

        assert result["defined"] is False
        assert result["num_defined"] == 0
        assert result["per_sample"] == []
        assert "no valid sample" in result["undefined_reason"]

    def test_partial_grid_keeps_other_samples(self):
        """One broken sample does not invalidate the whole grid."""
        records = grid(
            {
                1: {"gpt-4o": 3.0, "qwen": 2.0, "gemini": 2.0},
                2: {"gpt-4o": 6.0, "qwen": 2.0, "gemini": 2.0},
            }
        )

        result = evaluate_judge_bias(records)

        assert result["defined"] is True
        assert result["num_defined"] == 2
        assert result["mean_error_rate_se"] == pytest.approx((0.5 + 2.0) / 2)

    def test_duplicate_id_and_model_last_wins(self):
        """A repeated (id, output_model) pair keeps the last occurrence."""
        records = grid(
            {
                1: {"gpt-4o": 3.0, "qwen": 2.0, "gemini": 2.0},
                2: {"gpt-4o": 4.0, "qwen": 3.0, "gemini": 3.0},
            }
        )
        records.append(dict(records[0], score=0.0))

        result = evaluate_judge_bias(records)

        assert result["num_records"] == 7
        assert result["per_sample"][0]["y_self"] == pytest.approx(0.0)
        assert result["per_sample"][0]["error_rate_se"] == pytest.approx(-1.0)

    def test_invalid_scores_skipped(self):
        """Unparsable scores are dropped, the rest still evaluate."""
        records = grid(
            {
                1: {"gpt-4o": 3.0, "qwen": 2.0, "gemini": 2.0},
                2: {"gpt-4o": 4.0, "qwen": "not-a-number", "gemini": 3.0},
            }
        )
        records.append({"id": 3, "output_model": "gpt-4o", "judge_model": "gpt-4o"})
        records.append({"id": 3, "judge_model": "gpt-4o", "score": 9.0})
        records.append("bad")

        result = evaluate_judge_bias(records)

        assert result["num_records"] == 9
        assert result["num_invalid"] == 2
        assert result["num_scored"] == 5
        assert result["num_defined"] == 2

    def test_judge_from_records(self):
        """The judge name is taken from the records when not given."""
        result = evaluate_judge_bias(
            grid({1: {"qwen": 3.0, "gemini": 2.0, "gpt-4o": 2.0}}, judge_model="gemini")
        )

        assert result["judge_model"] == "gemini"
        assert result["per_sample"][0]["y_self"] == pytest.approx(2.0)
        assert result["per_sample"][0]["y_other"] == pytest.approx(2.5)

    def test_judge_override(self):
        """An explicit judge name overrides the recorded one."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)), judge_model="qwen")

        assert result["judge_model"] == "qwen"
        # y_self = 3.0, y_other = (3.6 + 3.0) / 2 = 3.3
        assert result["per_sample"][0]["error_rate_se"] == pytest.approx(3.0 / 3.3 - 1)

    def test_missing_judge_model(self):
        """Without judge_model metadata the judge cannot be determined."""
        records = [{"id": 1, "output_model": "gpt-4o", "score": 3.0}]

        with pytest.raises(ValueError, match="judge_model"):
            evaluate_judge_bias(records)

    def test_multiple_judge_models(self):
        """A file with two judges needs an explicit judge name."""
        records = grid(symmetric_grid()) + grid(symmetric_grid(), judge_model="qwen")

        with pytest.raises(ValueError, match="several judge_model values"):
            evaluate_judge_bias(records)

    def test_judge_not_among_output_models(self):
        """A judge outside the scored models cannot be evaluated."""
        records = grid({1: {"qwen": 3.0, "gemini": 2.0}}, judge_model="gpt-4o")

        result = evaluate_judge_bias(records)

        assert result["num_defined"] == 0
        assert result["undefined_ids"] == [1]

    def test_empty_records(self):
        """An empty record list yields no samples and no bias."""
        result = evaluate_judge_bias([], judge_model="gpt-4o")

        assert result["num_samples"] == 0
        assert result["defined"] is False
        assert result["mean_error_rate_se"] == 0.0


class TestFormatTable:
    """Tests for the per-sample table rendering."""

    def test_line_count(self):
        """One header row, one separator row and one row per valid sample."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))

        assert len(_format_table(result)) == 4

    def test_columns_align(self):
        """Every row has the same width and pipes in the same columns."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))

        lines = _format_table(result)

        assert len({len(line) for line in lines}) == 1
        assert len({tuple(i for i, char in enumerate(line) if char == "|") for line in lines}) == 1

    def test_reported_per_line(self):
        """The table is logged one row per record, so loguru prefixes stay aligned."""
        result = evaluate_judge_bias(grid(symmetric_grid(offset=0.6)))
        records: list[str] = []
        handler_id = loguru_logger.add(
            lambda message: records.append(message.record["message"]),
            format="{message}",
        )
        try:
            _log_report(result, "judge.jsonl")
        finally:
            loguru_logger.remove(handler_id)

        assert not any("\n" in record for record in records)
        assert _format_table(result) == [record for record in records if record.startswith("|")]


class TestEvalBiasCli:
    """Tests for the eval_bias command-line entry point."""

    def test_end_to_end(self, tmp_path, monkeypatch):
        """A grid file runs end to end and the report holds the per-sample bias."""
        path = write_records(tmp_path / "judge.jsonl", grid(symmetric_grid(offset=0.6)))
        output = tmp_path / "bias.json"
        monkeypatch.setattr("sys.argv", ["eval_bias.py", "--scores", path, "-o", str(output)])

        assert main() == 0

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["scores_file"] == path
        assert report["judge_model"] == "gpt-4o"
        assert report["num_samples"] == 2
        assert report["num_defined"] == 2
        assert len(report["per_sample"]) == 2
        assert report["mean_error_rate_se"] == pytest.approx(0.175)

    def test_default_file(self, tmp_path, monkeypatch):
        """The bundled simulation data is the default input."""
        monkeypatch.chdir(tmp_path)
        dataset = tmp_path / "dataset" / "simulat_data"
        dataset.mkdir(parents=True)
        write_records(dataset / "bias_judge_gpt-4o.jsonl", grid(symmetric_grid(offset=0.6)))
        monkeypatch.setattr("sys.argv", ["eval_bias.py"])

        assert main() == 0

    def test_judge_model_option(self, tmp_path, monkeypatch):
        """--judge-model selects the judge explicitly."""
        path = write_records(tmp_path / "judge.jsonl", grid(symmetric_grid(offset=0.6)))
        monkeypatch.setattr("sys.argv", ["eval_bias.py", "--scores", path, "--judge-model", "qwen"])

        assert main() == 0

    def test_missing_file(self, tmp_path, monkeypatch):
        """A missing input file fails with exit code 1."""
        monkeypatch.setattr(
            "sys.argv", ["eval_bias.py", "--scores", str(tmp_path / "absent.jsonl")]
        )

        assert main() == 1

    def test_ambiguous_judge(self, tmp_path, monkeypatch):
        """A file with two judges fails with exit code 1."""
        path = write_records(
            tmp_path / "judge.jsonl",
            grid(symmetric_grid()) + grid(symmetric_grid(), judge_model="qwen"),
        )
        monkeypatch.setattr("sys.argv", ["eval_bias.py", "--scores", path])

        assert main() == 1

    def test_undefined_bias_exits_nonzero(self, tmp_path, monkeypatch):
        """A grid without the judge's own output cannot produce a bias."""
        path = write_records(tmp_path / "judge.jsonl", grid({1: {"qwen": 2.0, "gemini": 2.0}}))
        monkeypatch.setattr("sys.argv", ["eval_bias.py", "--scores", path])

        assert main() == 1

    def test_empty_file(self, tmp_path, monkeypatch):
        """An empty input file fails with exit code 1."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["eval_bias.py", "--scores", str(path)])

        assert main() == 1
