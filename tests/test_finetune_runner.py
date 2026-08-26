"""
Optis Benchmark - Fine-tune Runner Tests

测试覆盖：
    - YAML 配置加载与环境变量展开
    - 配置校验（config.validate）
    - JSONL 文件校验（validate_jsonl）
    - 状态对象转换（from_dict_raw / to_dict / from_dict）
    - create_job 全流程（含失败回收）
    - wait_for_completion 状态机（成功 / 超时）
    - cancel_job / list_jobs / get_events
    - status 输出文件 读写往返
    - Provider 工厂函数
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.core.finetune_runner import (
    FineTuneJobStatus,
    FineTuneRunner,
    FineTuneRunnerConfig,
)
from src.llm import create_provider

# =============================================================================
# Stubs
# =============================================================================


class _StubProvider:
    """模拟 Provider，支持按序列返回状态。"""

    def __init__(self, status_sequence: list[str] | None = None) -> None:
        self._sequence = status_sequence or ["queued", "running", "succeeded"]
        self._call_count = 0
        self.uploaded_files: list[str] = []
        self.deleted_files: list[str] = []

    @property
    def client(self):
        return self

    async def close(self) -> None:
        pass


class _StubFineTuneRunner(FineTuneRunner):
    """注入 stub provider 的 FineTuneRunner。"""

    def __init__(self, config: FineTuneRunnerConfig, status_sequence: list[str] | None = None):
        super().__init__(config)
        self.provider_type = config.provider_config.get("type", "openai")
        self._stub = _StubProvider(status_sequence)
        self.provider = self._stub

    async def _upload_file(self, file_path: str) -> str:
        file_id = f"file-{len(self._stub.uploaded_files)}"
        self._stub.uploaded_files.append(file_id)
        return file_id

    async def _delete_file(self, file_id: str) -> bool:
        self._stub.deleted_files.append(file_id)
        return True

    async def _create_job(
        self, training_file_id, model, validation_file_id=None,
        method="supervised", suffix=None, seed=None, hyperparameters=None,
    ) -> dict[str, Any]:
        return {
            "job_id": "ftjob-test123",
            "status": "queued",
            "base_model": model,
            "fine_tuned_model": None,
            "trained_tokens": 0,
            "error": None,
        }

    async def _retrieve_job(self, job_id: str) -> dict[str, Any]:
        idx = min(self._stub._call_count, len(self._stub._sequence) - 1)
        status = self._stub._sequence[idx]
        self._stub._call_count += 1
        return {
            "job_id": job_id,
            "status": status,
            "base_model": "gpt-4o-mini",
            "fine_tuned_model": (
                "ft:gpt-4o-mini:org:suffix:ftjob123" if status == "succeeded" else None
            ),
            "trained_tokens": 12345 if status == "succeeded" else 0,
            "error": None,
        }

    async def _cancel_job(self, job_id: str) -> dict[str, Any]:
        self._stub._call_count += 1
        return {
            "job_id": job_id,
            "status": "cancelled",
            "base_model": "gpt-4o-mini",
            "fine_tuned_model": None,
            "trained_tokens": 0,
            "error": None,
        }

    async def _list_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "job_id": "ftjob-1",
                "status": "succeeded",
                "base_model": "gpt-4o-mini",
                "fine_tuned_model": "ft:gpt-4o-mini:org:optis:v1",
                "trained_tokens": 100,
                "error": None,
            },
            {
                "job_id": "ftjob-2",
                "status": "running",
                "base_model": "gpt-4o-mini",
                "fine_tuned_model": None,
                "trained_tokens": 50,
                "error": None,
            },
        ]

    async def _list_events(self, job_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "created_at": i * 100,
                "level": "info",
                "message": f"event {i}",
                "data": {"loss": 0.5},
            }
            for i in range(min(limit, 2))
        ]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_train_jsonl(tmp_path: Path) -> Path:
    """有效的 train.jsonl 文件。"""
    path = tmp_path / "train.jsonl"
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are an optics assistant."},
                {"role": "user", "content": "Paper title: Fiber Optics"},
                {
                    "role": "assistant",
                    "content": '{"title": "Fiber Optics", "DOI": "10.1234/fiber"}',
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an optics assistant."},
                {"role": "user", "content": "Paper title: Laser Systems"},
                {
                    "role": "assistant",
                    "content": '{"title": "Laser Systems", "DOI": "10.1234/laser"}',
                },
            ]
        },
    ]
    path.write_text(
        "\n".join(json.dumps(s) for s in samples) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_val_jsonl(tmp_path: Path) -> Path:
    """有效的 val.jsonl 文件。"""
    path = tmp_path / "val.jsonl"
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are an optics assistant."},
                {"role": "user", "content": "Paper title: Photonics"},
                {"role": "assistant", "content": '{"title": "Photonics"}'},
            ]
        },
    ]
    path.write_text(
        "\n".join(json.dumps(s) for s in samples) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def valid_config(tmp_path, sample_train_jsonl) -> FineTuneRunnerConfig:
    """合法配置。"""
    return FineTuneRunnerConfig(
        provider_config={
            "type": "openai",
            "api_key": "test-key-123",
            "base_url": "https://api.example.com/v1",
        },
        job_config={
            "training_file": str(sample_train_jsonl),
            "validation_file": None,
            "base_model": "gpt-4o-mini-2024-07-18",
            "method": "supervised",
            "suffix": "optis-bench",
            "seed": 42,
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": "auto",
                "learning_rate_multiplier": "auto",
            },
        },
        execution_config={
            "poll_interval": 1,
            "poll_timeout": 10,
            "status_output_path": str(tmp_path / "status.json"),
        },
    )


def _make_runner(
    config: FineTuneRunnerConfig, status_sequence: list[str] | None = None
) -> _StubFineTuneRunner:
    """构造注入 stub provider 的 FineTuneRunner。"""
    return _StubFineTuneRunner(config, status_sequence)


# =============================================================================
# Tests
# =============================================================================


class TestFineTuneJobStatus:
    """FineTuneJobStatus 数据类测试。"""

    def test_to_dict(self):
        status = FineTuneJobStatus(job_id="abc", status="succeeded", trained_tokens=100)
        d = status.to_dict()
        assert d["job_id"] == "abc"
        assert d["status"] == "succeeded"
        assert d["trained_tokens"] == 100

    def test_from_dict(self):
        d = {"job_id": "xyz", "status": "running", "error": None}
        status = FineTuneJobStatus.from_dict(d)
        assert status.job_id == "xyz"
        assert status.status == "running"

    def test_from_dict_raw(self):
        d = {
            "job_id": "ftjob-raw",
            "status": "succeeded",
            "base_model": "gpt-4o-mini",
            "fine_tuned_model": "ft:gpt-4o-mini:org:optis:123",
            "trained_tokens": 500,
            "error": None,
        }
        status = FineTuneJobStatus.from_dict_raw(d)
        assert status.job_id == "ftjob-raw"
        assert status.fine_tuned_model == "ft:gpt-4o-mini:org:optis:123"
        assert status.trained_tokens == 500
        assert status.base_model == "gpt-4o-mini"


class TestFineTuneRunnerConfig:
    """配置校验测试。"""

    def test_from_yaml_roundtrip(self, tmp_path, sample_train_jsonl):
        yaml_path = tmp_path / "ft.yaml"
        yaml_path.write_text(
            f"""
llm:
  provider:
    type: openai
    api_key: "${{TEST_FT_KEY}}"
    base_url: https://api.openai.com/v1
fine_tuning:
  training_file: "{sample_train_jsonl}"
  base_model: "gpt-4o-mini-2024-07-18"
  method: supervised
  suffix: "test-v1"
execution:
  poll_interval: 10
  poll_timeout: 3600
  status_output_path: "{tmp_path}/status.json"
""",
            encoding="utf-8",
        )
        os.environ["TEST_FT_KEY"] = "env-expanded-key"
        config = FineTuneRunnerConfig.from_yaml(str(yaml_path))
        assert config.provider_config["api_key"] == "env-expanded-key"
        assert config.job_config["base_model"] == "gpt-4o-mini-2024-07-18"
        assert config.poll_interval == 10

    def test_validate_passes(self, valid_config):
        errors = valid_config.validate()
        assert errors == []

    def test_validate_missing_api_key(self, valid_config):
        valid_config.provider_config["api_key"] = ""
        errors = valid_config.validate()
        assert any("api_key" in e for e in errors)

    def test_validate_missing_training_file(self, valid_config):
        valid_config.job_config["training_file"] = "/no/such/file.jsonl"
        errors = valid_config.validate()
        assert any("training_file" in e for e in errors)

    def test_validate_invalid_suffix(self, valid_config):
        valid_config.job_config["suffix"] = "x" * 20
        errors = valid_config.validate()
        assert any("suffix" in e for e in errors)

    def test_validate_invalid_method(self, valid_config):
        valid_config.job_config["method"] = "invalid_method"
        errors = valid_config.validate()
        assert any("method" in e for e in errors)


class TestValidateJsonl:
    """JSONL 文件校验测试。"""

    def test_valid_file(self, sample_train_jsonl):
        errors = FineTuneRunner.validate_jsonl(str(sample_train_jsonl))
        assert errors == []

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        errors = FineTuneRunner.validate_jsonl(str(path))
        assert any("no samples" in e for e in errors)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        errors = FineTuneRunner.validate_jsonl(str(path))
        assert any("invalid JSON" in e for e in errors)

    def test_missing_messages(self, tmp_path):
        path = tmp_path / "no_msgs.jsonl"
        path.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
        errors = FineTuneRunner.validate_jsonl(str(path))
        assert any("messages" in e for e in errors)

    def test_wrong_role_sequence(self, tmp_path):
        path = tmp_path / "bad_roles.jsonl"
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "oops"},
            ]
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        errors = FineTuneRunner.validate_jsonl(str(path))
        assert any("role sequence" in e for e in errors)

    def test_empty_content(self, tmp_path):
        path = tmp_path / "empty_content.jsonl"
        record = {
            "messages": [
                {"role": "system", "content": "ok"},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "response"},
            ]
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        errors = FineTuneRunner.validate_jsonl(str(path))
        assert any("empty content" in e for e in errors)


class TestCreateJob:
    """create_job 全流程测试。"""

    async def test_create_success(self, valid_config, tmp_path):
        runner = _make_runner(valid_config)
        status = await runner.create_job()
        assert status.job_id == "ftjob-test123"
        assert status.status == "queued"
        assert valid_config.job_config["base_model"] == "gpt-4o-mini-2024-07-18"
        assert Path(valid_config.status_output_path).exists()

    async def test_create_validates_before_upload(self):
        config = FineTuneRunnerConfig(
            provider_config={"type": "openai", "api_key": "k"},
            job_config={"base_model": "gpt-4o-mini", "method": "bad"},
            execution_config={},
        )
        runner = _make_runner(config)
        with pytest.raises(ValueError, match="invalid method"):
            await runner.create_job()

    async def test_create_rejects_invalid_jsonl(self, valid_config):
        invalid = valid_config.job_config.copy()
        invalid["training_file"] = "/no/file.jsonl"
        config = FineTuneRunnerConfig(
            provider_config=valid_config.provider_config,
            job_config=invalid,
            execution_config=valid_config.execution_config,
        )
        runner = _make_runner(config)
        with pytest.raises(ValueError, match="not found"):
            await runner.create_job()


class TestWaitForCompletion:
    """wait_for_completion 状态机测试。"""

    async def test_wait_succeeds(self, valid_config):
        runner = _make_runner(valid_config, ["queued", "running", "succeeded"])
        status = await runner.wait_for_completion(job_id="ftjob-test", interval=0.01, timeout=5)
        assert status.status == "succeeded"
        assert status.fine_tuned_model == "ft:gpt-4o-mini:org:suffix:ftjob123"

    async def test_wait_timeout(self, valid_config):
        runner = _make_runner(valid_config, ["running", "running"])
        valid_config.execution_config["poll_interval"] = 0.01
        status = await runner.wait_for_completion(job_id="ftjob-test", interval=0.01, timeout=0.05)
        assert status.error is not None
        assert "timed out" in status.error

    async def test_wait_failed_status(self, valid_config):
        runner = _make_runner(valid_config, ["running", "failed"])
        status = await runner.wait_for_completion(job_id="ftjob-test", interval=0.01, timeout=5)
        assert status.status == "failed"

    async def test_wait_no_job_id_raises(self, valid_config):
        valid_config.execution_config["status_output_path"] = "/no/such/status.json"
        runner = _make_runner(valid_config)
        with pytest.raises(ValueError, match="No job_id"):
            await runner.wait_for_completion()


class TestJobOperations:
    """cancel / list / events 操作测试。"""

    async def test_cancel(self, valid_config):
        runner = _make_runner(valid_config)
        status = await runner.cancel_job("ftjob-test123")
        assert status.status == "cancelled"

    async def test_list_jobs(self, valid_config):
        runner = _make_runner(valid_config)
        jobs = await runner.list_jobs(limit=5)
        assert len(jobs) == 2
        assert jobs[0].job_id == "ftjob-1"
        assert jobs[0].fine_tuned_model == "ft:gpt-4o-mini:org:optis:v1"

    async def test_get_events(self, valid_config):
        runner = _make_runner(valid_config)
        events = await runner.get_events("ftjob-test123", limit=5)
        assert len(events) == 2
        assert events[0]["level"] == "info"


class TestStatusPersistence:
    """status 文件写入 / 读取往返测试。"""

    def test_save_and_load(self, valid_config):
        runner = _make_runner(valid_config)
        status = FineTuneJobStatus(
            job_id="ftjob-persist",
            status="succeeded",
            base_model="gpt-4o-mini",
            fine_tuned_model="ft:gpt-4o-mini:org:v1:abc",
            trained_tokens=1000,
        )
        runner.save_status(status)
        loaded_id = runner._load_saved_job_id()
        assert loaded_id == "ftjob-persist"

    def test_load_missing_file(self, valid_config):
        valid_config.execution_config["status_output_path"] = "/tmp/nonexistent.json"
        runner = _make_runner(valid_config)
        assert runner._load_saved_job_id() == ""


class TestProviderFactory:
    """Provider 工厂函数测试。"""

    def test_create_openai_provider(self):
        provider = create_provider({
            "type": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        })
        assert type(provider).__name__ == "OpenAIProvider"

    def test_create_mistral_provider(self):
        provider = create_provider({
            "type": "mistral",
            "api_key": "test-key",
        })
        assert type(provider).__name__ == "MistralProvider"

    def test_create_together_provider(self):
        provider = create_provider({
            "type": "together",
            "api_key": "test-key",
            "base_url": "https://api.together.xyz",
        })
        assert type(provider).__name__ == "TogetherAIProvider"

    def test_create_dashscope_provider(self):
        provider = create_provider({
            "type": "dashscope",
            "api_key": "test-key",
            "base_url": "https://test.aliyuncs.com/compatible-mode/v1",
        })
        assert type(provider).__name__ == "DashScopeProvider"

    def test_create_bedrock_provider(self):
        try:
            import boto3  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")
        provider = create_provider({
            "type": "bedrock",
            "region": "us-east-1",
        })
        assert type(provider).__name__ == "BedrockProvider"

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="不支持的 Provider 类型"):
            create_provider({"type": "unsupported"})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="必须包含 'type' 字段"):
            create_provider({})
