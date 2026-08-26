"""
Optis Benchmark - Fine-tune Runner Module

多提供商微调任务管理器，对齐 LLMPredRunner 的架构风格。
通过 Provider 薄客户端包装 + 内联业务逻辑实现微调。

执行流程：
    1. validate()      - 校验配置与训练文件
    2. create_job()    - 上传 JSONL → 创建微调任务
    3. wait_for_completion() - 轮询状态直至终态（可选）
    4. save_status()   - 状态落盘（job id / ft 模型名）

微调完成后，将 status_output_path 中记录的 fine_tuned_model 名填入
configs/llm/GPT_OpenAI.yaml 的 model.name 即可复用现有推理/评测管线。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.llm import create_provider
from src.utils import logger

# =============================================================================
# Constants
# =============================================================================

# 终态状态集合（达到后停止轮询）
TERMINAL_STATUSES: set[str] = {"succeeded", "failed", "cancelled"}

# 支持的微调方法
VALID_METHODS: set[str] = {"supervised", "dpo"}

# suffix 最大长度（OpenAI 官方限制）
MAX_SUFFIX_LENGTH = 18

# 轮询默认值
DEFAULT_POLL_INTERVAL = 30
DEFAULT_POLL_TIMEOUT = 86400

# Bedrock 微调状态映射
BEDROCK_STATUS_MAP: dict[str, str] = {
    "InProgress": "running",
    "Completed": "succeeded",
    "Failed": "failed",
    "Stopping": "cancelling",
    "Stopped": "cancelled",
}

# DashScope 状态映射
DASHSCOPE_STATUS_MAP: dict[str, str] = {
    "PENDING": "queued",
    "QUEUING": "queued",
    "RUNNING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "CANCELED": "cancelled",
    "CANCELING": "cancelled",
}

# Together AI 端点
TOGETHER_FINETUNE_ENDPOINT = "/v1/fine-tunes"
TOGETHER_FILES_ENDPOINT = "/v1/files"


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables ${VAR_NAME}."""
    if isinstance(data, str):
        if data.startswith("${") and data.endswith("}"):
            return os.environ.get(data[2:-1], "")
        return data
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FineTuneJobStatus:
    """微调任务状态快照。"""

    job_id: str = ""
    status: str = ""
    base_model: str = ""
    fine_tuned_model: str | None = None
    trained_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTuneJobStatus:
        """Create from dictionary."""
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            base_model=data.get("base_model", ""),
            fine_tuned_model=data.get("fine_tuned_model"),
            trained_tokens=data.get("trained_tokens", 0),
            error=data.get("error"),
        )

    @classmethod
    def from_dict_raw(cls, data: dict[str, Any]) -> FineTuneJobStatus:
        """从适配器返回的字典创建状态快照。"""
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            base_model=data.get("base_model", ""),
            fine_tuned_model=data.get("fine_tuned_model"),
            trained_tokens=data.get("trained_tokens", 0),
            error=data.get("error"),
        )


@dataclass
class FineTuneRunnerConfig:
    """Fine-tune runner configuration."""

    provider_config: dict[str, Any]
    job_config: dict[str, Any]
    execution_config: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> FineTuneRunnerConfig:
        """Load configuration from YAML file.

        Args:
            path: YAML 配置文件路径

        Returns:
            FineTuneRunnerConfig 实例

        Raises:
            FileNotFoundError: 配置文件不存在时
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path_obj, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data = _expand_env_vars(data)

        llm_config = data.get("llm", {})
        return cls(
            provider_config=llm_config.get("provider", {}),
            job_config=data.get("fine_tuning", {}),
            execution_config=data.get("execution", {}),
        )

    def validate(self) -> list[str]:
        """校验配置合法性。

        Returns:
            错误列表；空列表表示配置合法
        """
        errors: list[str] = []

        if not self.provider_config.get("api_key"):
            provider_type = self.provider_config.get("type", "")
            if provider_type not in ("bedrock",):
                errors.append("llm.provider.api_key is empty (check env var expansion)")

        job = self.job_config
        training_file = job.get("training_file", "")
        if not training_file:
            errors.append("fine_tuning.training_file is required")
        elif not Path(training_file).exists():
            errors.append(f"training_file not found: {training_file}")
        if validation_file := job.get("validation_file"):
            if not Path(validation_file).exists():
                errors.append(f"validation_file not found: {validation_file}")

        if not job.get("base_model"):
            errors.append("fine_tuning.base_model is required")

        method = job.get("method", "supervised")
        if method not in VALID_METHODS:
            errors.append(f"invalid method '{method}', expected one of {sorted(VALID_METHODS)}")

        suffix = str(job.get("suffix") or "")
        if len(suffix) > MAX_SUFFIX_LENGTH:
            errors.append(f"suffix exceeds {MAX_SUFFIX_LENGTH} chars: '{suffix}'")

        return errors

    @property
    def poll_interval(self) -> int:
        """轮询间隔（秒）。"""
        return int(self.execution_config.get("poll_interval", DEFAULT_POLL_INTERVAL))

    @property
    def poll_timeout(self) -> int:
        """轮询超时（秒）。"""
        return int(self.execution_config.get("poll_timeout", DEFAULT_POLL_TIMEOUT))

    @property
    def status_output_path(self) -> str:
        """状态落盘路径。"""
        return self.execution_config.get("status_output_path", "results/finetune/job_status.json")


# =============================================================================
# Classes
# =============================================================================


class FineTuneRunner:
    """多提供商微调任务管理器。

    通过 Provider 薄客户端包装 + 内联业务逻辑实现微调。

    执行流程：
        1. setup() - 创建 Provider 实例
        2. create_job() / wait_for_completion() / cancel_job() 等任务操作
        3. teardown() - 清理资源
    """

    def __init__(self, config: FineTuneRunnerConfig):
        """初始化 Runner。

        Args:
            config: 微调运行配置
        """
        self.config = config
        self.provider: Any = None
        self.provider_type: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """创建 Provider 实例。"""
        self.provider_type = self.config.provider_config.get("type", "")
        self.provider = create_provider(self.config.provider_config)
        logger.info(f"Provider: {type(self.provider).__name__}")

    async def teardown(self) -> None:
        """清理资源。"""
        if self.provider:
            try:
                await self.provider.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Local Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_jsonl(path: str | Path) -> list[str]:
        """校验微调 JSONL 文件的格式与消息结构。

        Args:
            path: JSONL 文件路径

        Returns:
            错误列表；空列表表示合法
        """
        errors: list[str] = []
        path_obj = Path(path)

        line_count = 0
        with open(path_obj, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                line_count += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"line {line_no}: invalid JSON ({e})")
                    continue

                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    errors.append(f"line {line_no}: 'messages' must be a list with >= 2 items")
                    continue

                roles = [m.get("role", "") for m in messages]
                core_roles = roles[1:] if roles and roles[0] == "system" else roles
                if core_roles != ["user", "assistant"]:
                    errors.append(
                        f"line {line_no}: invalid role sequence {roles}, "
                        f"expected [system?] + user + assistant"
                    )
                if any(not m.get("content") for m in messages):
                    errors.append(f"line {line_no}: message with empty content")

        if line_count == 0:
            errors.append("file contains no samples")

        return errors

    # ------------------------------------------------------------------
    # Provider-specific: Upload File
    # ------------------------------------------------------------------

    async def _upload_file(self, file_path: str) -> str:
        """上传文件到微调提供商。

        Args:
            file_path: 本地文件路径

        Returns:
            文件 ID 或路径
        """
        if self.provider_type == "openai":
            return await self._upload_file_openai(file_path)
        elif self.provider_type == "mistral":
            return await self._upload_file_mistral(file_path)
        elif self.provider_type == "together":
            return await self._upload_file_together(file_path)
        elif self.provider_type == "bedrock":
            return await self._upload_file_bedrock(file_path)
        elif self.provider_type == "dashscope":
            return await self._upload_file_dashscope(file_path)
        else:
            raise ValueError(f"不支持的 Provider 类型: {self.provider_type}")

    async def _upload_file_openai(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_obj = await self.provider.client.files.create(file=f, purpose="fine-tune")
        return file_obj.id

    async def _upload_file_mistral(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_obj = await self.provider.client.files.upload(file=f, purpose="fine-tune")
        return file_obj.id

    async def _upload_file_together(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/jsonl")}
            data = {"purpose": "fine-tune"}
            response = await self.provider.client.post(
                TOGETHER_FILES_ENDPOINT, files=files, data=data,
            )
            response.raise_for_status()
            return response.json().get("id", "")

    async def _upload_file_bedrock(self, file_path: str) -> str:
        if file_path.startswith("s3://"):
            return file_path
        raise ValueError(
            "Bedrock 微调需要将训练数据上传到 S3。"
            "请使用 AWS CLI: aws s3 cp <local_path> s3://<bucket>/<key>"
        )

    async def _upload_file_dashscope(self, file_path: str) -> str:
        from dashscope import Files

        response = await asyncio.to_thread(Files.upload, file_path=file_path, purpose="fine-tune")
        if hasattr(response, "output") and response.output:
            output = response.output
            if isinstance(output, dict):
                uploaded_files = output.get("uploaded_files", [])
                return uploaded_files[0].get("file_id", "")
            return getattr(output, "file_id", "")
        raise RuntimeError(f"Failed to upload file: {response}")

    # ------------------------------------------------------------------
    # Provider-specific: Create Job
    # ------------------------------------------------------------------

    async def _create_job(
        self,
        training_file_id: str,
        model: str,
        validation_file_id: str | None = None,
        method: str = "supervised",
        suffix: str | None = None,
        seed: int | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """创建微调任务。

        Returns:
            标准任务状态字典
        """
        if self.provider_type == "openai":
            return await self._create_job_openai(
                training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
            )
        elif self.provider_type == "mistral":
            return await self._create_job_mistral(
                training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
            )
        elif self.provider_type == "together":
            return await self._create_job_together(
                training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
            )
        elif self.provider_type == "bedrock":
            return await self._create_job_bedrock(
                training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
            )
        elif self.provider_type == "dashscope":
            return await self._create_job_dashscope(
                training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
            )
        else:
            raise ValueError(f"不支持的 Provider 类型: {self.provider_type}")

    async def _create_job_openai(
        self, training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "training_file": training_file_id,
            "model": model,
            "suffix": suffix,
            "seed": seed,
        }
        if validation_file_id is not None:
            request["validation_file"] = validation_file_id
        method_config: dict[str, Any] = {"type": method}
        if hyperparameters:
            method_config[method] = {"hyperparameters": hyperparameters}
        request["method"] = method_config

        job = await self.provider.client.fine_tuning.jobs.create(**request)

        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", None)

        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": model,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _create_job_mistral(
        self, training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "training_files": [training_file_id],
            "model": model,
            "suffix": suffix,
        }
        if validation_file_id is not None:
            request["validation_files"] = [validation_file_id]
        if hyperparameters:
            request["hyperparameters"] = hyperparameters
        if seed is not None:
            request["seed"] = seed

        job = await self.provider.client.fine_tuning_jobs.create(**request)

        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", str(job.error))

        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": model,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _create_job_together(
        self, training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "training_file": training_file_id,
            "model": model,
        }
        if validation_file_id is not None:
            request["validation_file"] = validation_file_id
        if method != "supervised":
            request["method"] = method
        if suffix is not None:
            request["suffix"] = suffix
        if seed is not None:
            request["seed"] = seed
        if hyperparameters:
            request["hyperparameters"] = hyperparameters

        response = await self.provider.client.post(TOGETHER_FINETUNE_ENDPOINT, json=request)
        response.raise_for_status()
        result = response.json()

        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", ""),
            "base_model": model,
            "fine_tuned_model": result.get("fine_tuned_model"),
            "trained_tokens": result.get("trained_tokens", 0) or 0,
            "error": result.get("error"),
        }

    async def _create_job_bedrock(
        self, training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        job_name = suffix or f"finetune-{model.replace('.', '-')}"

        request_params: dict[str, Any] = {
            "jobName": job_name,
            "customModelName": f"{model}-custom",
            "baseModelIdentifier": model,
            "trainingDataConfig": {"s3Uri": training_file_id},
            "outputDataConfig": {},
        }
        if validation_file_id:
            request_params["validationDataConfig"] = {"s3Uri": validation_file_id}
        if hyperparameters:
            request_params["hyperParameters"] = hyperparameters

        response = await loop.run_in_executor(
            None, lambda: self.provider.client.create_model_customization_job(**request_params),
        )
        job_arn = response.get("jobArn", "")
        job_id = job_arn.split("/")[-1] if "/" in job_arn else job_arn

        return {
            "job_id": job_id,
            "status": "running",
            "base_model": model,
            "fine_tuned_model": None,
            "trained_tokens": 0,
            "error": None,
        }

    async def _create_job_dashscope(
        self, training_file_id, model, validation_file_id, method, suffix, seed, hyperparameters,
    ) -> dict[str, Any]:
        from dashscope import FineTunes

        training_type = self.config.job_config.get("training_type", "sft")
        params: dict[str, Any] = {
            "model": model,
            "training_file_ids": [training_file_id],
            "training_type": training_type,
        }
        if validation_file_id:
            params["validation_file_ids"] = [validation_file_id]
        if suffix:
            params["finetuned_output_suffix"] = suffix
        if hyperparameters:
            params["hyper_parameters"] = hyperparameters
        job_name = self.config.job_config.get("job_name", None)
        if job_name:
            params["job_name"] = job_name
        model_name = self.config.job_config.get("model_name", None)
        if model_name:
            params["model_name"] = model_name
        if seed is not None:
            if "hyper_parameters" not in params:
                params["hyper_parameters"] = {}
            params["hyper_parameters"]["seed"] = seed

        response = await asyncio.to_thread(FineTunes.call, **params)

        if hasattr(response, "status_code") and response.status_code != 200:
            error_msg = getattr(response, "message", str(response))
            raise RuntimeError(f"Failed to create fine-tune job: {error_msg}")

        return self._parse_dashscope_response(response)

    # ------------------------------------------------------------------
    # Provider-specific: Retrieve Job
    # ------------------------------------------------------------------

    async def _retrieve_job(self, job_id: str) -> dict[str, Any]:
        if self.provider_type == "openai":
            return await self._retrieve_job_openai(job_id)
        elif self.provider_type == "mistral":
            return await self._retrieve_job_mistral(job_id)
        elif self.provider_type == "together":
            return await self._retrieve_job_together(job_id)
        elif self.provider_type == "bedrock":
            return await self._retrieve_job_bedrock(job_id)
        elif self.provider_type == "dashscope":
            return await self._retrieve_job_dashscope(job_id)
        else:
            raise ValueError(f"不支持的 Provider 类型: {self.provider_type}")

    async def _retrieve_job_openai(self, job_id: str) -> dict[str, Any]:
        job = await self.provider.client.fine_tuning.jobs.retrieve(job_id)
        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", None)
        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": getattr(job, "model", ""),
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _retrieve_job_mistral(self, job_id: str) -> dict[str, Any]:
        job = await self.provider.client.fine_tuning_jobs.get(job_id)
        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", str(job.error))
        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": getattr(job, "model", ""),
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _retrieve_job_together(self, job_id: str) -> dict[str, Any]:
        response = await self.provider.client.get(f"{TOGETHER_FINETUNE_ENDPOINT}/{job_id}")
        response.raise_for_status()
        result = response.json()
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", ""),
            "base_model": result.get("model", ""),
            "fine_tuned_model": result.get("fine_tuned_model"),
            "trained_tokens": result.get("trained_tokens", 0) or 0,
            "error": result.get("error"),
        }

    async def _retrieve_job_bedrock(self, job_id: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.provider.client.get_model_customization_job(jobIdentifier=job_id),
        )
        status_raw = response.get("status", "")
        status = BEDROCK_STATUS_MAP.get(status_raw, status_raw.lower())
        model_id = response.get("baseModelIdentifier", "")
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        return {
            "job_id": response.get("jobArn", job_id),
            "status": status,
            "base_model": model_name,
            "fine_tuned_model": response.get("outputModelArn"),
            "trained_tokens": 0,
            "error": response.get("failureMessage"),
        }

    async def _retrieve_job_dashscope(self, job_id: str) -> dict[str, Any]:
        from dashscope import FineTunes

        response = await asyncio.to_thread(FineTunes.get, job_id)
        if hasattr(response, "status_code") and response.status_code != 200:
            error_msg = getattr(response, "message", str(response))
            raise RuntimeError(f"Failed to retrieve job: {error_msg}")
        return self._parse_dashscope_response(response)

    # ------------------------------------------------------------------
    # Provider-specific: Cancel Job
    # ------------------------------------------------------------------

    async def _cancel_job(self, job_id: str) -> dict[str, Any]:
        if self.provider_type == "openai":
            return await self._cancel_job_openai(job_id)
        elif self.provider_type == "mistral":
            return await self._cancel_job_mistral(job_id)
        elif self.provider_type == "together":
            return await self._cancel_job_together(job_id)
        elif self.provider_type == "bedrock":
            return await self._cancel_job_bedrock(job_id)
        elif self.provider_type == "dashscope":
            return await self._cancel_job_dashscope(job_id)
        else:
            raise ValueError(f"不支持的 Provider 类型: {self.provider_type}")

    async def _cancel_job_openai(self, job_id: str) -> dict[str, Any]:
        job = await self.provider.client.fine_tuning.jobs.cancel(job_id)
        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", None)
        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": getattr(job, "model", ""),
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _cancel_job_mistral(self, job_id: str) -> dict[str, Any]:
        job = await self.provider.client.fine_tuning_jobs.cancel(job_id)
        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", str(job.error))
        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": getattr(job, "model", ""),
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _cancel_job_together(self, job_id: str) -> dict[str, Any]:
        response = await self.provider.client.post(f"{TOGETHER_FINETUNE_ENDPOINT}/{job_id}/cancel")
        response.raise_for_status()
        result = response.json()
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", ""),
            "base_model": result.get("model", ""),
            "fine_tuned_model": result.get("fine_tuned_model"),
            "trained_tokens": result.get("trained_tokens", 0) or 0,
            "error": result.get("error"),
        }

    async def _cancel_job_bedrock(self, job_id: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: self.provider.client.stop_model_customization_job(jobIdentifier=job_id),
        )
        return await self._retrieve_job_bedrock(job_id)

    async def _cancel_job_dashscope(self, job_id: str) -> dict[str, Any]:
        from dashscope import FineTunes

        response = await asyncio.to_thread(FineTunes.cancel, job_id)
        if hasattr(response, "status_code") and response.status_code != 200:
            error_msg = getattr(response, "message", str(response))
            raise RuntimeError(f"Failed to cancel job: {error_msg}")
        return await self._retrieve_job_dashscope(job_id)

    # ------------------------------------------------------------------
    # Provider-specific: List Jobs
    # ------------------------------------------------------------------

    async def _list_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        if self.provider_type == "openai":
            return await self._list_jobs_openai(limit)
        elif self.provider_type == "mistral":
            return await self._list_jobs_mistral(limit)
        elif self.provider_type == "together":
            return await self._list_jobs_together(limit)
        elif self.provider_type == "bedrock":
            return await self._list_jobs_bedrock(limit)
        elif self.provider_type == "dashscope":
            return await self._list_jobs_dashscope(limit)
        else:
            return []

    async def _list_jobs_openai(self, limit: int) -> list[dict[str, Any]]:
        response = await self.provider.client.fine_tuning.jobs.list(limit=limit)
        statuses = []
        for job in response.data:
            error = None
            if hasattr(job, "error") and job.error:
                error = getattr(job.error, "message", None)
            statuses.append({
                "job_id": getattr(job, "id", ""),
                "status": getattr(job, "status", ""),
                "base_model": getattr(job, "model", ""),
                "fine_tuned_model": getattr(job, "fine_tuned_model", None),
                "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
                "error": error,
            })
        return statuses

    async def _list_jobs_mistral(self, limit: int) -> list[dict[str, Any]]:
        response = await self.provider.client.fine_tuning_jobs.list()
        statuses = []
        for job in response.data[:limit]:
            error = None
            if hasattr(job, "error") and job.error:
                error = getattr(job.error, "message", str(job.error))
            statuses.append({
                "job_id": getattr(job, "id", ""),
                "status": getattr(job, "status", ""),
                "base_model": getattr(job, "model", ""),
                "fine_tuned_model": getattr(job, "fine_tuned_model", None),
                "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
                "error": error,
            })
        return statuses

    async def _list_jobs_together(self, limit: int) -> list[dict[str, Any]]:
        response = await self.provider.client.get(TOGETHER_FINETUNE_ENDPOINT, params={"limit": limit})
        response.raise_for_status()
        result = response.json()
        statuses = []
        for job in result.get("data", []):
            statuses.append({
                "job_id": job.get("id", ""),
                "status": job.get("status", ""),
                "base_model": job.get("model", ""),
                "fine_tuned_model": job.get("fine_tuned_model"),
                "trained_tokens": job.get("trained_tokens", 0) or 0,
                "error": job.get("error"),
            })
        return statuses

    async def _list_jobs_bedrock(self, limit: int) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.provider.client.list_model_customization_jobs(maxResults=limit),
        )
        statuses = []
        for job in response.get("modelCustomizationJobSummaries", []):
            status_raw = job.get("status", "")
            status = BEDROCK_STATUS_MAP.get(status_raw, status_raw.lower())
            model_id = job.get("baseModelIdentifier", "")
            model_name = model_id.split("/")[-1] if "/" in model_id else model_id
            statuses.append({
                "job_id": job.get("jobArn", ""),
                "status": status,
                "base_model": model_name,
                "fine_tuned_model": job.get("outputModelArn"),
                "trained_tokens": 0,
                "error": job.get("failureMessage"),
            })
        return statuses

    async def _list_jobs_dashscope(self, limit: int) -> list[dict[str, Any]]:
        from dashscope import FineTunes

        response = await asyncio.to_thread(FineTunes.list, page_size=limit)
        if hasattr(response, "status_code") and response.status_code != 200:
            return []
        output = getattr(response, "output", None) or {}
        jobs = output.get("jobs", []) if isinstance(output, dict) else []
        return [self._parse_dashscope_response({"output": job}) for job in jobs]

    # ------------------------------------------------------------------
    # Provider-specific: List Events
    # ------------------------------------------------------------------

    async def _list_events(self, job_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if self.provider_type == "openai":
            return await self._list_events_openai(job_id, limit)
        elif self.provider_type == "dashscope":
            return await self._list_events_dashscope(job_id, limit)
        else:
            return []

    async def _list_events_openai(self, job_id: str, limit: int) -> list[dict[str, Any]]:
        resp = await self.provider.client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id, limit=limit,
        )
        events = []
        for ev in reversed(resp.data):
            events.append({
                "created_at": ev.created_at,
                "level": ev.level,
                "message": ev.message,
                "data": ev.data.model_dump() if hasattr(ev.data, "model_dump") else ev.data,
            })
        return events

    async def _list_events_dashscope(self, job_id: str, limit: int) -> list[dict[str, Any]]:
        from dashscope import FineTunes

        try:
            response = await asyncio.to_thread(FineTunes.stream_events, job_id)
            events = []
            for event in response:
                if hasattr(event, "output") and event.output:
                    output = event.output
                    if isinstance(output, dict):
                        events.append({
                            "message": output.get("message", ""),
                            "level": output.get("level", "info"),
                            "data": output,
                        })
                    else:
                        events.append({"message": str(output), "level": "info", "data": {}})
                if len(events) >= limit:
                    break
            return events
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Provider-specific: Delete File
    # ------------------------------------------------------------------

    async def _delete_file(self, file_id: str) -> bool:
        if self.provider_type == "openai":
            await self.provider.client.files.delete(file_id)
            return True
        elif self.provider_type == "dashscope":
            from dashscope import Files

            try:
                await asyncio.to_thread(Files.delete, file_id=file_id)
                return True
            except Exception:
                return False
        else:
            return True

    # ------------------------------------------------------------------
    # DashScope Response Parser
    # ------------------------------------------------------------------

    def _parse_dashscope_response(self, response: Any) -> dict[str, Any]:
        output = getattr(response, "output", None) or {}
        if isinstance(output, dict):
            job_id = output.get("job_id", "")
            status = output.get("status", "")
            model = output.get("model", "") or output.get("base_model", "")
            fine_tuned_model = output.get("finetuned_output")
            usage = output.get("usage", 0) or 0
        else:
            job_id = getattr(output, "job_id", "")
            status = getattr(output, "status", "")
            model = getattr(output, "model", "") or getattr(output, "base_model", "")
            fine_tuned_model = getattr(output, "finetuned_output", None)
            usage = getattr(output, "usage", 0) or 0

        error_msg = None
        if status == "FAILED":
            error = getattr(output, "error", None) or {}
            if isinstance(error, dict):
                error_msg = error.get("message", "Training failed")
            else:
                error_msg = str(error) if error else "Training failed"

        mapped_status = DASHSCOPE_STATUS_MAP.get(status, status.lower())

        return {
            "job_id": job_id,
            "status": mapped_status,
            "base_model": model,
            "fine_tuned_model": fine_tuned_model,
            "trained_tokens": usage,
            "error": error_msg,
        }

    # ------------------------------------------------------------------
    # Public Job Operations
    # ------------------------------------------------------------------

    async def create_job(self) -> FineTuneJobStatus:
        """上传训练文件并创建微调任务。

        Returns:
            任务状态快照（含 job_id），并写入 status_output_path
        """
        if errors := self.config.validate():
            raise ValueError(f"Invalid fine-tune config: {'; '.join(errors)}")

        job_cfg = self.config.job_config
        training_file = job_cfg["training_file"]
        validation_file = job_cfg.get("validation_file")

        for fpath in filter(None, [training_file, validation_file]):
            if errs := self.validate_jsonl(fpath):
                raise ValueError(f"Invalid fine-tune JSONL '{fpath}': {'; '.join(errs[:3])}")

        logger.info(f"Uploading training file: {training_file}")
        train_file_id = await self._upload_file(training_file)

        val_file_id = None
        if validation_file:
            logger.info(f"Uploading validation file: {validation_file}")
            val_file_id = await self._upload_file(validation_file)

        hyperparams = job_cfg.get("hyperparameters") or {}
        method = job_cfg.get("method", "supervised")

        logger.info("Creating fine-tune job:")
        logger.info(f"  model: {job_cfg['base_model']}")
        logger.info(f"  method: {method}")
        if job_cfg.get("suffix"):
            logger.info(f"  suffix: {job_cfg['suffix']}")
        if hyperparams:
            hp_desc = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
            logger.info(f"  hyperparameters: {hp_desc}")

        try:
            result = await self._create_job(
                training_file_id=train_file_id,
                model=job_cfg["base_model"],
                validation_file_id=val_file_id,
                method=method,
                suffix=job_cfg.get("suffix"),
                seed=job_cfg.get("seed"),
                hyperparameters=hyperparams if hyperparams else None,
            )
        except Exception:
            await self._safe_delete_file(train_file_id)
            if val_file_id is not None:
                await self._safe_delete_file(val_file_id)
            raise

        status = FineTuneJobStatus.from_dict_raw(result)
        logger.info(f"Job created: {status.job_id} (status: {status.status})")
        self.save_status(status)
        return status

    async def retrieve_job(self, job_id: str) -> FineTuneJobStatus:
        """查询任务最新状态。"""
        result = await self._retrieve_job(job_id)
        return FineTuneJobStatus.from_dict_raw(result)

    async def wait_for_completion(
        self,
        job_id: str | None = None,
        interval: int | None = None,
        timeout: int | None = None,
    ) -> FineTuneJobStatus:
        """轮询任务直至终态或超时。"""
        job_id = job_id or self._load_saved_job_id()
        if not job_id:
            raise ValueError("No job_id provided and no saved status found")

        interval = interval or self.config.poll_interval
        timeout = timeout or self.config.poll_timeout
        start = time.monotonic()
        seen_event_messages: set[str] = set()

        logger.info(f"Polling job {job_id} every {interval}s (timeout {timeout}s)")

        while True:
            try:
                status = await self.retrieve_job(job_id)
                await self._log_new_events(job_id, seen_event_messages)
            except Exception as e:
                if time.monotonic() - start > timeout:
                    return FineTuneJobStatus(
                        job_id=job_id, status="unknown",
                        error=f"polling failed after timeout: {e}",
                    )
                logger.warning(f"Retrieve failed ({e}), retrying in {interval}s...")
                await asyncio.sleep(interval)
                continue

            logger.info(
                f"[{job_id}] status: {status.status}, trained_tokens: {status.trained_tokens}"
            )

            if status.status in TERMINAL_STATUSES:
                if status.status == "succeeded":
                    logger.info(f"Fine-tuned model: {status.fine_tuned_model}")
                else:
                    logger.error(f"Job ended with status '{status.status}': {status.error}")
                self.save_status(status)
                return status

            if time.monotonic() - start > timeout:
                status.error = f"polling timed out after {timeout}s (job still running)"
                logger.error(status.error)
                logger.error(f"Re-check later via: python src/finetune.py --status {job_id}")
                self.save_status(status)
                return status

            await asyncio.sleep(interval)

    async def cancel_job(self, job_id: str) -> FineTuneJobStatus:
        """取消任务。"""
        result = await self._cancel_job(job_id)
        status = FineTuneJobStatus.from_dict_raw(result)
        logger.info(f"Job cancelled: {job_id} (status: {status.status})")
        self.save_status(status)
        return status

    async def list_jobs(self, limit: int = 10) -> list[FineTuneJobStatus]:
        """列出最近的微调任务。"""
        results = await self._list_jobs(limit=limit)
        statuses = [FineTuneJobStatus.from_dict_raw(r) for r in results]
        for s in statuses:
            logger.info(f"{s.job_id}  {s.status:<12}  {s.fine_tuned_model or s.base_model}")
        return statuses

    async def get_events(self, job_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """导出任务事件日志。"""
        events = await self._list_events(job_id=job_id, limit=limit)
        logger.info(f"Fetched {len(events)} events for job {job_id}")
        return events

    # ------------------------------------------------------------------
    # Status Persistence
    # ------------------------------------------------------------------

    def save_status(self, status: FineTuneJobStatus) -> None:
        """保存任务状态到 status_output_path。"""
        out_path = Path(self.config.status_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = status.to_dict()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_saved_job_id(self) -> str:
        """从 status_output_path 读取上次保存的 job_id。"""
        path = Path(self.config.status_output_path)
        if not path.exists():
            return ""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("job_id", "")
        except Exception as e:
            logger.warning(f"Failed to load saved status from {path}: {e}")
            return ""

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _log_new_events(self, job_id: str, seen: set[str]) -> None:
        """打印轮询期间新增的事件消息。"""
        try:
            events = await self._list_events(job_id=job_id, limit=20)
            for ev in events:
                msg = ev.get("message", "")
                if msg and msg not in seen:
                    seen.add(msg)
                    level = ev.get("level", "info")
                    logger.info(f"  [event/{level}] {msg}")
        except Exception as e:
            logger.debug(f"Failed to fetch events: {e}")

    async def _safe_delete_file(self, file_id: str) -> None:
        """尽力删除已上传的文件，失败仅警告。"""
        try:
            await self._delete_file(file_id)
            logger.info(f"Orphan file deleted: {file_id}")
        except Exception as e:
            logger.warning(f"Failed to delete orphan file {file_id}: {e}")
