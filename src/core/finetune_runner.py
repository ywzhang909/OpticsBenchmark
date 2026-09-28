"""
Optis Benchmark - Fine-tune Runner Module

Multi-provider fine-tuning job manager, aligned with the architectural style of LLMPredRunner.
Implements fine-tuning via a thin Provider client wrapper plus inline business logic.

Execution flow:
    1. validate()      - Validate config and training files
    2. create_job()    - Upload JSONL -> create fine-tuning job
    3. wait_for_completion() - Poll status until a terminal state (optional)
    4. save_status()   - Persist status (job id / ft model name)

After fine-tuning completes, fill the fine_tuned_model name recorded in status_output_path
into model.name in configs/llm/GPT_OpenAI.yaml to reuse the existing inference/evaluation pipeline.
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

# Terminal statuses (stop polling once reached)
TERMINAL_STATUSES: set[str] = {"succeeded", "failed", "cancelled"}

# Maximum suffix length (OpenAI official limit)
MAX_SUFFIX_LENGTH = 18

# Polling defaults
DEFAULT_POLL_INTERVAL = 30
DEFAULT_POLL_TIMEOUT = 86400

# Bedrock fine-tuning status mapping
BEDROCK_STATUS_MAP: dict[str, str] = {
    "InProgress": "running",
    "Completed": "succeeded",
    "Failed": "failed",
    "Stopping": "cancelling",
    "Stopped": "cancelled",
}

# DashScope status mapping
DASHSCOPE_STATUS_MAP: dict[str, str] = {
    "PENDING": "queued",
    "QUEUING": "queued",
    "RUNNING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "CANCELED": "cancelled",
    "CANCELING": "cancelled",
}

# Together AI status mapping (official SDK 返回值 -> runner 规范状态)
TOGETHER_STATUS_MAP: dict[str, str] = {
    "pending": "queued",
    "queued": "queued",
    "running": "running",
    "compressing": "running",
    "uploading": "running",
    "cancel_requested": "cancelling",
    "cancelled": "cancelled",
    "completed": "succeeded",
    "error": "failed",
    "failed": "failed",
}

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
    """Fine-tuning job status snapshot."""

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
        """Create a status snapshot from a dictionary returned by the adapter."""
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
            path: YAML config file path

        Returns:
            A FineTuneRunnerConfig instance

        Raises:
            FileNotFoundError: when the config file does not exist
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
        """Validate config correctness.

        Returns:
            List of errors; an empty list means the config is valid
        """
        errors: list[str] = []

        if not self.provider_config.get("api_key"):
            provider_type = self.provider_config.get("type", "")
            if provider_type not in ("bedrock",):
                errors.append("llm.provider.api_key is empty (check env var expansion)")

        job = self.job_config

        # 验证 base_model 存在且非空
        if not job.get("base_model"):
            errors.append("fine_tuning.base_model is required")

        # 验证 training_file 存在且路径有效
        training_file = job.get("training_file", "")
        if not training_file:
            errors.append("fine_tuning.training_file is required")
        elif not Path(training_file).exists():
            errors.append(f"training_file not found: {training_file}")
        elif not Path(training_file).is_file():
            errors.append(f"training_file is not a file: {training_file}")

        # 验证 validation_file 如果存在则路径有效
        if validation_file := job.get("validation_file"):
            if not Path(validation_file).exists():
                errors.append(f"validation_file not found: {validation_file}")
            elif not Path(validation_file).is_file():
                errors.append(f"validation_file is not a file: {validation_file}")

        # 验证 suffix 长度（如果存在）
        suffix = str(job.get("suffix") or "")
        if len(suffix) > MAX_SUFFIX_LENGTH:
            errors.append(f"suffix exceeds {MAX_SUFFIX_LENGTH} chars: '{suffix}'")

        return errors

    @property
    def poll_interval(self) -> int:
        """Polling interval (seconds)."""
        return int(self.execution_config.get("poll_interval", DEFAULT_POLL_INTERVAL))

    @property
    def poll_timeout(self) -> int:
        """Polling timeout (seconds)."""
        return int(self.execution_config.get("poll_timeout", DEFAULT_POLL_TIMEOUT))

    @property
    def status_output_path(self) -> str:
        """Path where status is persisted."""
        return self.execution_config.get("status_output_path", "results/finetune/job_status.json")


# =============================================================================
# Classes
# =============================================================================


class FineTuneRunner:
    """Multi-provider fine-tuning job manager.

    Implements fine-tuning via a thin Provider client wrapper plus inline business logic.

    Execution flow:
        1. setup() - Create a Provider instance
        2. create_job() / wait_for_completion() / cancel_job() and other job operations
        3. teardown() - Clean up resources
    """

    def __init__(self, config: FineTuneRunnerConfig):
        """Initialize the Runner.

        Args:
            config: Fine-tuning run configuration
        """
        self.config = config
        self.provider: Any = None
        self.provider_type: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create a Provider instance."""
        self.provider_type = self.config.provider_config.get("type", "")
        self.provider = create_provider(self.config.provider_config)
        logger.info(f"Provider: {type(self.provider).__name__}")

    async def teardown(self) -> None:
        """Clean up resources."""
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
        """Validate the format and message structure of a fine-tuning JSONL file.

        Args:
            path: JSONL file path

        Returns:
            List of errors; an empty list means the file is valid
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
        """Upload a file to the fine-tuning provider.

        Args:
            file_path: Local file path

        Returns:
            File ID or path
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
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

    async def _upload_file_openai(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_obj = await self.provider.client.files.create(file=f, purpose="fine-tune")
        return file_obj.id

    async def _upload_file_mistral(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_obj = await self.provider.client.files.upload(
                file={
                    "file_name": Path(file_path).name,
                    "content": f
                },
                purpose="fine-tune"
            )
        return file_obj.id

    async def _upload_file_together(self, file_path: str) -> str:

        file_obj = await self.provider.client.files.upload(
            file=file_path,
            purpose="fine-tune",
            check=True
        )
        file_id = file_obj.id

        # 轮询等待服务器端校验完成，直到 processing_status == "COMPLETED"
        # 才返回文件 id；出现 INVALID_FORMAT / FAILED 则报错并结束。
        # 参照官方: docs.together.ai/docs/fine-tuning/data-preparation
        deadline = time.monotonic() + 3600
        while True:
            meta = await self.provider.client.files.retrieve(file_id)
            status = meta.processing_status
            if status == "COMPLETED":
                break
            if status == "INVALID_FORMAT":
                raise ValueError(
                    f"file is not valid for fine-tuning: {meta.validation_report}"
                )
            if status == "FAILED":
                raise RuntimeError(
                    f"file processing did not complete: {status}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {self.config.poll_timeout}s waiting for "
                    f"file {file_id} to be processed (status: {status})"
                )
            await asyncio.sleep(5)

        return file_id

    async def _upload_file_bedrock(self, file_path: str) -> str:
        if file_path.startswith("s3://"):
            return file_path
        raise ValueError(
            "Bedrock fine-tuning requires uploading training data to S3. "
            "Use the AWS CLI: aws s3 cp <local_path> s3://<bucket>/<key>"
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
        job_cfg: dict[str, Any],
        training_file_id: str,
        validation_file_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a fine-tuning job.

        Args:
            job_cfg: Fine-tuning configuration (the fine_tuning.* block from YAML),
                from which each provider extracts only the keys relevant to its own API.
            training_file_id: ID/path of the uploaded training set.
            validation_file_id: Optional ID/path of the uploaded validation set.

        Returns:
            Standard job status dictionary
        """
        if self.provider_type == "openai":
            return await self._create_job_openai(
                job_cfg, training_file_id, validation_file_id,
            )
        elif self.provider_type == "mistral":
            return await self._create_job_mistral(
                job_cfg, training_file_id, validation_file_id,
            )
        elif self.provider_type == "together":
            return await self._create_job_together(
                job_cfg, training_file_id, validation_file_id,
            )
        elif self.provider_type == "bedrock":
            return await self._create_job_bedrock(
                job_cfg, training_file_id, validation_file_id,
            )
        elif self.provider_type == "dashscope":
            return await self._create_job_dashscope(
                job_cfg, training_file_id, validation_file_id,
            )
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

    async def _create_job_openai(
        self, job_cfg, training_file_id, validation_file_id=None,
    ) -> dict[str, Any]:
        model = job_cfg["base_model"]

        request: dict[str, Any] = {
            "training_file": training_file_id,
            "model": model,
        }
        if validation_file_id is not None:
            request["validation_file"] = validation_file_id
        if job_cfg.get("suffix", None):
            request["suffix"] = job_cfg.get("suffix")
        if job_cfg.get("seed", None):
            request["seed"] = job_cfg.get("seed")
        if job_cfg.get("metadata", None):
            request["metadata"] = job_cfg.get("metadata", None)
        if job_cfg.get("method", None):
            method_type = job_cfg.get("method")
            if job_cfg.get("hyperparameters", None):
                request["method"] = self._build_openai_method(
                    method_type,
                    job_cfg.get("hyperparameters"),
                    grader=job_cfg.get("grader"),
                )
            else:
                raise ValueError(
                    f"fine_tuning.method is '{method_type}' but "
                    "fine_tuning.hyperparameters is missing. "
                    "hyperparameters is required when a method is specified."
                )
        else:
            logger.info(
                "No fine-tune method specified; using OpenAI default method "
                "(no explicit 'method' body). Set 'fine_tuning.method' and "
                "'fine_tuning.hyperparameters' to customize "
                "(supervised / dpo / reinforcement)."
            )

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

    def _build_openai_method(
        self,
        method_type: str,
        hyperparameters: dict[str, Any],
        grader: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the OpenAI fine-tune 'method' request body for a given type.

        Mirrors the OpenAI SDK structure:
            supervised     -> {"type": "supervised",
                               "supervised": SupervisedMethod(
                                   hyperparameters=SupervisedHyperparameters(...))}
            dpo            -> {"type": "dpo",
                               "dpo": DpoMethod(
                                   hyperparameters=DpoHyperparameters(...))}
            reinforcement  -> {"type": "reinforcement",
                               "reinforcement": ReinforcementMethod(
                                   grader=StringCheckGrader(...),
                                   hyperparameters=ReinforcementHyperparameters(...))}

        Args:
            method_type: One of "supervised", "dpo", "reinforcement".
            hyperparameters: Hyperparameter dict (e.g. n_epochs/batch_size).
            grader: Optional grader config for "reinforcement" (StringCheckGrader,
                requires name/type/input/operation/reference).

        Returns:
            The "method" dict to be passed to fine_tuning.jobs.create().

        Raises:
            ValueError: unsupported method_type, or missing grader for reinforcement.
        """
        from openai.types.fine_tuning import (
            DpoHyperparameters,
            DpoMethod,
            ReinforcementHyperparameters,
            ReinforcementMethod,
            SupervisedHyperparameters,
            SupervisedMethod,
        )

        if method_type == "supervised":
            return {
                "type": "supervised",
                "supervised": SupervisedMethod(
                    hyperparameters=SupervisedHyperparameters(**hyperparameters),
                ),
            }
        elif method_type == "dpo":
            return {
                "type": "dpo",
                "dpo": DpoMethod(
                    hyperparameters=DpoHyperparameters(**hyperparameters),
                ),
            }
        elif method_type == "reinforcement":
            if not grader:
                raise ValueError(
                    "fine_tuning.grader is required when method is 'reinforcement'. "
                    "Provide a StringCheckGrader config "
                    "(name/type/input/operation/reference)."
                )
            from openai.types.graders import StringCheckGrader

            return {
                "type": "reinforcement",
                "reinforcement": ReinforcementMethod(
                    grader=StringCheckGrader(**grader),
                    hyperparameters=ReinforcementHyperparameters(**hyperparameters),
                ),
            }
        else:
            raise ValueError(f"Unsupported OpenAI fine-tune method: {method_type}")

    async def _create_job_mistral(
        self, job_cfg, training_file_id, validation_file_id=None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": job_cfg["base_model"],
            "training_files": [{"file_id": training_file_id, "weight": 1}],
            "hyperparameters" : job_cfg.get("hyperparameters", None),
            "auto_start" : job_cfg.get("auto_start", False),
            "integrations" : job_cfg.get("integrations", None),
        }
        if validation_file_id is not None:
            request["validation_files"] = [validation_file_id]

        job = await self.provider.client.fine_tuning_jobs.create(**request)

        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", str(job.error))

        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": job_cfg["base_model"],
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", 0) or 0,
            "error": error,
        }

    async def _create_job_together(
        self, job_cfg, training_file_id, validation_file_id=None,
    ) -> dict[str, Any]:

        request: dict[str, Any] = {
            "training_file": training_file_id,
            "model": job_cfg["base_model"],
            "packing" : job_cfg.get("packing", True),
            "max_seq_length" : job_cfg.get("max_seq_length", None),
            "n_epochs" : job_cfg.get("n_epochs", 1),
            "n_checkpoints" : job_cfg.get("n_checkpoints", 1),
            "n_evals" : job_cfg.get("n_evals", 0),
            "batch_size" : job_cfg.get("batch_size", "max"),
            "gradient_accumulation_steps" : job_cfg.get("gradient_accumulation_steps", 0),
            "learning_rate" : job_cfg.get("learning_rate", 0.00001),
            "warmup_ratio" : job_cfg.get("warmup_ratio", 0),
            "max_grad_norm" : job_cfg.get("max_grad_norm", 1),
            "weight_decay" : job_cfg.get("weight_decay", 0),
            "random_seed" : job_cfg.get("random_seed", None),
            "early_stopping_enabled" : job_cfg.get("early_stopping_enabled", False),
            "suffix" : job_cfg.get("suffix", None),
            "wandb_api_key" : job_cfg.get("wandb_api_key", None),
            "wandb_base_url" : job_cfg.get("wandb_base_url", None),
            "wandb_project_name" : job_cfg.get("wandb_project_name", None),
            "wandb_name" : job_cfg.get("wandb_name", None),
            "wandb_entity" : job_cfg.get("wandb_entity", None),

        }

        # 提取 lr_scheduler 配置（映射为 SDk 扁平参数）
        lr_scheduler = job_cfg.get("lr_scheduler")
        if lr_scheduler:
            # 验证 lr_scheduler_type
            lr_scheduler_type = lr_scheduler.get("lr_scheduler_type")
            if lr_scheduler_type not in ["linear", "cosine"]:
                raise ValueError(f"无效的 lr_scheduler_type: {lr_scheduler_type}。必须是 'linear' 或 'cosine'")

            # 验证 lr_scheduler_args
            lr_scheduler_args = lr_scheduler.get("lr_scheduler_args", {})
            min_lr_ratio = lr_scheduler_args.get("min_lr_ratio", 0)

            request["lr_scheduler_type"] = lr_scheduler_type
            request["min_lr_ratio"] = min_lr_ratio
            if lr_scheduler_type == "cosine":
                request["scheduler_num_cycles"] = lr_scheduler_args.get("num_cycles", 0.5)
        if validation_file_id is not None:
            request["validation_file"] = validation_file_id
        early_stopping_enabled = job_cfg.get("early_stopping_enabled", False)
        if early_stopping_enabled:
            request["early_stopping_enabled"] = early_stopping_enabled
            request["early_stopping_patience"] = job_cfg.get("early_stopping_patience", 2)
            request["early_stopping_min_delta"] = job_cfg.get("early_stopping_min_delta", 0)
            request["early_stopping_warmup_evals"] = job_cfg.get("early_stopping_warmup_evals", 1)

        training_method = job_cfg.get("training_method")
        if training_method:
            method = training_method.get("method", None)
            if method not in ["sft", "dpo"]:
                raise ValueError(f"无效的 training_method.method: {method}。必须是 'sft' 或 'dpo'")
            request["training_method"] = method
            if method == "sft":
                train_on_inputs = training_method.get("train_on_inputs", "auto")
                if train_on_inputs is not None:
                    request["train_on_inputs"] = train_on_inputs
            elif method == "dpo":
                dpo_beta = training_method.get("dpo_beta", 0.1)
                rpo_alpha = training_method.get("rpo_alpha", 0)
                dpo_normalize_logratios_by_length = training_method.get("dpo_normalize_logratios_by_length", False)
                simpo_gamma = training_method.get("simpo_gamma", 0)
                request["dpo_beta"] = dpo_beta
                request["rpo_alpha"] = rpo_alpha
                request["dpo_normalize_logratios_by_length"] = dpo_normalize_logratios_by_length
                request["simpo_gamma"] = simpo_gamma

        # 官方 SDK 使用 Omit 语义，剔除未设置的 None 值
        request = {k: v for k, v in request.items() if v is not None}

        response = await self.provider.client.fine_tuning.create(**request)

        return {
            "job_id": getattr(response, "id", ""),
            "status": getattr(response, "status", ""),
            "base_model": job_cfg["base_model"],
            "fine_tuned_model": getattr(response, "x_model_output_name", None)
                or getattr(response, "api_model_object_name", None),
            "trained_tokens": getattr(response, "token_count", 0) or 0,
            "error": None,
        }

    async def _create_job_bedrock(
        self, job_cfg, training_file_id, validation_file_id=None,
    ) -> dict[str, Any]:
        model = job_cfg["base_model"]
        suffix = job_cfg.get("suffix")
        hyperparameters = job_cfg.get("hyperparameters") or None

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
        self, job_cfg, training_file_id, validation_file_id=None,
    ) -> dict[str, Any]:
        from dashscope import FineTunes

        model = job_cfg["base_model"]
        seed = job_cfg.get("seed")
        hyperparameters = job_cfg.get("hyperparameters") or None
        training_type = job_cfg.get("training_type", "sft")
        suffix = job_cfg.get("suffix")
        job_name = job_cfg.get("job_name")
        model_name = job_cfg.get("model_name")

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
        if job_name:
            params["job_name"] = job_name
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
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

    async def _retrieve_job_openai(self, job_id: str) -> dict[str, Any]:
        job = await self.provider.client.fine_tuning.jobs.retrieve(job_id)
        error = None
        if hasattr(job, "error") and job.error:
            error = getattr(job.error, "message", None)
        return {
            "job_id": getattr(job, "id", ""),
            "status": getattr(job, "status", ""),
            "base_model": getattr(job, "model", ""),
            "fine_tuned_model": getattr(job, "model_object_revision_id", None),
            "trained_tokens": getattr(job, "token_count", 0),
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
        response = await self.provider.client.fine_tuning.retrieve(id=job_id)
        return {
            "job_id": getattr(response, "id", "") or getattr(response, "job_id", ""),
            "status": TOGETHER_STATUS_MAP.get(
                getattr(response, "status", ""), getattr(response, "status", "").lower()
            ),
            "base_model": getattr(response, "model", ""),
            "fine_tuned_model": getattr(response, "x_model_output_name", None)
                or getattr(response, "api_model_object_name", None),
            "trained_tokens": getattr(response, "token_count", 0) or 0,
            "error": None,
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
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

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
        response = await self.provider.client.fine_tuning.cancel(id=job_id)
        return {
            "job_id": getattr(response, "id", "") or getattr(response, "job_id", ""),
            "status": TOGETHER_STATUS_MAP.get(
                getattr(response, "status", ""), getattr(response, "status", "").lower()
            ),
            "base_model": getattr(response, "model", ""),
            "fine_tuned_model": None,
            "trained_tokens": getattr(response, "token_count", 0) or 0,
            "error": None,
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
        response = await self.provider.client.fine_tuning.list()
        statuses = []
        for job in response.data[:limit]:
            statuses.append({
                "job_id": getattr(job, "id", ""),
                "status": TOGETHER_STATUS_MAP.get(
                    getattr(job, "status", ""), getattr(job, "status", "").lower()
                ),
                "base_model": getattr(job, "model", ""),
                "fine_tuned_model": getattr(job, "x_model_output_name", None),
                "trained_tokens": getattr(job, "token_count", 0) or 0,
                "error": None,
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
        """Upload the training file and create a fine-tuning job.

        Returns:
            Job status snapshot (including job_id), also written to status_output_path
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
                job_cfg=job_cfg,
                training_file_id=train_file_id,
                validation_file_id=val_file_id,
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
        """Retrieve the latest status of a job."""
        result = await self._retrieve_job(job_id)
        return FineTuneJobStatus.from_dict_raw(result)

    async def wait_for_completion(
        self,
        job_id: str | None = None,
        interval: int | None = None,
        timeout: int | None = None,
    ) -> FineTuneJobStatus:
        """Poll the job until a terminal state or timeout."""
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
        """Cancel a job."""
        result = await self._cancel_job(job_id)
        status = FineTuneJobStatus.from_dict_raw(result)
        logger.info(f"Job cancelled: {job_id} (status: {status.status})")
        self.save_status(status)
        return status

    async def list_jobs(self, limit: int = 10) -> list[FineTuneJobStatus]:
        """List recent fine-tuning jobs."""
        results = await self._list_jobs(limit=limit)
        statuses = [FineTuneJobStatus.from_dict_raw(r) for r in results]
        for s in statuses:
            logger.info(f"{s.job_id}  {s.status:<12}  {s.fine_tuned_model or s.base_model}")
        return statuses

    async def get_events(self, job_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Export job events log."""
        events = await self._list_events(job_id=job_id, limit=limit)
        logger.info(f"Fetched {len(events)} events for job {job_id}")
        return events

    # ------------------------------------------------------------------
    # Status Persistence
    # ------------------------------------------------------------------

    def save_status(self, status: FineTuneJobStatus) -> None:
        """Save job status to status_output_path."""
        out_path = Path(self.config.status_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = status.to_dict()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_saved_job_id(self) -> str:
        """Read the job_id last saved in status_output_path."""
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
        """Print new event messages collected during polling."""
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
        """Best-effort deletion of an uploaded file; only warn on failure."""
        try:
            await self._delete_file(file_id)
            logger.info(f"Orphan file deleted: {file_id}")
        except Exception as e:
            logger.warning(f"Failed to delete orphan file {file_id}: {e}")
