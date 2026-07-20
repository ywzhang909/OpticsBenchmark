"""
GPU 模型注册表 — 所有 GPU 模型的统一生命周期管理。

所有 GPU 常驻模型应通过此注册表加载，以便追踪和显式释放。
评估器通过 ModelRegistry 加载模型，评估批次结束后卸载，
避免多个大型模型同时占用显存导致 CUDA OOM。
"""
from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any

from src.utils import logger

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


class ModelRegistry:
    """线程安全的单例注册表，管理 GPU 模型的加载、缓存和卸载。

    Key naming convention: ``"{category}:{model_name}"``
    e.g. ``"sentence_embedder:BAAI/bge-m3"``, ``"bert_scorer:roberta-large"``.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        force_reload: bool = False,
    ) -> Any:
        """获取已缓存的模型，或通过 ``loader`` 创建并缓存。

        Args:
            key: 模型唯一标识（类别:模型名）。
            loader: 无参工厂函数，返回模型对象。
            force_reload: 为 True 时忽略缓存，强制重新加载。

        Returns:
            缓存或新加载的模型对象。
        """
        if not force_reload and key in self._registry:
            return self._registry[key]
        with self._lock:
            if not force_reload and key in self._registry:
                return self._registry[key]
            logger.debug(f"Loading model: {key}")
            model = loader()
            self._registry[key] = model
            return model

    def unload(self, key: str) -> None:
        """卸载指定模型并释放 GPU 显存。

        删除引用后执行 ``gc.collect()`` + ``torch.cuda.empty_cache()``
        强制回收显存碎片。
        """
        with self._lock:
            if key not in self._registry:
                return
            logger.debug(f"Unloading model: {key}")
            del self._registry[key]
        gc.collect()
        if _CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """卸载所有已注册模型。"""
        with self._lock:
            keys = list(self._registry.keys())
        for key in keys:
            self.unload(key)

    def is_loaded(self, key: str) -> bool:
        """检查指定模型是否已加载。"""
        return key in self._registry

    def get_loaded_keys(self) -> list[str]:
        """返回当前已加载的所有模型键名。"""
        return list(self._registry.keys())

    def gpu_memory_snapshot(self) -> dict[str, Any]:
        """返回当前 GPU 显存状态快照，用于调试和日志。"""
        if not _CUDA_AVAILABLE:
            return {"cuda_available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda_available": True,
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2),
            "loaded_models": self.get_loaded_keys(),
        }


# 进程级单例
model_registry = ModelRegistry()
