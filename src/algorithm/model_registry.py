"""
GPU Model Registry — Unified lifecycle management for all GPU models.

All GPU-resident models should be loaded through this registry for tracking
and explicit unloading. Evaluators load models via ModelRegistry and unload
after evaluation batches to prevent multiple large models from consuming
GPU memory simultaneously (CUDA OOM).
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
    """Thread-safe singleton registry for managing GPU model loading, caching, and unloading.

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
        """Get a cached model or create it via ``loader``.

        Args:
            key: Unique model identifier (category:model_name).
            loader: No-argument factory function that returns the model object.
            force_reload: When True, ignore cache and force reload.

        Returns:
            Cached or newly loaded model object.
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
        """Unload the specified model and release GPU memory.

        After deleting the reference, executes ``gc.collect()`` +
        ``torch.cuda.empty_cache()`` to force reclaim GPU memory fragments.
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
        """Unload all registered models."""
        with self._lock:
            keys = list(self._registry.keys())
        for key in keys:
            self.unload(key)

    def is_loaded(self, key: str) -> bool:
        """Check if the specified model is loaded."""
        return key in self._registry

    def get_loaded_keys(self) -> list[str]:
        """Return all currently loaded model keys."""
        return list(self._registry.keys())

    def gpu_memory_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current GPU memory status for debugging and logging."""
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


# Process-level singleton
model_registry = ModelRegistry()
