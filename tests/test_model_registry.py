"""
OptiS Benchmark - Model Registry Tests

Tests for ModelRegistry GPU model lifecycle management.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithm.model_registry import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def setup_method(self):
        """Create a fresh registry for each test."""
        self.registry = ModelRegistry()

    def test_get_or_load_caches(self):
        """Test that get_or_load caches the result and only calls loader once."""
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return MagicMock()

        # First call should load
        result1 = self.registry.get_or_load("test:model1", loader)
        assert call_count == 1
        assert result1 is not None

        # Second call should return cached version
        result2 = self.registry.get_or_load("test:model1", loader)
        assert call_count == 1
        assert result1 is result2

    def test_unload_releases(self):
        """Test that unload removes the model from cache."""
        def loader():
            return MagicMock()

        self.registry.get_or_load("test:model1", loader)
        assert self.registry.is_loaded("test:model1")

        self.registry.unload("test:model1")
        assert not self.registry.is_loaded("test:model1")

    def test_unload_all(self):
        """Test that unload_all removes all cached models."""
        def loader():
            return MagicMock()

        self.registry.get_or_load("test:model1", loader)
        self.registry.get_or_load("test:model2", loader)
        self.registry.get_or_load("test:model3", loader)

        assert len(self.registry.get_loaded_keys()) == 3

        self.registry.unload_all()
        assert len(self.registry.get_loaded_keys()) == 0

    def test_force_reload(self):
        """Test that force_reload bypasses cache."""
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return MagicMock()

        # First call
        result1 = self.registry.get_or_load("test:model1", loader)
        assert call_count == 1

        # Force reload
        result2 = self.registry.get_or_load("test:model1", loader, force_reload=True)
        assert call_count == 2
        assert result1 is not result2

    def test_thread_safety(self):
        """Test that concurrent get_or_load is thread-safe."""
        call_count = 0
        lock = threading.Lock()

        def loader():
            nonlocal call_count
            with lock:
                call_count += 1
            return MagicMock()

        # Create multiple threads trying to load the same key
        results = []
        threads = []
        for _ in range(10):
            def task():
                result = self.registry.get_or_load("test:model1", loader)
                results.append(result)

            thread = threading.Thread(target=task)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should get the same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)
        # Loader should only be called once
        assert call_count == 1

    def test_get_loaded_keys(self):
        """Test that get_loaded_keys returns correct list."""
        def loader():
            return MagicMock()

        self.registry.get_or_load("test:model1", loader)
        self.registry.get_or_load("test:model2", loader)

        keys = self.registry.get_loaded_keys()
        assert len(keys) == 2
        assert "test:model1" in keys
        assert "test:model2" in keys

    def test_unload_nonexistent_key(self):
        """Test that unloading a nonexistent key does not raise error."""
        # Should not raise
        self.registry.unload("nonexistent:key")

    def test_is_loaded(self):
        """Test is_loaded returns correct status."""
        def loader():
            return MagicMock()

        assert not self.registry.is_loaded("test:model1")
        self.registry.get_or_load("test:model1", loader)
        assert self.registry.is_loaded("test:model1")
        self.registry.unload("test:model1")
        assert not self.registry.is_loaded("test:model1")

    @patch("algorithm.model_registry._CUDA_AVAILABLE", False)
    def test_gpu_memory_snapshot_no_cuda(self):
        """Test gpu_memory_snapshot when CUDA is not available."""
        snapshot = self.registry.gpu_memory_snapshot()
        assert snapshot == {"cuda_available": False}

    @patch("algorithm.model_registry._CUDA_AVAILABLE", True)
    @patch("algorithm.model_registry.torch")
    def test_gpu_memory_snapshot_with_cuda(self, mock_torch):
        """Test gpu_memory_snapshot when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1e9
        mock_torch.cuda.memory_reserved.return_value = 2e9
        mock_torch.cuda.mem_get_info.return_value = (20e9, 24e9)

        snapshot = self.registry.gpu_memory_snapshot()
        assert snapshot["cuda_available"] is True
        assert snapshot["allocated_gb"] == 1.0
        assert snapshot["reserved_gb"] == 2.0
        assert snapshot["free_gb"] == 20.0
        assert snapshot["total_gb"] == 24.0

    def test_process_singleton(self):
        """Test that the module-level model_registry is a singleton."""
        from algorithm.model_registry import model_registry as registry1
        from algorithm.model_registry import model_registry as registry2
        assert registry1 is registry2


class TestSortEvaluatorsByPriority:
    """Tests for sort_evaluators_by_priority function."""

    def test_sort_by_priority(self):
        """Test that evaluators are sorted by priority from config."""
        from evaluators.factory import sort_evaluators_by_priority

        # Create mock evaluators
        eval1 = MagicMock()
        eval2 = MagicMock()
        eval3 = MagicMock()

        named_evaluators = [
            ("exact_match", eval1),
            ("citation", eval2),
            ("bert_score", eval3),
        ]

        config = {
            "eval_metrics": {
                "exact_match": {"priority": 4},
                "citation": {"priority": 1},
                "bert_score": {"priority": 2},
            }
        }

        sorted_evaluators = sort_evaluators_by_priority(named_evaluators, config)
        names = [name for name, _ in sorted_evaluators]

        assert names == ["citation", "bert_score", "exact_match"]

    def test_default_priority(self):
        """Test that evaluators without priority get default 999."""
        from evaluators.factory import sort_evaluators_by_priority

        eval1 = MagicMock()
        eval2 = MagicMock()

        named_evaluators = [
            ("exact_match", eval1),
            ("citation", eval2),
        ]

        config = {
            "eval_metrics": {
                "exact_match": {},  # No priority
                "citation": {"priority": 1},
            }
        }

        sorted_evaluators = sort_evaluators_by_priority(named_evaluators, config)
        names = [name for name, _ in sorted_evaluators]

        # citation (priority 1) should come before exact_match (default 999)
        assert names == ["citation", "exact_match"]

    def test_stable_sort(self):
        """Test that evaluators with same priority maintain original order."""
        from evaluators.factory import sort_evaluators_by_priority

        eval1 = MagicMock()
        eval2 = MagicMock()
        eval3 = MagicMock()

        named_evaluators = [
            ("exact_match", eval1),
            ("rouge", eval2),
            ("bert_score", eval3),
        ]

        config = {
            "eval_metrics": {
                "exact_match": {"priority": 1},
                "rouge": {"priority": 1},
                "bert_score": {"priority": 1},
            }
        }

        sorted_evaluators = sort_evaluators_by_priority(named_evaluators, config)
        names = [name for name, _ in sorted_evaluators]

        # All have same priority, should maintain original order
        assert names == ["exact_match", "rouge", "bert_score"]
