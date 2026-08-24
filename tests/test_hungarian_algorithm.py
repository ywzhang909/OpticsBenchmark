"""
Optis Benchmark - Hungarian Algorithm Utils Tests

Tests for optimal assignment using the Hungarian algorithm.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "algorithm"))

from hungarian_algorithm_utils import hungarian_match

# =============================================================================
# Classes
# =============================================================================


class TestHungarianMatch:
    """Tests for hungarian_match function."""

    def test_match_equal_sizes(self):
        """Test matching with equal n and m."""
        sim = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 2
        assert total == pytest.approx(1.7, rel=1e-5)

    def test_match_more_predictions(self):
        """Test matching when predictions > gold (n > m) — returns min(n,m) assignments."""
        sim = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.4]], dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 2  # min(3, 2) = 2
        assigned_cols = [c for _, c in assignments]
        assert len(set(assigned_cols)) == len(assigned_cols)  # no duplicate gold
        assert total > 0

    def test_match_more_gold(self):
        """Test matching when gold > predictions (n < m)."""
        sim = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]], dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 2
        assert total > 0

    def test_match_single_element(self):
        """Test matching with 1x1 matrix."""
        sim = np.array([[0.75]], dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert assignments == [(0, 0)]
        assert total == pytest.approx(0.75, rel=1e-5)

    def test_match_identity_matrix(self):
        """Test matching with identity matrix (diagonal dominance)."""
        sim = np.eye(3, dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 3
        assert total == pytest.approx(3.0, rel=1e-5)

    def test_match_all_zeros(self):
        """Test matching with all-zero similarity."""
        sim = np.zeros((3, 3), dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 3
        assert total == pytest.approx(0.0, rel=1e-5)

    def test_match_all_ones(self):
        """Test matching with all-ones similarity."""
        sim = np.ones((3, 3), dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 3
        assert total == pytest.approx(3.0, rel=1e-5)

    def test_match_rectangular_2x3(self):
        """Test matching with 2x3 rectangular matrix."""
        sim = np.array([[0.5, 0.9, 0.3], [0.8, 0.4, 0.7]], dtype=np.float64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 2
        # Each row assigned to a distinct column
        assigned_cols = [c for _, c in assignments]
        assert len(set(assigned_cols)) == len(assigned_cols)
        assert total > 0

    def test_match_returns_float_score(self):
        """Test that total_score is a native float."""
        sim = np.array([[0.5, 0.3], [0.2, 0.9]], dtype=np.float64)
        _, total = hungarian_match(sim)
        assert isinstance(total, float)

    def test_match_int_matrix(self):
        """Test matching with integer similarity matrix."""
        sim = np.array([[1, 0], [0, 1]], dtype=np.int64)
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 2
        assert total == pytest.approx(2.0, rel=1e-5)

    def test_match_non_square_4x2(self):
        """Test matching with 4x2 matrix (more predictions than gold) — returns min(4,2) = 2."""
        sim = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]], dtype=np.float64
        )
        assignments, total = hungarian_match(sim)
        assert len(assignments) == 2  # min(4, 2) = 2
        assigned_cols = [c for _, c in assignments]
        assert len(set(assigned_cols)) == 2
        assert total > 0
