# hungarian_algorithm_utils.py
"""
Hungarian Algorithm Utilities Module

This module provides functionality to solve the optimal assignment problem
for BERTScore matrices using the Hungarian algorithm. The optimization
objective is to maximize total BERTScore by minimizing (1 - BERTScore).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(sim_matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Solve optimal assignment for a similarity matrix using Hungarian algorithm.

    Given an n×m similarity matrix where element (i, j) is the similarity
    between prediction i and gold j, find the one-to-one matching
    that maximizes the total similarity.

    If n <= m, each prediction is assigned to a distinct gold.
    If n > m, each prediction is assigned to a distinct gold.

    Args:
        sim_matrix: An n×m numpy array of similarity values in [0, 1].

    Returns:
        Tuple of (assignments, total_score):
            assignments: List of (pred_idx, gold_idx) pairs representing
                the optimal matching.
            total_score: Sum of similarity for the optimal assignment.
    """
    cost_matrix = np.asarray(1 - sim_matrix, dtype=np.float64)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    assignments = list(zip(row_indices.tolist(), col_indices.tolist()))
    total_score = float(sim_matrix[row_indices, col_indices].sum())
    return assignments, total_score
