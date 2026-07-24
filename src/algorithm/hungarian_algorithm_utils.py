# hungarian_algorithm_utils.py
"""
Hungarian Algorithm Utilities Module

This module provides functionality to solve the optimal assignment problem
for BERTScore matrices using the Hungarian algorithm. The optimization
objective is to maximize total BERTScore by minimizing (1 - BERTScore).
"""

import argparse
import csv
import json
import os

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


def main():
    parser = argparse.ArgumentParser(description="Hungarian Algorithm — Optimal Assignment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--matrix",
        type=str,
        help="Similarity matrix as semicolon-delimited rows, comma-delimited columns, e.g. '0.9,0.1;0.1,0.8'",
    )
    group.add_argument("--matrix-file", type=str, help="Path to JSON file containing 2D list or CSV file")
    args = parser.parse_args()

    if args.matrix_file:
        _, ext = os.path.splitext(args.matrix_file)
        if ext.lower() == ".csv":
            matrix = []
            with open(args.matrix_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    matrix.append([float(x) for x in row])
        else:
            with open(args.matrix_file, "r") as f:
                matrix = json.load(f)
    else:
        matrix = [[float(x) for x in row.split(",")] for row in args.matrix.split(";")]

    sim_matrix = np.array(matrix, dtype=np.float64)
    assignments, total_score = hungarian_match(sim_matrix)

    result = {
        "assignments": [[int(a), int(b)] for a, b in assignments],
        "total_score": round(total_score, 4),
        "matrix_shape": list(sim_matrix.shape),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
