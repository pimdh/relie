"""
Symmetries of Tetrahedron
Equals Permutation group of order 4.
"""
import numpy as np
from itertools import permutations as permutation_fn
from .utils import permutation_matrices


def tetrahedron_coordinates():
    return np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=np.float32)


def tetrahedron_permutations():
    """
    All permutations with positive determinant permutation matrix.
    :return: Array of shape (4!/2, 4)
    """
    permutations = list(permutation_fn(range(4)))
    matrices = permutation_matrices(permutations)
    return [p for p, m in zip(permutations, matrices) if np.linalg.det(m) > 0]

