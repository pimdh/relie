"""
Symmetries of Tetrahedron
Equals Permutation group of order 4.
"""
import numpy as np
from itertools import permutations as permutation_fn


def tetrahedron_coordinates():
    return np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=np.float32)


def tetrahedron_permutations():
    return permutation_fn(range(4))
