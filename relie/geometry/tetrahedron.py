"""
Symmetries of Tetrahedron
Equals Permutation group of order 4.
"""
import numpy as np
from itertools import permutations as permutation_fn


def coordinates():
    return np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=np.float32)


def permutations():
    """
    Compute permutation matrices
    :return: Array of shape (4!, 4, 4)
    """
    perms = permutation_fn(range(4))
    eye = np.eye(4)
    return np.stack([eye[p, :] for p in perms])


def rotations():
    """
    Return corresponding rotation matrices by solving Linear system with pinv.
    :return: Array of shape (4!, 3, 3)
    """
    x = coordinates()
    perms = permutations()
    rs = []
    for perm in perms:
        x_permuted = perm @ x
        # Y = X @ R.T  => R.T = X^+ @ Y
        r = (np.linalg.pinv(x) @ x_permuted).T
        rs.append(r)
    return np.stack(rs)



