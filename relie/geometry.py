from itertools import permutations as permutation_fn

import numpy as np
import torch


def cyclic_coordinates(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([
        np.cos(theta),
        np.sin(theta),
        np.zeros_like(theta)
    ], 1)


def cyclic_permutations(n):
    l = list(range(n))
    return [l[-i:] + l[:-i] for i in range(n)]


def invariant_loss(x, y, symmetry):
    """
    Finds permutation invariant loss, for fixed set of allowed permutations.
    Computes picking minimal loss under all permutations.

    Uses L2 loss.
    :param x: Input of shape (..., d)
    :param y: Output of shape (..., d)
    :param symmetry: Symmetry transformation matrices of shape (n, d, d).
        Must include identity
    :return: Loss of shape (...)
    """
    batch_shape = x.shape[:-1]
    d = x.shape[-1]
    x = x.view(-1, d)  # [b, d]
    y = y.view(-1, d)  # [b, d]
    x_transformed = torch.einsum('nde,be->nbd', [symmetry, x])
    diff = (y - x_transformed).pow(2).sum(2)  # [n, b]
    min_diff = torch.min(diff, dim=0)[0]  # [b]
    return min_diff.view(batch_shape)


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


def permutation_matrices(permutations):
    """
    Create permutation matrices from permutation lists.
    :param permutations: (n_sym, n_points)
    :return: (n_sym, n_points, n_points)
    """
    n = len(next(iter(permutations)))
    eye = np.eye(n)
    return np.stack([eye[p, :] for p in permutations])


def rotation_matrices(coordinates, permutations):
    """
    Return corresponding rotation matrices by solving Linear system with pinv.
    :param coordinates: (n_points, d)
    :param permutations: (n_sym, n_points)
    :return: Array of shape (n_sym, d, d)
    """
    x = coordinates
    perms = permutation_matrices(permutations)
    rs = []
    for perm in perms:
        x_permuted = perm @ x
        # Y = X @ R.T  => R.T = X^+ @ Y
        r = (np.linalg.pinv(x) @ x_permuted).T

        rs.append(r)
    rs = np.stack(rs)
    # For 2D subspaces, fix
    if (x[:, 2] == 0).all():
        rs[:, 2, :] = 0.
        rs[:, :, 2] = 0.
        rs[:, 2, 2] = 1.
    return rs