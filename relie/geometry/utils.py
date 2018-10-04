import numpy as np


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
    return np.stack(rs)
