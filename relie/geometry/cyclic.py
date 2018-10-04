"""
Cyclic object
"""
import numpy as np


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


