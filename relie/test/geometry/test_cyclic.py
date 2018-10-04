from numpy.testing import assert_array_almost_equal
from relie.geometry import *


def test_rotations():
    for n in range(2, 10):
        permutations = permutation_matrices(cyclic_permutations(n))
        rotations = rotation_matrices(cyclic_coordinates(n), cyclic_permutations(n))
        coords = cyclic_coordinates(n)

        for p, r in zip(permutations, rotations):
            assert_array_almost_equal(p @ coords, coords @ r.T)
