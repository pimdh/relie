from numpy.testing import assert_array_equal
from relie.geometry import *


def test_rotations():
    permutations = permutation_matrices(tetrahedron_permutations())
    rotations = rotation_matrices(tetrahedron_coordinates(), tetrahedron_permutations())
    coords = tetrahedron_coordinates()

    for p, r in zip(permutations, rotations):
        assert_array_equal(p @ coords, coords @ r.T)
