import torch
from numpy.testing import assert_array_almost_equal, assert_array_equal

from relie.geometry import permutation_matrices, cyclic_permutations, rotation_matrices, cyclic_coordinates, \
    tetrahedron_coordinates, tetrahedron_permutations, invariant_loss
from relie.utils.so3_tools import so3_uniform_random


def test_rotations():
    for n in range(2, 10):
        permutations = permutation_matrices(cyclic_permutations(n))
        rotations = rotation_matrices(cyclic_coordinates(n), cyclic_permutations(n))
        coords = cyclic_coordinates(n)

        for p, r in zip(permutations, rotations):
            assert_array_almost_equal(p @ coords, coords @ r.T)


def test_invariant_loss():
    x = torch.from_numpy(tetrahedron_coordinates()).float()
    permutations = torch.from_numpy(permutation_matrices(tetrahedron_permutations())).float()
    rotations = torch.from_numpy(rotation_matrices(tetrahedron_coordinates(), tetrahedron_permutations())).float()
    for p in permutations:
        y = p @ x
        l = invariant_loss(x, y, rotations)
        assert_array_equal(l, torch.zeros(len(x)))

    for r in rotations:
        y = x @ r.t()
        l = invariant_loss(x, y, rotations)
        assert_array_equal(l, torch.zeros(len(x)))

    torch.manual_seed(0)
    for r in so3_uniform_random(5):
        y = x @ r.t()
        l = invariant_loss(x, y, rotations)
        assert (l > .01).all()


def test_tetrahedron_rotations():
    permutations = permutation_matrices(tetrahedron_permutations())
    rotations = rotation_matrices(tetrahedron_coordinates(), tetrahedron_permutations())
    coords = tetrahedron_coordinates()

    for p, r in zip(permutations, rotations):
        assert_array_equal(p @ coords, coords @ r.T)