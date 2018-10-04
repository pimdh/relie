import torch
from relie.geometry import tetrahedron
from relie.geometry import invariant_loss
from numpy.testing import assert_array_equal
from relie.utils.so3_tools import so3_uniform_random


def test_invariant_loss():
    x = torch.from_numpy(tetrahedron.coordinates()).float()
    permutations = torch.from_numpy(tetrahedron.permutations()).float()
    rotations = torch.from_numpy(tetrahedron.rotations()).float()
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
