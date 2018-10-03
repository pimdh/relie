from numpy.testing import assert_array_equal
from relie.geometry.tetrahedron import *


def test_rotations():
    perms = permutations()
    rs = rotations()
    coords = coordinates()

    for p, r in zip(perms, rs):
        assert_array_equal(p @ coords, coords @ r.T)
