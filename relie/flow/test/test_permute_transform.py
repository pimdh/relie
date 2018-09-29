import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from relie.flow.permute_transform import PermuteTransform


def test_permute():
    d = 7
    batch_size = 64

    for _ in range(100):
        p = np.random.permutation(d)
        t = PermuteTransform(p)
        x = torch.randn(batch_size, d)
        y = torch.tensor(t(x))  # Force copy
        x_recon = t.inv(y)
        assert_array_almost_equal(x.detach(), x_recon.detach())
