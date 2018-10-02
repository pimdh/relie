import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from relie.utils.numerical import sample_ball
from relie.flow.radial_tanh_transform import RadialTanhTransform


def test_transform():
    batch_size = 64

    for _ in range(100):
        d = np.random.choice(10)+1
        radius = np.random.rand() * 5
        t = RadialTanhTransform(radius)

        x = torch.randn(batch_size, d, dtype=torch.double)
        x[0] = x[0] * 1E-10  # Test near 0

        y = t(x.clone())
        x_recon = t.inv(y)
        assert_array_almost_equal(x.detach(), x_recon.detach())

        y_norm = y.norm(dim=1)
        assert (y_norm < radius+1E-5).all()

        y = sample_ball(batch_size, d, dtype=torch.double) * radius
        y[0] = y[0] * 1E-10  # Test near 0
        x = t.inv(y)
        y_recon = t(x.clone())
        assert_array_almost_equal(y_recon.detach(), y.detach())
