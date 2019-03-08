from functools import partial
from numpy.testing import assert_array_equal, assert_array_almost_equal
import torch
import torch.nn as nn
from relie.flow.coupling_transform import CouplingTransform


def f_fixed(x, d_transform):
    z = x.new_zeros(x.shape[0], d_transform)
    log_s = torch.arange(d_transform, dtype=x.dtype, device=x.device).expand(
        x.shape[0], -1
    )
    return torch.cat([log_s, z], 1)


def test_coupling():
    d = 7
    d_residue = 3
    batch_size = 64

    for _ in range(100):
        f = nn.Sequential(
            nn.Linear(d_residue, 10), nn.ReLU(), nn.Linear(10, 2 * (d - d_residue))
        )
        t = CouplingTransform(d_residue, f, cache_size=1)

        x = torch.randn(batch_size, d)
        y = torch.tensor(t(x))  # Force copy

        x_recon = t.inv(y)
        assert_array_almost_equal(x.detach(), x_recon.detach())


def test_logdet():
    d = 7
    d_residue = 3
    batch_size = 64
    f = partial(f_fixed, d_transform=d - d_residue)
    x = torch.randn(batch_size, d)
    t = CouplingTransform(d_residue, f)

    logabsdet = t.log_abs_det_jacobian(x, t(x))
    assert_array_equal(logabsdet, torch.full((batch_size,), sum(range(d - d_residue))))
