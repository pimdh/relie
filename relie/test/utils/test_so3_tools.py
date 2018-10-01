import numpy as np
from numpy.testing import assert_array_almost_equal

from relie.utils.numerical import sample_ball
from relie.utils.so3_tools import *


def test_exp_not_nan():
    ball = sample_ball(1000000, 3) * 2 * np.pi
    r = so3_exp(ball)
    assert torch.isnan(r).sum() == 0
    r = so3_exp(torch.zeros(3))
    assert torch.isnan(r).sum() == 0


def test_log_not_nan():
    r = so3_exp(sample_ball(1000000, 3, dtype=torch.double) * 2 * np.pi)
    assert torch.isnan(so3_log(r)).sum() == 0
    r = quaternions_to_so3_matrix(random_quaternions(100000, dtype=torch.double))
    assert torch.isnan(so3_log(r)).sum() == 0


def test_diffeo_region():
    x = sample_ball(1000000, 3, dtype=torch.double) * (1 * np.pi - 1E-6)
    r = so3_exp(x)
    x_recon = so3_vee(so3_log(r))
    assert_array_almost_equal(x_recon, x)


def assert_reconstruction(x, x_recon):
    # Test vectors parallel
    x_norm = x.norm(dim=1, keepdim=True)
    x_normed = x / x_norm
    x_recon_norm = x_recon.norm(dim=1, keepdim=True)
    x_recon_normed = x_recon / x_recon_norm
    diff = torch.min(
        (x_normed-x_recon_normed).pow(2).sum(1),
        (x_normed+x_recon_normed).pow(2).sum(1))
    assert_array_almost_equal(diff, torch.zeros_like(diff))

    # Test xset
    xset = so3_xset(x, 3)
    diff = (xset - x_recon[None]).abs().sum(2).min(dim=0)[0]
    assert_array_almost_equal(diff, torch.zeros_like(diff), 5)


def test_regular_pi():
    """
    Test behaviour of Log near radius pi.
    :return:
    """
    x = torch.randn(100, 3).double()
    x = x / x.norm(dim=1, keepdim=True) * np.pi
    r = so3_exp(x)
    x_recon = so3_vee(so3_log(r))
    assert_reconstruction(x, x_recon)


def test_regular_region():
    x = sample_ball(1000000, 3, dtype=torch.double) * 2 * np.pi
    r = so3_exp(x)
    x_recon = so3_vee(so3_log(r))
    assert_reconstruction(x, x_recon)

    xset = so3_xset(x, 2)

    diff = (xset - x_recon[None]).abs().sum(2).min(dim=0)[0]
    assert_array_almost_equal(diff, torch.zeros_like(diff), 4)


def test_so3_log_pi():
    """Test so3_log_pi in particular."""
    x = torch.randn(1, 3)
    x = x / x.norm(dim=1, keepdim=True) * .2
    theta = torch.tensor([.2])[:, None, None]
    r = so3_exp(x)
    x_recon = so3_vee(so3_log_pi(r, theta))
    assert_array_almost_equal(x, x_recon, 4)
