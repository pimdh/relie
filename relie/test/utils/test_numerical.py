from numpy.testing import assert_array_almost_equal
import torch
from relie.utils.numerical import batch_trace, atanh


def test_batch_trace():
    ms = torch.randn(10000, 5, 5)
    x = batch_trace(ms)
    y = [torch.trace(m) for m in ms]
    assert_array_almost_equal(x, y)


def test_atanh():
    x = torch.randn(100000, dtype=torch.double)
    y = torch.tanh(x)
    x_recon = atanh(y)
    assert_array_almost_equal(x_recon, x)

    y = torch.rand(1000, dtype=torch.double) * (2 - 1E-5) - 1
    x = atanh(y)
    y_recon = torch.tanh(x)
    assert_array_almost_equal(y, y_recon)


def test_atanh_grad():
    y = torch.rand(10000, requires_grad=True, dtype=torch.double) * (2-1E-3) - 1
    assert torch.autograd.gradcheck(atanh, (y,), eps=1e-6, atol=1e-4)
