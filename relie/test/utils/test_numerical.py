from numpy.testing import assert_array_almost_equal
import torch
from relie.utils.numerical import batch_trace


def test_batch_trace():
    ms = torch.randn(10000, 5, 5)
    x = batch_trace(ms)
    y = [torch.trace(m) for m in ms]
    assert_array_almost_equal(x, y)
