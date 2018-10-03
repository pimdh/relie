import torch
import torch.nn as nn
from numpy.testing import assert_array_almost_equal

from relie.flow.batch_norm_transform import BatchNormTransform


def test_batch_norm_transform_eval():
    d = 7
    batch_size = 10000

    for _ in range(100):
        module = nn.BatchNorm1d(d)
        t = BatchNormTransform(module)
        t(torch.randn(10000, d))

        module.eval()
        x = torch.randn(batch_size, d)
        y = t(x).clone()
        x_recon = t.inv(y)
        assert_array_almost_equal(x.detach(), x_recon.detach())



def test_batch_norm_transform_train():
    d = 7
    batch_size = 10000

    for _ in range(100):
        module = nn.BatchNorm1d(d)
        t = BatchNormTransform(module)

        x = torch.randn(batch_size, d)
        module.eval()
        y = t(x).clone()
        x_recon = t.inv(y)
        assert_array_almost_equal(x.detach(), x_recon.detach())
