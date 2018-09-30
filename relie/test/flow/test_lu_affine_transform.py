from numpy.testing import assert_array_equal, assert_array_almost_equal
import torch
from relie.flow.lu_affine_transform import LUAffineTransform


def test_lu_affine_transform():
    d = 5
    batch_size = 64
    for _ in range(100):
        lower = (torch.randn(d, d).tril(-1) + torch.eye(d)).double()
        upper = torch.randn(d, d).triu(1).double()
        diag = torch.randn(d).double()
        bias = torch.randn(d).double()
        w = lower @ (upper + torch.diagflat(diag))

        t = LUAffineTransform(lower, upper, diag, bias)

        x = torch.randn(batch_size, d).double()
        y = x @ w.t() + bias

        assert_array_equal(t.w, w)
        assert_array_equal(t.w_inv, w.inverse())

        assert_array_equal(t(x), y)
        assert_array_almost_equal(t.inv(torch.tensor(y)), x, 5)

        assert_array_almost_equal(
            t.log_abs_det_jacobian(x, y)[0],
            torch.log(torch.abs(torch.det(w))),
            decimal=5)
