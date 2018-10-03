import math

import torch
from torch import nn as nn
from torch.distributions import Transform, constraints
from torch.distributions.utils import lazy_property


class LUAffineTransform(Transform):
    r"""
    Affine transform of batch of vectors using matrices in LU parametrization.

    lower must be lower triangular with 0 diagonals (otherwise will be made so).
    upper must be upper triangular with 0 diagonals (otherwise will be made so).
    diag must be vector
    bias must be vector

    See GLOW 3.2
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, lower, upper, diag, bias, cache_size=1):
        super().__init__(cache_size=cache_size)
        d = lower.shape[0]
        assert lower.shape == (d, d)
        assert upper.shape == (d, d)
        assert diag.shape == (d,)
        assert bias.shape == (d, )
        self.lower = lower.tril(-1) + torch.eye(
            d, dtype=lower.dtype, device=lower.device)
        self.upper = upper.triu(1)
        self.diag = diag
        self.bias = bias

    @lazy_property
    def w(self):
        return self.lower @ (self.upper + torch.diagflat(self.diag))

    @lazy_property
    def w_inv(self):
        return self.w.inverse()

    @lazy_property
    def sign(self):
        return self.diag.prod().sign()

    def _call(self, x):
        return x @ self.w.t() + self.bias

    def _inverse(self, y):
        return (y - self.bias) @ self.w_inv.t()

    def log_abs_det_jacobian(self, x, y):
        return self.diag.abs().log().sum(-1)


def lu_affine_transform_parameters(d):
    # stdv = 1. / math.sqrt(d) / 100
    stdv = 0.
    tensors = {
        'lower': torch.Tensor(d, d).uniform_(-stdv, stdv),
        'upper': torch.Tensor(d, d).uniform_(-stdv, stdv),
        'diag': torch.Tensor(d).uniform_(-stdv, stdv) + 1,
        'bias': torch.Tensor(d).uniform_(-stdv, stdv),
    }
    return nn.ParameterDict({k: nn.Parameter(t) for k, t in tensors.items()})