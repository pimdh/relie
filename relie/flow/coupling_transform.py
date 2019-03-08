import math
import torch
import torch.nn.functional as F
from torch.distributions import Transform, constraints


class CouplingTransform(Transform):
    r"""
    NICE coupling.

    dim = d_residue + d_transform
    f : batch x d_residue -> batch x (2*d_transform)
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, d_residue, f, cache_size=1):
        super().__init__(cache_size=cache_size)
        self.d_residue = d_residue
        self._f = f
        self._cached_s = None

    def _call(self, x):
        x_transform, x_residue = self.partition(x)
        pre_s, bias = self.f_split(x_residue)
        s = F.softplus(pre_s + math.log(math.e - 1))
        self._cached_s = s
        y_transform = s * x_transform + bias
        y = torch.cat([y_transform, x_residue], -1)
        return y

    def _inverse(self, y):
        y_transform, y_residue = self.partition(y)
        pre_s, bias = self.f_split(y_residue)
        s = F.softplus(pre_s + math.log(math.e - 1))
        x_transform = (y_transform - bias) / s
        x = torch.cat([x_transform, y_residue], -1)
        return x

    def log_abs_det_jacobian(self, x, y):
        s = self._get_cached_s(x)
        if s is None:
            x_residue = self.partition(x)[1]
            pre_s, _ = self.f_split(x_residue)
            s = F.softplus(pre_s + math.log(math.e - 1))
        return s.abs().log().sum(-1)

    def _get_cached_s(self, x):
        x_old, _ = self._cached_x_y
        if self._cached_s is not None and x is x_old:
            return self._cached_s
        return None

    def partition(self, x):
        d = x.shape[-1]
        return x[..., : d - self.d_residue], x[..., d - self.d_residue :]

    def f_split(self, x):
        out = self._f(x)
        d = out.shape[-1] // 2
        return out[..., :d], out[..., d:]
