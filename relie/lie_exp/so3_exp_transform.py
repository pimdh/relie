import math
import torch
from torch.distributions import constraints

from relie.flow import LocalDiffeoTransform
from relie.utils.so3_tools import so3_exp, so3_log, so3_vee


class SO3ExpTransform(LocalDiffeoTransform):
    domain = constraints.real
    codomain = constraints.real

    event_dim = 1

    def __init__(self, k_max=1):
        """
        :param k_max: Returns inverse set with k \in [-k_max, k_max]
        """
        super().__init__()
        self.k_max = k_max

    def _call(self, x):
        return so3_exp(x)

    def _inverse_set(self, y):
        return self._xset(so3_vee(so3_log(y)))

    def _xset(self, x):
        x = x[None]
        x_norm = x.norm(dim=-1, keepdim=True)
        shape = [-1, *[1]*(x.dim()-1)]
        k_range = torch.arange(-self.k_max, self.k_max+1, dtype=x.dtype).view(shape)
        return x / x_norm * (x_norm + 2 * math.pi * k_range)

    def log_abs_det_jacobian(self, x, y):
        """
        Log abs det of forward jacobian of exp map.
        :param x: Algebra element shape (..., 3)
        :param y: Group element (..., 3, 3)
        :return: Jacobian of exp shape (...)
        """
        x_norm = x.norm(dim=-1)
        return torch.log(2 - 2 * torch.cos(x_norm)) - torch.log(x_norm ** 2)
