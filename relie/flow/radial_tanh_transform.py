import math
import torch
from torch.distributions import Transform, constraints


class RadialTanhTransform(Transform):
    r"""
    Transform R^d of radius (0, inf) to (0, R)
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, radius):
        super().__init__(cache_size=1)
        self.radius = radius

    def _call(self, x):
        x_norm = x.norm(dim=-1, keepdim=True)
        x_normed = x / x_norm
        y = torch.tanh(x_norm) * x_normed * self.radius

        return torch.where(
            x_norm > 1E-8,
            y,
            x * self.radius
        )

    def _inverse(self, y):
        y_norm = y.norm(dim=-1, keepdim=True)
        y_normed = y / y_norm
        x = self.tanh_inverse(y_norm / self.radius) * y_normed

        return torch.where(
            y_norm > 1E-8,
            x,
            y / self.radius
        )

    @staticmethod
    def tanh_inverse(y):
        """
        Inverse tanh.

        From: http://mathworld.wolfram.com/InverseHyperbolicTangent.html
        :param y: tensor
        :return: tensor
        """
        return .5 * (torch.log1p(y) - torch.log1p(-y))

    def log_abs_det_jacobian(self, x, y):
        """
        Uses d tanh /dx = 1-tanh^2
        :param x: Tensor
        :param y: Tensor
        :return: Tensor
        """
        y_norm = y.norm(dim=-1)
        d = y.shape[-1]
        tanh = y_norm / self.radius
        log_dtanh = torch.log1p(-tanh**2)

        log_radius = torch.full_like(log_dtanh, math.log(self.radius))
        return d * torch.where(
            y_norm > 1E-8,
            log_dtanh + log_radius,
            log_radius,
        )
