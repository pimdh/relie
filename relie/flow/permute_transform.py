import torch
from torch.distributions import Transform, constraints


class PermuteTransform(Transform):
    r"""
    NICE coupling.

    dim = d_residue + d_transform
    f : batch x d_residue -> (batch x d_transform, batch x d_transform)
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, permutation):
        super().__init__(cache_size=1)
        self.permutation = torch.tensor(permutation, dtype=torch.long)
        self.inv_permutation = torch.sort(self.permutation)[1]

    def _call(self, x):
        return x[..., self.permutation]

    def _inv_call(self, y):
        return y[..., self.inv_permutation]

    def log_abs_det_jacobian(self, x, y):
        return x.new_zeros(x.shape[:-1])
