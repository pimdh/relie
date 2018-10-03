import torch
from torch.distributions import Transform, constraints


class BatchNormTransform(Transform):
    r"""
    Batch Normalization.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, module):
        super().__init__(cache_size=1)
        self.module = module
        self._cache_stats = None, None, None

    def _call(self, x):
        org_x = x
        batch_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        mu = x.mean(dim=0)
        var = x.var(dim=0)
        self._cache_stats = org_x, mu, var
        return self.module(x).view(*batch_shape, x.shape[-1])

    def _inverse(self, y):
        """Take mean and var from running mean. Inverse of test mode BatchNorm."""
        batch_shape = y.shape[:-1]
        y = y.view(-1, y.shape[-1])
        mu = self.module.running_mean.detach()
        var = self.module.running_var.detach()
        sigma = torch.sqrt(var + self.module.eps)
        x = ((y - self.module.bias) / self.module.weight
             * sigma + mu)
        x = x.view(*batch_shape, y.shape[-1])
        self._cache_stats = x, mu, var
        return x

    def log_abs_det_jacobian(self, x, y):
        old_x, mu, var = self._cache_stats
        assert old_x is x
        sigma = torch.sqrt(var + self.module.eps)
        scale = self.module.weight / sigma
        j = scale.abs().log().sum()
        return j.expand(x.shape[:-1])
