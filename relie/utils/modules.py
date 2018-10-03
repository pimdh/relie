import torch
import torch.nn as nn
from torch.distributions import Transform


class MLP(nn.Module):
    """Helper module to create MLPs."""
    def __init__(self, input_dims, output_dims, hidden_dims,
                 num_layers=1, activation=nn.ReLU, batch_norm=False):
        super().__init__()
        if num_layers == 0:
            self.module = nn.Linear(input_dims, output_dims)
        else:
            dims = [input_dims, *[hidden_dims]*num_layers, output_dims]
            modules = []
            for l, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
                modules.append(nn.Linear(d_in, d_out))

                if l < num_layers:
                    if batch_norm:
                        modules.append(nn.BatchNorm1d(d_out))
                    modules.append(activation())
            self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class ConditionalModule(nn.Module):
    """
    Module that wraps submodule for conditional flow.
    Broadcasts conditional variable to allow for extra batch dims.
    Concats conditional var and input.
    Reshapes to feed batch dim = 1 to submodule.

    Note:
        x and y must have same -2'th dimension (usually batch)
    """
    def __init__(self, submodule, x):
        super().__init__()
        self.x = x
        self.submodule = submodule

    def forward(self, y):
        x = self.x.expand(*y.shape[:y.dim() - 2], -1, -1)
        z = torch.cat([x, y], -1)
        batch_shape = z.shape[:-1]
        z = z.view(-1, z.shape[-1])
        res = self.submodule(z).view(*batch_shape, -1)
        return res


class BatchSqueezeModule(nn.Module):
    """
    Module that reshapes batch dim to single dimension, calls submodule and reshapes.
    """
    def __init__(self, submodule, feature_dims=1):
        super().__init__()
        self.submodule = submodule
        self.feature_dims = feature_dims

    def forward(self, x):
        batch_shape = x.shape[:-self.feature_dims]
        x = x.view(-1, *x.shape[-self.feature_dims:])
        y = self.submodule(x)
        return y.view(*batch_shape, *y.shape[1:])


class ToTransform(Transform):
    """
    Transform dtype or device.
    """
    event_dim = 0
    sign = 1

    def __init__(self, options_in, options_out):
        super().__init__(1)
        self.options_in = options_in
        self.options_out = options_out

    def _call(self, x):
        return x.to(**self.options_out)

    def _inverse(self, y):
        return y.to(**self.options_in)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x).float()