import torch
import torch.nn as nn
from torch import nn as nn
from torch.distributions import Transform


class MLP(nn.Sequential):
    """Helper module to create MLPs."""
    def __init__(self, input_dims, output_dims, hidden_dims,
                 num_layers=1, activation=nn.ReLU):
        if num_layers == 0:
            super().__init__(nn.Linear(input_dims, output_dims))
        else:
            super().__init__(
                nn.Linear(input_dims, hidden_dims),
                activation(),
                *[l for _ in range(num_layers-1)
                  for l in [nn.Linear(hidden_dims, hidden_dims), activation()]],
                nn.Linear(hidden_dims, output_dims)
            )


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

    def _inv_call(self, y):
        return y.to(**self.options_in)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x).float()