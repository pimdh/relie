import math
from itertools import cycle

import torch
from torch import nn as nn
from torch.distributions import ComposeTransform, Normal

from relie.flow import BatchNormTransform, CouplingTransform, PermuteTransform, RadialTanhTransform, \
    LocalDiffeoTransformedDistribution as LDTD
from relie.lie_distr import SO3ExpCompactTransform
from relie.utils.modules import MLP, BatchSqueezeModule, ToTransform


class Flow(nn.Module):
    def __init__(self, d, n_layers, batch_norm=True, net_layers=3):
        super().__init__()
        self.d = d
        self.d_residue = 1
        self.d_transform = d - self.d_residue

        self.nets = nn.ModuleList([
            MLP(self.d_residue,
                2 * self.d_transform,
                50,
                net_layers,
                batch_norm=False) for _ in range(n_layers)
        ])
        self._set_params()
        r = list(range(3))
        self.permutations = [r[i:] + r[:i] for i in range(3)]

        if batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(d) for _ in range(n_layers)])
        else:
            self.batch_norms = [None] * n_layers

    def forward(self):
        transforms = []
        for i, (net, bn, permutation) in enumerate(
                zip(self.nets, self.batch_norms, cycle(self.permutations))):
            if bn is not None:
                transforms.append(BatchNormTransform(bn))
            transforms.extend([
                CouplingTransform(self.d_residue, BatchSqueezeModule(net)),
                PermuteTransform(permutation),
            ])
        return ComposeTransform(transforms)

    def _set_params(self):
        """
        Initialize coupling layers to be identity.
        """
        for net in self.nets:
            last_module = list(net.modules())[-1]
            last_module.weight.data = torch.zeros_like(last_module.weight)
            last_module.bias.data = torch.zeros_like(last_module.bias)


class FlowDistribution(nn.Module):
    def __init__(self, flow, algebra_support_radius=math.pi * 1.6):
        super().__init__()
        self.flow = flow
        self.register_buffer('prior_loc', torch.zeros(3))
        self.register_buffer('prior_scale', torch.ones(3))
        self.intermediate_transform = ComposeTransform([
            RadialTanhTransform(algebra_support_radius),
            ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64))
        ])
        self.algebra_support_radius = algebra_support_radius

    def transforms(self):
        transforms = [
            self.flow(),
            self.intermediate_transform,
            SO3ExpCompactTransform(self.algebra_support_radius),
        ]
        return transforms

    def forward(self):
        prior = Normal(self.prior_loc, self.prior_scale)
        return LDTD(prior, self.transforms())