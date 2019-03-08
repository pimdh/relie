import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import functional as F

from relie.flow import LocalDiffeoTransformedDistribution as LDTD
from relie.lie_distr import SO3ExpTransform, SO3MultiplyTransform
from relie.utils.so3_tools import so3_exp


class PushedGaussianDistribution(nn.Module):
    def __init__(self, lie_multiply=False):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(3).double())
        self.pre_scale = nn.Parameter(torch.ones(3).double())
        self.transform = SO3ExpTransform(k_max=3)
        self.lie_multiply = lie_multiply

    @property
    def scale(self):
        return F.softplus(self.pre_scale)

    def forward(self):
        if self.lie_multiply:
            alg_distr = Normal(torch.zeros(3).double(), self.scale)
            loc = so3_exp(self.loc)
            transforms = [self.transform, SO3MultiplyTransform(loc)]
        else:
            alg_distr = Normal(self.loc * 1, self.scale)
            transforms = [self.transform]
        return LDTD(alg_distr, transforms)
