import torch
import numpy as np
from torch.distributions import constraints, Distribution
from relie.utils.so3_tools import so3_uniform_random


class SO3Prior(Distribution):
    domain = constraints.real
    codomain = constraints.real

    event_dim = 2

    def __init__(self, device=None, dtype=None):
        super().__init__(event_shape=(3, 3))
        self.device = device
        self.dtype = dtype

    def sample(self, shape=torch.Size()):
        n = np.prod(shape)
        return so3_uniform_random(n, device=self.device, dtype=self.dtype)\
            .view(*shape, 3, 3)

    def log_prob(self, value):
        return value.new_full(value.shape[:-2], 1 / (8 * np.pi ** 2))

