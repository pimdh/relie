import numpy as np
from torch.distributions import constraints, Distribution
from relie.utils.so3_tools import quaternions_to_so3_matrix, random_quaternions


class SO3Prior(Distribution):
    domain = constraints.real
    codomain = constraints.real

    event_dim = 2

    def __init__(self, device=None, dtype=None):
        super().__init__(event_shape=(3, 3))
        self.device = device
        self.dtype = dtype

    def sample(self, shape):
        n = np.prod(shape)
        q = random_quaternions(n, device=self.device, dtype=self.dtype)
        return quaternions_to_so3_matrix(q.view(*shape, 4))

    def log_prob(self, value):
        return value.new_full(value[:-2], 1 / (8 * np.pi ** 2))

