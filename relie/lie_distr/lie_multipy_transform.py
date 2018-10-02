from functools import partial
from torch.distributions import Transform, constraints
from torch.distributions.utils import lazy_property
from relie.utils.so3_tools import so3_inv
from relie.utils.se3_tools import se3_inv


class LieMultiplyTransform(Transform):
    r"""
    Left multiply with (batch of) matrix Lie group elements.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 2

    def __init__(self, g, inverse_fn):
        super().__init__(cache_size=1)
        self.g = g
        self.inverse_fn = inverse_fn

    @lazy_property
    def _g_inv(self):
        return self.inverse_fn(self.g)

    def _call(self, x):
        return self.g.expand_as(x) @ x

    def _inverse(self, y):
        return self._g_inv.expand_as(y) @ y

    def log_abs_det_jacobian(self, x, y):
        """Haar measure left invariant."""
        return x.new_zeros(x.shape[:-2])


SO3MultiplyTransform = partial(LieMultiplyTransform, inverse_fn=so3_inv)
SE3MultiplyTransform = partial(LieMultiplyTransform, inverse_fn=se3_inv)

