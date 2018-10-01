from torch.distributions import constraints

from relie.flow import LocalDiffeoTransform
from relie.utils.so3_tools import so3_exp, so3_log, so3_vee, so3_xset,\
    so3_log_abs_det_jacobian


class SO3ExpTransform(LocalDiffeoTransform):
    domain = constraints.real
    codomain = constraints.real

    event_dim = 1

    def __init__(self, k_max=5):
        """
        :param k_max: Returns inverse set with k \in [-k_max, k_max]
        """
        super().__init__()
        self.k_max = k_max

    def _call(self, x):
        return so3_exp(x)

    def _inverse_set(self, y):
        return self._xset(so3_vee(so3_log(y)))

    def _xset(self, x):
        return so3_xset(x, self.k_max)

    def log_abs_det_jacobian(self, x, y):
        """
        Log abs det of forward jacobian of exp map.
        :param x: Algebra element shape (..., 3)
        :param y: Group element (..., 3, 3)
        :return: Jacobian of exp shape (...)
        """
        return so3_log_abs_det_jacobian(x)
