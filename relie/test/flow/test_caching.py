import torch
from torch.distributions import Normal

from relie.flow import LocalDiffeoTransformedDistribution as LDTD, CouplingTransform
from relie.lie_distr import SO3ExpTransform
from relie.utils.modules import MLP, BatchSqueezeModule, ToTransform


def mock_inverse(y):
    """Test original is not inverted."""
    assert y.shape == (4, 64, 3)
    return y


def test_caching():

    net = MLP(1, 2 * 2, 50, 1)
    prior = Normal(torch.zeros(3), torch.ones(3))
    t = CouplingTransform(1, BatchSqueezeModule(net))
    transforms = [
        t,
        ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64)),
        SO3ExpTransform(2)
    ]

    distr = LDTD(prior, transforms)
    t._inverse = mock_inverse
    g = distr.rsample((64,))
    distr.log_prob(g)

