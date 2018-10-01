from itertools import product
import torch


def batch_trace(m):
    d = m.shape[-1]
    assert m.shape[-2] == d
    i = torch.arange(d, dtype=torch.int64)
    return m[..., i, i].sum(-1)


def sample_ball(n, d, device=None, dtype=None):
    u = torch.rand(n, device=device, dtype=dtype) ** (1/d)
    x = torch.randn(n, d, device=device, dtype=dtype)
    return x / x.norm(dim=1, keepdim=True) * u[:, None]


def zero_one_outer_product(n, dtype=None, device=None):
    return torch.tensor(
        list(product([0, 1], repeat=n)), dtype=dtype, device=device)
