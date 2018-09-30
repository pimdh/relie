import torch
from relie.utils.numerical import batch_trace


def so3_hat(v):
    """
    Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra. Inverse of so3_vee.

    :param v: Lie algebra in vector rep of shape (..., 3
    :return: Lie algebar in matrix rep of shape (..., 3, 3)
    """
    assert v.shape[-1] == 3

    e_x = v.new_tensor([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])

    e_y = v.new_tensor([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])

    e_z = v.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])

    x = e_x * v[..., 0, None, None] + \
        e_y * v[..., 1, None, None] + \
        e_z * v[..., 2, None, None]
    return x


def so3_vee(x):
    """
    Map Lie algebra in ordinary (3, 3) matrix rep to vector.
    Inverse of so3_hat
    :param x: Lie algebar in matrix rep of shape (..., 3, 3)
    :return:  Lie algebra in vector rep of shape (..., 3
    """
    assert x.shape[-2:] == (3, 3)
    return torch.stack((-x[..., 1, 2], x[..., 0, 2], -x[..., 0, 1]), -1)


def so3_exp(v):
    """
    Exponential map of SO(3) with Rordigues formula.
    :param v: algebra vector of shape (..., 3)
    :return: group element of shape (..., 3, 3)
    """
    theta = v.norm(p=2, dim=-1, keepdim=True)
    k = so3_hat(v / theta)

    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    r = eye + torch.sin(theta)[..., None]*k \
        + (1. - torch.cos(theta))[..., None]*(k@k)
    return r


def so3_log(r):
    """
    Logarithm map of SO(3).
    :param r: group element of shape (..., 3, 3)
    :return: Algebra element in matrix basis of shape (..., 3, 3)

    Uses https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Logarithm_map
    """
    anti_sym = .5 * (r - r.transpose(-1, -2))
    theta = torch.acos(.5 * (batch_trace(r)[..., None, None] - 1))
    return theta / torch.sin(theta) * anti_sym
