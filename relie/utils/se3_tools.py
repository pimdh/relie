import math
import torch
from relie.utils.numerical import batch_trace
from so3_tools import so3_hat, so3_vee, so3_exp, so3_log


def se3_hat(v):
    """
    Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra. Inverse of so3_vee.

    :param v: Lie algebra in vector rep of shape (..., 6)
    :return: Lie algebar in matrix rep of shape (..., 4, 4)
    """
    assert v.shape[-1] == 6
    
    fill_shape = list(v.shape[:-1])
    
    v_so3, v_r = v.split([3,3], dim=-1)
    v_r = v_r.unsqueeze(-1)
    alg_so3 = so3_hat(v_so3) #(..., 3, 3)
    
    
    filler = torch.tensor([[0.,0.,0.,0.]]).view([1]*\
                            (len(fill_shape)+1)+[4]).repeat(fill_shape + [1,1])
    alg_se3 = torch.cat([alg_so3, v_r], -1).type(torch.float32)
    alg_se3 = torch.cat([alg_se3, filler], -2)

    return alg_se3


def so3_vee(x):
    """
    Map Lie algebra in ordinary (3, 3) matrix rep to vector.
    Inverse of so3_hat
    :param x: Lie algebar in matrix rep of shape (..., 4, 4)
    :return:  Lie algebra in vector rep of shape (..., 6
    """
    assert x.shape[-2:] == (4, 4)
    se3_alg, filler = x.split([3,1], -2)
    so3_alg, r3_alg = se3_alg.split([3,1],-1)
    v_so3 = so3_vee(so3_alg)
    
    return torch.cat([v_so3, r3_alg.squeeze(-1)], -1)

def se3_exp(v):
    """
    Exponential map of SE(3) with Rordigues formula.
    :param v: algebra vector of shape (..., 6)
    :return: group element of shape (..., 4, 4)
    """
    fill_shape = list(v.shape[:-1])
    
    v_so3, v_r = v.split([3,3], dim=-1)
    
    so3 = so3_exp(v_so3)
    
    theta = v_so3.norm(p=2, dim=-1, keepdim=True)
    
    k = so3_hat(v_so3)

    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    V = eye + ((1 - torch.cos(theta))/theta**2)[..., None]*k \
        + ((theta - torch.sin(theta))/theta**3)[..., None]*(k@k)
    r3 = (V@v_r.unsqueeze(-1))
    
    filler = torch.tensor([[0.,0.,0.,1.]]).view([1]*\
                            (len(fill_shape)+1)+[4]).repeat(fill_shape + [1,1])
    se3 = torch.cat([so3, r3], -1).type(torch.float32)
    se3 = torch.cat([se3, filler], -2)
    
    return se3

"""
def so3_log(r):
   
    Logarithm map of SO(3).
    :param r: group element of shape (..., 3, 3)
    :return: Algebra element in matrix basis of shape (..., 3, 3)

    Uses https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Logarithm_map

    anti_sym = .5 * (r - r.transpose(-1, -2))
    theta = torch.acos(.5 * (batch_trace(r)[..., None, None] - 1))
    return theta / torch.sin(theta) * anti_sym
"""

