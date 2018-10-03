import math
import torch
from relie.utils.numerical import batch_trace
from relie.utils.so3_tools import so3_hat, so3_vee, so3_exp, so3_log

def se3_fill(so3, r3, filler = "group"):
    """ 
    Conctenates a 3x3 matrix with an R^3 vector, adding on the bottom a filler
    :param so3: a tensor of shape (..., 3, 3) either an so3 repr or a lie algebra of so3 repr 
    :param r3: a tensor of shape (..., 3) 
    """
    
    assert so3.shape[-2:] == (3,3)
    assert r3.shape[-1] == 3
    
    fill_shape = list(r3.shape[:-1])
    
    if filler == "group":
        filler = torch.tensor([[0.,0.,0.,1.]]).view([1]*\
                            (len(fill_shape)+1)+[4]).repeat(fill_shape + [1,1])
    elif filler == "alg" or filler == "algebra":
        filler = torch.tensor([[0.,0.,0.,0.]]).view([1]*\
                            (len(fill_shape)+1)+[4]).repeat(fill_shape + [1,1])
    filler = filler.type(so3.dtype)
    se3 = torch.cat([so3, r3.unsqueeze(-1)], -1)
    se3 = torch.cat([se3, filler], -2)
    
    return se3
    

def se3_hat(v):
    """
    Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra. Inverse of so3_vee.

    :param v: Lie algebra in vector rep of shape (..., 6)
    :return: Lie algebar in matrix rep of shape (..., 4, 4)
    """
    assert v.shape[-1] == 6

    v_so3, v_r = v.split([3,3], dim=-1)
    v_r = v_r.unsqueeze(-1)
    alg_so3 = so3_hat(v_so3) #(..., 3, 3)
    
    return se3_fill(alg_so3, v_r, "alg")


def se3_vee(x):
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
    
    v_so3, v_r = v.split([3,3], dim=-1)
    
    so3 = so3_exp(v_so3)
    
    theta = v_so3.norm(p=2, dim=-1, keepdim=True)
    
    k = so3_hat(v_so3)

    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    V = eye + ((1 - torch.cos(theta))/theta**2)[..., None]*k \
        + ((theta - torch.sin(theta))/theta**3)[..., None]*(k@k)
    r3 = (V@v_r.unsqueeze(-1))
    
    
    return se3_fill(so3, r3.squeeze(-1), "group")

def se3_log(r):
    """   
    Logarithm map of SO(3).
    :param r: group element of shape (..., 4, 4)
    :return: Algebra element in matrix basis of shape (..., 4, 4)
   
    """
    
    
    se3, filler = r.split([3,1], -2)
    so3, r3 = se3.split([3,1],-1)
    
    so3_alg = so3_log(so3)
    #print(so3, so3.shape)
    theta = so3_vee(so3_alg).norm(p=2, dim=-1, keepdim=True)
    print(theta)
    
    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    
    Vinv = eye + so3_alg/2 \
        + ((1 - theta*torch.sin(theta)/(2-2*torch.cos(theta)))/theta**2)[..., None]*(so3_alg@so3_alg)
    #print(theta.shape)   
    #print(((1 - theta*torch.sin(theta)/(2-2*torch.cos(theta)))/theta**2)[..., None])
    
    r3_alg = Vinv@r3
    
    return se3_fill(so3_alg, r3_alg.squeeze(-1), "alg")

def se3_inv(x):
    """   
    Computes inverse of se3 element in matrix representation
    
    :x: group element of shape (..., 4, 4)
    :return: inverse of x of shape (..., 4, 4)
    """
    se3, filler = x.split([3,1], -2)
    so3, r3 = se3.split([3,1],-1)
    
    so3_inv = so3.transpose(-2,-1)
    
    r3_inv = so3_inv@r3
    
    return se3_fill(so3_inv, r3_inv.squeeze(-1))
    

