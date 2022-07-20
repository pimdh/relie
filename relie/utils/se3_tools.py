import torch
from pytorch3d.common.compat import solve
from pytorch3d.transforms.se3 import se3_exp_map, se3_log_map
from pytorch3d.transforms.so3 import hat, so3_log_map
from relie.utils.so3_tools import so3_log_abs_det_jacobian, so3_xset


def get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    nrms = (log_rotation ** 2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles

def se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles ** 2))[:, None, None]
        + (
            log_rotation_hat_square
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles ** 3))[
                :, None, None
            ]
        )
    )

    return V

def se3_fill(r3, so3, filler="group"):
    """ 
    Conctenates a 3x3 matrix with an R^3 vector, adding on the bottom a filler
    :param r3: a tensor of shape (..., 3) 
    :param so3: a tensor of shape (..., 3, 3) either an so3 repr or a lie algebra of so3 repr 
    """
    raise NotImplementedError


def se3_hat(v):
    """
    Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra. Inverse of so3_vee.
    :param v: Lie algebra in vector rep of shape (..., 6)
    :return: Lie algebar in matrix rep of shape (..., 4, 4)
    """
    raise NotImplementedError


def se3_vee(x):
    """
    Map Lie algebra in ordinary (3, 3) matrix rep to vector.
    Inverse of so3_hat
    :param x: Lie algebar in matrix rep of shape (..., 4, 4)
    :return:  Lie algebra in vector rep of shape (..., 6
    """

    raise NotImplementedError


def se3_exp(v):
    """
    Exponential map of SE(3) with Rordigues formula.
    :param v: algebra vector of shape (..., 6)
    :return: group element [R, T] of shape (..., 4, 4)
    """
    
    return se3_exp_map(v)


def se3_log(r):
    """   
    Logarithm map of SE(3).
    :param r: group element of shape (..., 4, 4)
    :return: Algebra element [log_translation, log_rotation] in matrix basis of shape (..., 4, 4)
   
    """
    return se3_log_map(r)


def se3_inv(x):
    """   
    Computes inverse of se3 element in matrix representation
    
    :x: group element of shape (..., 4, 4)
    :return: inverse of x of shape (..., 4, 4)
    """
    raise NotImplementedError


def se3_xset(x, k_max):
    """
    Returns set of x's that have same image as exp(x) excluding x itself.
    :param x: Tensor of shape (..., 6) of algebra elements.
    :param k_max: int. Number of 2pi shifts in either direction
    :return: Tensor of shape (2 * k_max+1, ..., 6)
    """

    transform = se3_exp(x)
    T = transform[:, 3, :3]
    R = transform[:, :3, :3].permute(0, 2, 1)
    log_rotation = so3_log_map(R)
    log_rotation = so3_xset(log_rotation, k_max).squeeze(dim=1)
    
    N = log_rotation.shape[1]
    T = torch.tile(T.unsqueeze(0), (2 * k_max, 1, 1)).reshape(2 * k_max * N, 3)
    log_rotation = log_rotation.reshape(2 * k_max * N, 3)

    V = se3_V_matrix(*get_se3_V_input(log_rotation))

    log_translation = solve(V, T[:, :, None])[:, :, 0]

    log_translation = log_translation.reshape(2*k_max, N, 3)
    log_rotation = log_rotation.reshape(2*k_max, N, 3)

    return torch.cat([log_translation, log_rotation], dim=-1)


def se3_log_abs_det_jacobian(x):    
    """
    Returns element wise log abs det Jacobian of exponential map for SE(3)

    :param v: Algebra vector [log_translation, log_rotation] of shape (..., 6)
    :return: Tensor of shape (..., 3)

    Removable pole: (2-2 cos x)/x^2 -> 1-x^2/12 as x->0
    """

    log_translation, log_rotation = x.split([3, 3], dim=-1)
    return 2*so3_log_abs_det_jacobian(log_rotation)
