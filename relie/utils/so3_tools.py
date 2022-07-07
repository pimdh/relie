import math
import torch
from relie.utils.numerical import batch_trace, zero_one_outer_product


def so3_hat(v):
    """
    Map a point in R^N to the tangent space at the identity, i.e.
    to the Lie Algebra. Inverse of so3_vee.

    :param v: Lie algebra in vector rep of shape (..., 3
    :return: Lie algebar in matrix rep of shape (..., 3, 3)
    """
    assert v.shape[-1] == 3

    e_x = v.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    e_y = v.new_tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    e_z = v.new_tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    x = (
        e_x * v[..., 0, None, None]
        + e_y * v[..., 1, None, None]
        + e_z * v[..., 2, None, None]
    )
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
    assert v.dtype == torch.double
    theta = v.norm(p=2, dim=-1)

    mask = theta > 1e-10
    theta = torch.where(mask, theta, torch.ones_like(theta))

    # sin(x)/x -> 1-x^2/6 as x->0
    alpha = torch.where(mask, torch.sin(theta) / theta, 1 - theta ** 2 / 6)
    # (1-cos(x))/x^2 -> 0.5-x^2/24 as x->0
    beta = torch.where(mask, (1 - torch.cos(theta)) / theta ** 2, 0.5 - theta ** 2 / 24)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    x = so3_hat(v)
    return eye + alpha[..., None, None] * x + beta[..., None, None] * x @ x


def so3_log(r):
    """
    Logarithm map of SO(3).
    :param r: group element of shape (..., 3, 3)
    :return: Algebra element in matrix basis of shape (..., 3, 3)

    Uses https://en.wikipedia.org/wiki/Rotation_group_SO(3)#Logarithm_map
    """
    assert r.dtype == torch.double
    anti_sym = 0.5 * (r - r.transpose(-1, -2))
    cos_theta = 0.5 * (batch_trace(r)[..., None, None] - 1)
    cos_theta = cos_theta.clamp(-1, 1)  # Ensure we get a correct angle
    theta = torch.acos(cos_theta)
    ratio = theta / torch.sin(theta)

    # x/sin(x) -> 1 + x^2/6 as x->0
    mask = (theta[..., 0, 0] < 1e-20).nonzero()
    ratio[mask] = 1 + theta[mask] ** 2 / 6

    log = ratio * anti_sym

    # Separately handle theta close to pi
    mask = ((math.pi - theta[..., 0, 0]).abs() < 1e-2).nonzero()
    if mask.numel():
        log[mask[:, 0]] = so3_log_pi(r[mask[:, 0]], theta[mask[:, 0]])

    return log


def so3_log_pi(r, theta):
    """
    Logarithm map of SO(3) for cases with theta close to pi.
    Note: inaccurate for theta around 0.
    :param r: group element of shape (..., 3, 3)
    :param theta: rotation angle
    :return: Algebra element in matrix basis of shape (..., 3, 3)
    """
    sym = 0.5 * (r + r.transpose(-1, -2))
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(sym)
    z = theta ** 2 / (1 - torch.cos(theta)) * (sym - eye)

    q_1 = z[..., 0, 0]
    q_2 = z[..., 1, 1]
    q_3 = z[..., 2, 2]
    x_1 = torch.sqrt((q_1 - q_2 - q_3) / 2)
    x_2 = torch.sqrt((-q_1 + q_2 - q_3) / 2)
    x_3 = torch.sqrt((-q_1 - q_2 + q_3) / 2)
    x = torch.stack([x_1, x_2, x_3], -1)

    # Flatten batch dim
    batch_shape = x.shape[:-1]
    x = x.view(-1, 3)
    r = r.view(-1, 3, 3)

    # We know components up to a sign, search for correct one
    signs = zero_one_outer_product(3, dtype=x.dtype, device=x.device) * 2 - 1
    x_stack = signs.view(8, 1, 3) * x[None]
    with torch.no_grad():
        r_stack = so3_exp(x_stack)
        diff = (r[None] - r_stack).pow(2).sum(-1).sum(-1)
        selector = torch.argmin(diff, dim=0)
    x = x_stack[selector, torch.arange(len(selector))]

    # Restore shape
    x = x.view(*batch_shape, 3)

    return so3_hat(x)


def so3_xset(x, k_max):
    """
    Return set of x's that have same image as exp(x) excluding x itself.
    :param x: Tensor of shape (..., 3) of algebra elements.
    :param k_max: int. Number of 2pi shifts in either direction
    :return: Tensor of shape (2 * k_max+1, ..., 3)
    """
    x = x[None]
    x_norm = x.norm(dim=-1, keepdim=True)
    non_identity = torch.all(x != 0, dim=-1)
    shape = [-1, *[1] * (x.dim() - 1)]
    k_range = torch.arange(1, k_max + 1, dtype=x.dtype, device=x.device)
    k_range = torch.cat([-k_range, k_range]).view(shape)
    xset_norm = torch.zeros_like(x)
    xset_norm[non_identity] = x[non_identity] / x_norm[non_identity]
    return xset_norm * (x_norm + 2 * math.pi * k_range)


def so3_log_abs_det_jacobian(x):
    """
    Return element wise log abs det jacobian of exponential map
    :param x: Algebra tensor of shape (..., 3)
    :return: Tensor of shape (..., 3)

    Removable pole: (2-2 cos x)/x^2 -> 1-x^2/12 as x->0
    """
    x_norm = x.double().norm(dim=-1)
    mask = x_norm > 1e-10
    x_norm = torch.where(mask, x_norm, torch.ones_like(x_norm))

    ratio = torch.where(
        mask, (2 - 2 * torch.cos(x_norm)) / x_norm ** 2, 1 - x_norm ** 2 / 12
    )
    return torch.log(ratio).to(x.dtype)


def so3_matrix_to_quaternions(r):
    """
    Map batch of SO(3) matrices to quaternions.
    :param r: Batch of SO(3) matrices of shape (..., 3, 3)
    :return: Quaternions of shape (..., 4)
    """
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], "Input must be 3x3 matrices"
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack(
        [
            1 + diags[0] - diags[1] - diags[2],
            1 - diags[0] + diags[1] - diags[2],
            1 - diags[0] - diags[1] + diags[2],
            1 + diags[0] + diags[1] + diags[2],
        ],
        1,
    )
    denom = 0.5 * torch.sqrt(1e-6 + torch.abs(denom_pre))

    case0 = torch.stack(
        [
            denom[:, 0],
            (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
            (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
            (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0]),
        ],
        1,
    )
    case1 = torch.stack(
        [
            (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
            denom[:, 1],
            (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
            (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1]),
        ],
        1,
    )
    case2 = torch.stack(
        [
            (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
            (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
            denom[:, 2],
            (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2]),
        ],
        1,
    )
    case3 = torch.stack(
        [
            (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
            (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
            (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
            denom[:, 3],
        ],
        1,
    )

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[
        torch.arange(n, dtype=torch.long), torch.argmax(denom.detach(), 1)
    ]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_eazyz(q):
    """Map batch of quaternion to Euler angles ZYZ. Output is not mod 2pi."""
    eps = 1e-6
    return torch.stack(
        [
            torch.atan2(
                q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3],
                q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3],
            ),
            torch.acos(
                torch.clamp(
                    q[:, 3] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2,
                    -1.0 + eps,
                    1.0 - eps,
                )
            ),
            torch.atan2(
                q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2],
                q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2],
            ),
        ],
        1,
    )


def so3_matrix_to_eazyz(r):
    """Map batch of SO(3) matrices to Euler angles ZYZ."""
    return quaternions_to_eazyz(so3_matrix_to_quaternions(r))


def quaternions_to_so3_matrix(q):
    """Normalises q and maps to group matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack(
        [
            r * r - i * i - j * j + k * k,
            2 * (r * i + j * k),
            2 * (r * j - i * k),
            2 * (r * i - j * k),
            -r * r + i * i - j * j + k * k,
            2 * (i * j + r * k),
            2 * (r * j + i * k),
            2 * (i * j - r * k),
            -r * r - i * i + j * j + k * k,
        ],
        -1,
    ).view(*q.shape[:-1], 3, 3)


def random_quaternions(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack(
        (
            torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2),
            torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
            torch.sqrt(u1) * torch.cos(2 * math.pi * u3),
        ),
        1,
    )


def so3_uniform_random(n, dtype=torch.float32, device=None):
    return quaternions_to_so3_matrix(random_quaternions(n, dtype, device))


def so3_inv(el):
    return el.transpose(-2, -1)


def s2s2_gram_schmidt(v1, v2):
    """Normalise 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix."""
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-5)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)
