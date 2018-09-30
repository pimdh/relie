import math
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


def so3_matrix_to_quaternions(r):
    """
    Map batch of SO(3) matrices to quaternions.
    :param r: Batch of SO(3) matrices of shape (..., 3, 3)
    :return: Quaternions of shape (..., 4)
    """
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2],
        1 + diags[0] + diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1E-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[torch.arange(n, dtype=torch.long),
                        torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)


def quaternions_to_eazyz(q):
    """Map batch of quaternion to Euler angles ZYZ. Output is not mod 2pi."""
    eps = 1E-6
    return torch.stack([
        torch.atan2(q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3],
                    q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
        torch.acos(torch.clamp(q[:, 3] ** 2 - q[:, 0] ** 2
                               - q[:, 1] ** 2 + q[:, 2] ** 2,
                               -1.0+eps, 1.0-eps)),
        torch.atan2(q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2],
                    q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    ], 1)


def so3_matrix_to_eazyz(r):
    """Map batch of SO(3) matrices to Euler angles ZYZ."""
    return quaternions_to_eazyz(so3_matrix_to_quaternions(r))


def quaternions_to_so3_matrix(q):
    """Normalises q and maps to group matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        r*r - i*i - j*j + k*k, 2*(r*i + j*k), 2*(r*j - i*k),
        2*(r*i - j*k), -r*r + i*i - j*j + k*k, 2*(i*j + r*k),
        2*(r*j + i*k), 2*(i*j - r*k), -r*r - i*i + j*j + k*k
    ], -1).view(*q.shape[:-1], 3, 3)


def _z_rot_mat(angle, l):
    m = angle.new_zeros((angle.size(0), 2 * l + 1, 2 * l + 1))

    inds = torch.arange(
        0, 2 * l + 1, 1, dtype=torch.long, device=angle.device)
    reversed_inds = torch.arange(
        2 * l, -1, -1, dtype=torch.long, device=angle.device)

    frequencies = torch.arange(
        l, -l - 1, -1, dtype=angle.dtype, device=angle.device)[None]

    m[:, inds, reversed_inds] = torch.sin(frequencies * angle[:, None])
    m[:, inds, inds] = torch.cos(frequencies * angle[:, None])
    return m


class JContainer:
    data = {}

    @classmethod
    def get(cls, device):
        if str(device) in cls.data:
            return cls.data[str(device)]

        from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense \
            import Jd as Jd_np

        device_data = [torch.tensor(J, dtype=torch.float32, device=device)
                       for J in Jd_np]
        cls.data[str(device)] = device_data

        return device_data


def wigner_d_matrix(angles, degree):
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    J = JContainer.get(angles.device)[degree][None]
    x_a = _z_rot_mat(angles[:, 0], degree)
    x_b = _z_rot_mat(angles[:, 1], degree)
    x_c = _z_rot_mat(angles[:, 2], degree)
    return x_a.matmul(J).matmul(x_b).matmul(J).matmul(x_c)


def block_wigner_matrix_multiply(angles, data, max_degree):
    """Transform data using wigner d matrices for all degrees.

    vector_dim is dictated by max_degree by the expression:
    vector_dim = \sum_{i=0}^max_degree (2 * max_degree + 1) = (max_degree+1)^2

    The representation is the direct sum of the irreps of the degrees up to max.
    The computation is equivalent to a block-wise matrix multiply.

    The data are the Fourier modes of a R^{data_dim} signal.

    Input:
    - angles (batch, 3)  ZYZ Euler angles
    - vector (batch, vector_dim, data_dim)

    Output: (batch, vector_dim, data_dim)
    """
    outputs = []
    start = 0
    for degree in range(max_degree+1):
        dim = 2 * degree + 1
        matrix = wigner_d_matrix(angles, degree)
        outputs.append(matrix.bmm(data[:, start:start+dim, :]))
        start += dim
    return torch.cat(outputs, 1)


def random_quaternions(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1-u1) * torch.sin(2 * math.pi * u2),
        torch.sqrt(1-u1) * torch.cos(2 * math.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * math.pi * u3),
    ), 1)
